'''export OPENAI_API_KEY=sk-...'''
'''Example command: python src/classify_query_intent.py --org_name 'openai' --input_file query-scenario/query_scenarios_num10_rewritten.jsonl --query_template prompts/query_intent_labeling.md --scenario_template prompts/scenario_intent_labeling.md'''

import argparse
import json
import random
import openai
from openai import OpenAI
import ast
from tqdm import tqdm
import anthropic
import os 

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)

parser = argparse.ArgumentParser()
# parser.add_argument("--api_key", required=True)
parser.add_argument("--org_id", default="org-...") 
parser.add_argument("--org_name", required=True, help="choose either openai for gpt-x or anthropic for claude")
parser.add_argument("--input_file", type=str, required=True, help="Input JSONL file with scenarios")
parser.add_argument("--model", default="gpt-4o", choices=["gpt-4", "gpt-3.5-turbo", "gpt-4-1106-preview", "gpt-4o", "claude-sonnet"])
parser.add_argument("--max_tokens", type=int, default=1024)
parser.add_argument("--temperature", type=float, default=0.7)
parser.add_argument("--top_p", type=float, default=1.0)
# parser.add_argument("--output_file", type=str, required=True)
parser.add_argument("--query_template", type=str, required=True, help="Path to query-only instruction template file.")
parser.add_argument("--scenario_template", type=str, required=True, help="Path to scenario-based instruction template file.")


MODEL_COSTS = {
    "gpt-4": [0.00003, 0.00006], 
    "gpt-3.5-turbo": [0.0000015, 0.000002], 
    "gpt-4-1106-preview": [0.00001, 0.00003], 
    "gpt-4o": [0.000005, 0.000015], 
    "claude-sonnet": [0.000003, 0.000015]
}

def get_output_filenames(input_file):
    """Generate output filenames based on input filename pattern"""
    # Get directory and filename
    dir_name = os.path.dirname(input_file)
    base_name = os.path.basename(input_file)
    
    # Split filename and extension
    name, ext = os.path.splitext(base_name)
    
    # Generate new filenames
    output_file = os.path.join(dir_name, f"{name}_labeled{ext}")
    return output_file

def generate_prompt(template, query, scenario=None):
    if scenario:
        return f"{template}\n\nPlease classify the intent of the following query given the scenario:\nQuery: {query}\nScenario: {scenario}"
    return f"{template}\n\nPlease classify the following query:\n{query}"

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def completion_with_backoff(**kwargs):
    client = OpenAI()
    return client.chat.completions.create(**kwargs)

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def completion_with_backoff_anthropic(args, **kwargs):
    client = anthropic.Anthropic(
        api_key=args.api_key,
    )
    return client.messages.create(**kwargs)

def main():
    args = parser.parse_args()
    api_key = os.getenv('OPENAI_API_KEY')
    # openai.api_key = args.api_key
    openai.api_key = api_key
    openai.organization = args.org_id
    
    in_cost_per_token, out_cost_per_token = MODEL_COSTS[args.model]
    total_input_tokens, total_output_tokens = 0, 0
    
    with open(args.query_template, 'r') as f:
        query_template = f.read()
    with open(args.scenario_template, 'r') as f:
        scenario_template = f.read()
        
    output_file = get_output_filenames(args.input_file)
    with open(args.input_file, 'r') as fin, open(output_file, 'w') as fout:
        for line in tqdm(fin, desc="Processing instances"):
            instance = json.loads(line)
            # confirm the column names of the input file 
            query, benign_scenario, malicious_scenario = instance["query"], instance["benign_scenario"], instance["malicious_scenario"]
            # query, benign_scenario, malicious_scenario = instance["query"], instance["rewritten_benign_scenario"], instance["rewritten_malicious_scenario"]
            
            prompts = {
                "query": generate_prompt(query_template, query),
                "benign": generate_prompt(scenario_template, query, benign_scenario),
                "malicious": generate_prompt(scenario_template, query, malicious_scenario)
            }
            
            if args.org_name == 'openai':
                responses = {}
                for key, prompt in prompts.items():
                    if prompt:
                        response = completion_with_backoff(
                            model=args.model,
                            messages=[
                                {"role": "user", "content": prompt}
                            ],
                            max_tokens=args.max_tokens,
                            n=1, 
                            temperature=args.temperature
                        )
                        total_input_tokens += response.usage.prompt_tokens
                        total_output_tokens += response.usage.completion_tokens
                        responses[key] = response.choices[0].message.content
                        
            else:
                responses = {}
                for key, prompt in prompts.items():
                    if prompt: 
                        response = completion_with_backoff_anthropic(
                            args,
                            model="claude-3-5-sonnet-20240620",
                            max_tokens=args.max_tokens,
                            temperature=args.temperature,
                            messages=[
                                {"role": "user", "content": prompt}
                            ]
                        )
                        total_input_tokens += response.usage.input_tokens
                        total_output_tokens += response.usage.output_tokens
                        responses[key] = response.choices[0].message.content
            
            output_data = {**instance}
            for key, content in responses.items():
                try:
                    parsed_response = json.loads(content.replace("```json\n", "").replace("\n```", "").strip())
                    output_data[f"{key}_intent"] = parsed_response.get("intent", "Ambiguous")
                    output_data[f"{key}_reasoning"] = parsed_response.get("reasoning", "")
                    output_data['model'] = args.model
                except Exception as e:
                    print(f"Error processing query: {query}")
                    print(f"Error: {e}")
                    print(f"Response content: {content}")
                    
            fout.write(json.dumps(output_data) + "\n")
            fout.flush()
            
        print(f"Total input tokens: {total_input_tokens}")
        print(f"Total output tokens: {total_output_tokens}")
        print(f"Estimated cost: ${in_cost_per_token * total_input_tokens + out_cost_per_token * total_output_tokens}")

if __name__ == "__main__":
    main()
   
            
    

