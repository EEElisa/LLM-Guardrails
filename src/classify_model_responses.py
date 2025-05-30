'''export OPENAI_API_KEY=sk-...'''
'''Example command: python src/classify_model_responses.py --org_name 'openai' --input_file model-responses/CoCoNot/coconot_test_contrast_safety-gpt-4o.jsonl'''

'''Example command: python src/classify_model_responses.py --org_name 'openai' --input_file model-responses/CoCoNot/coconot_test_contrast_safety_Qwen_Qwen2.5-72B-Instruct-Turbo.jsonl'''
import argparse
import json
import os 
from tqdm import tqdm
import pandas as pd 
import openai
# from openai import AsyncOpenAI
from openai import OpenAI
import ast
from anthropic import Anthropic

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
# parser.add_argument("--output_dir", type=str, required=True, help="Directory for output files")
parser.add_argument("--instruction_template", type=str, default='prompts/strategy_classification.md')

def get_output_filenames(input_file):
    """Generate output filenames based on input filename pattern"""
    # Get directory and filename
    dir_name = os.path.dirname(input_file)
    base_name = os.path.basename(input_file)
    
    # Split filename and extension
    name, ext = os.path.splitext(base_name)
    
    # Generate new filenames
    rewritten_file = os.path.join(dir_name, f"{name}_classification{ext}")
    
    return rewritten_file

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def completion_with_backoff(client, **kwargs):
    # client = OpenAI()
    return client.chat.completions.create(**kwargs)

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def completion_with_backoff_anthropic(args, **kwargs):
    client = Anthropic(
        api_key=args.api_key,
    )
    return client.messages.create(**kwargs)

MODEL_COSTS = {
    "gpt-4": [0.00003, 0.00006], 
    "gpt-3.5-turbo": [0.0000015, 0.000002], 
    "gpt-4-1106-preview": [0.00001, 0.00003], 
    "gpt-4o": [0.000005, 0.000015], 
    "claude-sonnet": [0.000003, 0.000015]
}

total_input_tokens = 0
total_output_tokens = 0

def model_response(instance, instruction_template, args):
    global total_input_tokens, total_output_tokens
    
    # Prepare the prompt with the current scenario
    prompt = instruction_template.replace("{$prompt}", instance['prompt']).replace("{$response}", instance['response'])
    
    if args.org_name == 'openai':
        client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'), organization=args.org_id)
        response = completion_with_backoff(
            client, 
            model=args.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=args.max_tokens,
            n=1, 
            temperature=args.temperature
        )
        total_input_tokens += response.usage.prompt_tokens
        total_output_tokens += response.usage.completion_tokens
        response_content = response.choices[0].message.content
    else:
        client = Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
        response = completion_with_backoff(
            client,
            model="claude-3-5-sonnet-20240620",
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            messages=[{"role": "user", "content": prompt}]
        )
        total_input_tokens += response.usage.input_tokens
        total_output_tokens += response.usage.output_tokens
        response_content = response.content[0].text
    
    response_content = response_content.replace("```json\n", "").replace("\n```", "").strip()
    # Parse the response
    try:
        classification = json.loads(response_content)
        # Preserve metadata from original scenario
        classification.update({
            "prompt": instance["prompt"],
            "response": instance["response"],
            "category": instance["category"],
            "subcategory": instance["subcategory"],
            "model": args.model,
        })
        return classification
    except Exception as e:
        print(f"Error parsing response for query {instance['prompt']}: {e}")
        print(f"Response content: {response_content}")
        return None
    
def main():
    args = parser.parse_args()
    
    output_file = get_output_filenames(args.input_file)
    
    # Read instruction template
    with open(args.instruction_template, 'r') as f:
        instruction_template = f.read()
    
    # Store responses
    query_responses = []
    
    # Process scenarios
    with open(args.input_file, 'r') as fin, \
         open(output_file, 'w') as fout:
             
        for line in tqdm(fin):
            instance = json.loads(line.strip())
            query_response = model_response(instance, instruction_template, args)
            if model_response:
                query_responses.append(query_response)
                fout.write(json.dumps(query_response) + '\n')
            fout.flush()
    
    in_cost_per_token, out_cost_per_token = MODEL_COSTS[args.model]
    estimated_cost = in_cost_per_token * total_input_tokens + out_cost_per_token * total_output_tokens
    
    print(f"\nProcessing complete!")
    print(f"Input file: {args.input_file}")
    print(f"Response classification saved to: {output_file}")
    print(f"Total input tokens: {total_input_tokens}")
    print(f"Total output tokens: {total_output_tokens}")
    print(f"Estimated cost: ${estimated_cost:.4f}")
    
if __name__ == "__main__":
    main()