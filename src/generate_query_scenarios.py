'''export OPENAI_API_KEY=sk-...'''
'''Example command: python src/generate_query_scenarios.py --org_name 'openai' --seed_examples prompts/snowballing_seed_examples.json --output_file query-scenario/query_scenarios_snowball.jsonl --instruction_template prompts/query_scenarios-v2.md --num_instances 10'''
import argparse
import json
import random
import openai
# from openai import AsyncOpenAI
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
parser.add_argument("--seed_examples", type=str, required=True, help="JSON file with seed examples")
parser.add_argument("--model", default="gpt-4o", choices=["gpt-4", "gpt-3.5-turbo", "gpt-4-1106-preview", "gpt-4o", "claude-sonnet"])
parser.add_argument("--max_tokens", type=int, default=1024)
parser.add_argument("--temperature", type=float, default=0.7)
parser.add_argument("--top_p", type=float, default=1.0)
parser.add_argument("--output_file", type=str, required=True)
parser.add_argument("--instruction_template", type=str, required=True)
parser.add_argument("--num_instances", type=int, default=10, help="Number of instances to generate per category")

def flexible_format(template, **kwargs):
    """
    A more robust string formatting method that replaces placeholders
    """
    for key, value in kwargs.items():
        # If the value is a dictionary, convert to a pretty-printed JSON string
        if isinstance(value, dict):
            value = json.dumps(value, indent=2)
        
        # Replace both {key} and exact string patterns
        template = template.replace(f"{{{key}}}", str(value))
    
    return template

args = parser.parse_args()
api_key = os.getenv('OPENAI_API_KEY')
# openai.api_key = args.api_key
openai.api_key = api_key
openai.organization = args.org_id

if os.path.exists(args.output_file):
    raise FileExistsError(f"Output file '{args.output_file}' already exists. Please choose a different filename or remove the existing file.")

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def completion_with_backoff(**kwargs):
    client = OpenAI()
    # return openai.ChatCompletion.create(**kwargs)
    return client.chat.completions.create(**kwargs)

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def completion_with_backoff_anthropic(args, **kwargs):
    client = anthropic.Anthropic(
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
in_cost_per_token, out_cost_per_token = MODEL_COSTS[args.model]

total_input_tokens = 0
total_output_tokens = 0

with open(args.instruction_template, 'r') as f:
    instruction_template = f.read()

# Load seed examples
try:
    # first try to load as json list
    with open(args.seed_examples, 'r') as f:
        seed_examples = json.load(f)
except json.JSONDecodeError:
    # if that fails, load as jsonl
    with open(args.seed_examples, 'r') as f:
        seed_examples = [json.loads(line) for line in f if line.strip()] # skip empty lines

with open(args.output_file, "w") as fout:
    for seed_example in seed_examples:
        # Prepare the specific category prompt
        category = seed_example['category']
        
        # Format the prompt
        # prompt = instruction_template.format(
        #     category=category,
        #     num_instances=args.num_instances,
        #     seed_example=json.dumps(seed_example, indent=2)
        # )
        prompt = flexible_format(
            instruction_template, 
            category=category,
            num_instances=args.num_instances,
            seed_example=seed_example
        )
        
        if args.org_name == 'openai':
            response = completion_with_backoff(
                model=args.model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=args.max_tokens,
                n=1,
                temperature=args.temperature
            )
            # print(response)
            total_input_tokens += response.usage.prompt_tokens
            total_output_tokens += response.usage.completion_tokens
            response_content = response.choices[0].message.content
        elif args.org_name == 'anthropic':
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
            response_content = response.content[0].text
            
        response_content = response_content.replace("```json\n", "").replace("\n```", "").strip()
        try:
            # Try to parse the response as a list of dictionaries
            scenarios = ast.literal_eval(response_content)
            
            # Write each instance to the output file
            for scenario in scenarios:
                # Add metadata to the scenario
                scenario.update({
                    "category": category,
                    "model": args.model
                })
                fout.write(json.dumps(scenario) + "\n")
            
            fout.flush()
        
        except Exception as e:
            print(f"Could not parse the outputs for category {category}!")
            print(f"Error: {e}")
            print(f"Response content: {response_content}")
        
        print(f"{total_input_tokens=}")
        print(f"{total_output_tokens=}")
        print(f"estimated cost= ${in_cost_per_token * total_input_tokens + out_cost_per_token * total_output_tokens}")

