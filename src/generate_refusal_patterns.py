'''
Example command: python generate_refusal_patterns.py --api_key XXX --input_file ../evaluation_data/final_evaluation_data.jsonl --output_file ./coconot_refusal_patterns_responses.jsonl --model gpt-4o --org_name openai --instruction_template templates/refusal_patterns.md
'''
import argparse
from dis import Instruction
import json
import random
import openai
import ast
from tqdm import tqdm
import anthropic

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for

# random.seed(40281)

parser = argparse.ArgumentParser()
parser.add_argument("--api_key", required=True)
parser.add_argument("--org_id", default="org-...") 
parser.add_argument("--org_name", required=True, help="choose either openai for gpt-x or anthropic for claude")
parser.add_argument("--input_file", type=str, required=True)
parser.add_argument("--model", default="gpt-3.5-turbo", choices=["gpt-4", "gpt-3.5-turbo", "gpt-4-1106-preview", "gpt-4o", "claude-sonnet"])
parser.add_argument("--max_tokens", type=int, default=1024)
parser.add_argument("--temperature", type=float, default=0.7)
parser.add_argument("--top_p", type=float, default=1.0)
parser.add_argument("--output_file", type=str, required=True)
parser.add_argument("--instruction_template", type=str, required=True)

args = parser.parse_args()

openai.api_key = args.api_key
openai.organization = args.org_id
@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def completion_with_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def completion_with_backoff_anthropic(args, **kwargs):
    client = anthropic.Anthropic(
        api_key=args.api_key,
    )
    return client.messages.create(**kwargs)


MODEL_COSTS = {"gpt-4": [0.00003, 0.00006], "gpt-3.5-turbo": [0.0000015, 0.000002], "gpt-4-1106-preview": [0.00001, 0.00003], 'gpt-4o': [0.000005, 0.000015], 'claude-sonnet':[0.000003, 0.000015]}
in_cost_per_token, out_cost_per_token = MODEL_COSTS[args.model]

total_input_tokens = 0
total_output_tokens = 0


with open(args.instruction_template) as f:
    instruction_template = f.read()

with open(args.input_file) as fin, open(args.output_file, "w") as fout:
    for idx, line in tqdm(enumerate(fin)):
        # if idx > 50:
        #     break
        requestdata = json.loads(line)
        prompt = instruction_template.replace("{$prompt}", requestdata["prompt"])

        if args.org_name == 'openai':
            response = completion_with_backoff(
                model=args.model,
                messages = [
                    # {"role": "system", "content": system_instruction},
                    {"role": "user", "content": prompt}
                    ],
                max_tokens=args.max_tokens,
                n=1,
                temperature=args.temperature
            )
            

            total_input_tokens += response.usage.prompt_tokens
            total_output_tokens += response.usage.completion_tokens

            response = response.choices[0].message.content


        elif args.org_name == 'anthropic':
            response = completion_with_backoff_anthropic(args,
                model="claude-3-5-sonnet-20240620",
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                # system=system_instruction,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            


            total_input_tokens += response.usage.input_tokens
            total_output_tokens += response.usage.output_tokens
            
            response = response.content[0].text

        response = response.replace("```json\n", "").replace("\n```", "").strip()
        try:
            response = ast.literal_eval(response)
            requestdata.update(response)
            requestdata.update({"model": args.model})
            if "response" in requestdata:
                del requestdata["response"]
            fout.write(json.dumps(requestdata) + "\n")
            fout.flush()
        except:
            print("Could not parse the outputs!")


    print(f"{total_input_tokens=}")
    print(f"{total_output_tokens=}")
    print(f"estimated cost= ${in_cost_per_token * total_input_tokens + out_cost_per_token * total_output_tokens}")
