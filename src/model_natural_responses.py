'''export OPENAI_API_KEY=sk-...'''
'''Example command: python src/model_natural_responses.py --org_name 'openai' --input_file model-responses/CoCoNot/coconot_test_contrast_safety.jsonl'''
'''python src/model_natural_responses.py --org_name 'together' --model 'meta-llama/Llama-3.3-70B-Instruct-Turbo, meta-llama/Llama-3.1-8B-Instruct-Turbo, Qwen/Qwen2.5-72B-Instruct-Turbo, mistralai/Mistral-7B-Instruct-v0.3' --input_file query-scenario/valid_quorum_queries_all.jsonl'''
'''python src/model_natural_responses.py --org_name 'huggingface' --model 'allenai/OLMo-2-0325-32B' --input_file query-scenario/valid_quorum_queries_all.jsonl --use_vllm --tensor_parallel_size 2'''
'''input file for casebench: data/case_bench_formatted.jsonl'''
import argparse
import json
import os 
HUGGINGFACE_CACHE = '/usr1/data/models_cache'
HF_HOME = HUGGINGFACE_CACHE
HF_DATASETS_CACHE = HUGGINGFACE_CACHE
TRANSFORMERS_CACHE = HUGGINGFACE_CACHE
os.environ['HF_HOME'] = HF_HOME
os.environ['HF_DATASETS_CACHE'] = HF_DATASETS_CACHE
os.environ['HF_TOKEN'] = 'hf_...'
from tqdm import tqdm
import pandas as pd 
import openai
# from openai import AsyncOpenAI
from openai import OpenAI
import ast
from anthropic import Anthropic
from together import Together
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
from vllm import LLM, SamplingParams

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)
os.environ["OPENAI_API_KEY"] = "sk-..."
os.environ["TOGETHER_API_KEY"] = "tgp_..."

parser = argparse.ArgumentParser()
# parser.add_argument("--api_key", required=True)
parser.add_argument("--org_id", default="org-...") 
parser.add_argument("--org_name", required=True, help="choose either openai for gpt-x, anthropic for claude, together for open models, or huggingface for HF models")
parser.add_argument("--input_file", type=str, required=True, help="Input JSONL file with scenarios")
parser.add_argument("--model", type=str, help="Comma-separated list of models to use. Available models: gpt-4, gpt-3.5-turbo, gpt-4-1106-preview, gpt-4o, claude-sonnet, meta-llama/Llama-2-70b-hf, meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo, mistralai/Mistral-7B-Instruct-v0.3, meta-llama/Llama-3.3-70B-Instruct-Turbo, Qwen/Qwen2.5-72B-Instruct-Turbo")
parser.add_argument("--max_tokens", type=int, default=1024)
parser.add_argument("--temperature", type=float, default=0.7)
parser.add_argument("--top_p", type=float, default=1.0)
parser.add_argument("--instruction_template", type=str, default='prompts/strategy_classification.md')
parser.add_argument("--use_vllm", action="store_true", help="Use vLLM for faster inference")
parser.add_argument("--tensor_parallel_size", type=int, default=1, help="Number of GPUs to use for tensor parallelism with vLLM")
# parser.add_argument("--output_dir", type=str, required=True, help="Directory for output files")

def get_output_filenames(input_file, model_name):
    """Generate output filenames based on input filename pattern"""
    # Get base filename
    base_name = os.path.basename(input_file)
    
    # Split filename and extension
    name, ext = os.path.splitext(base_name)
    if name == "valid_quorum_queries_all":
        name = "blind_refusal"
    if name == "case_bench_formatted":
        name = "case_bench"
    
    # Clean model name for filename (replace slashes with underscores)
    model_name = model_name.replace('/', '_')
    
    # Create model-responses directory if it doesn't exist
    output_dir = "model-responses"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate new filename in model-responses directory
    output_file = os.path.join(output_dir, f"{name}_{model_name}{ext}")
    
    return output_file

def check_and_confirm_overwrite(file_path):
    """Check if file exists and ask for confirmation before overwriting."""
    if os.path.exists(file_path):
        response = input(f"File {file_path} already exists. Do you want to overwrite it? (y/n): ")
        if response.lower() != 'y':
            print(f"Skipping {file_path}")
            return False
        print(f"Overwriting {file_path}")
    return True

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
    "claude-sonnet": [0.000003, 0.000015],
}

total_input_tokens = 0
total_output_tokens = 0

def format_prompt_with_scenario(query, scenario):
    """Format the prompt to include the scenario in first person."""
    if scenario:
        # print(scenario)
        # Convert scenario to first person if it starts with "You are" or "You're"
        if scenario.lower().startswith("you are"):
            scenario = "I am" + scenario[7:]
        elif scenario.lower().startswith("you're"):
            scenario = "I am" + scenario[6:]
        return f"I want to ask: \"{query}\"\nContext: {scenario}\nPlease provide a response (limit to 150 words)."
    return f"{query} (limit your response to 150 words)"

def model_response(instance, args):
    global total_input_tokens, total_output_tokens
    
    # Get the base query
    query = instance['query'] if 'query' in instance else instance['prompt']
    
    # Check if we have scenarios
    has_scenarios = 'benign_scenario' in instance or 'malicious_scenario' in instance
    
    responses = []
    
    # Always get response for query without scenario
    prompt = format_prompt_with_scenario(query, None)
    response = get_model_completion(prompt, args)
    responses.append({
        "prompt": prompt,
        "query": query,
        "response": response,
        "category": instance.get("category", ""),
        "topic": instance.get("topic", ""),
        "scenario_type": "no_scenario",
        "model": args.model,
    })
    
    if has_scenarios:
        # Process both scenarios if available
        if 'benign_scenario' in instance:
            prompt = format_prompt_with_scenario(query, instance['benign_scenario'])
            response = get_model_completion(prompt, args)
            responses.append({
                "prompt": prompt,
                "query": query,
                "response": response,
                "category": instance.get("category", ""),
                "topic": instance.get("topic", ""),
                "scenario_type": "benign",
                "model": args.model,
            })
            
        if 'malicious_scenario' in instance:
            prompt = format_prompt_with_scenario(query, instance['malicious_scenario'])
            response = get_model_completion(prompt, args)
            responses.append({
                "prompt": prompt,
                "query": query,
                "response": response,
                "category": instance.get("category", ""),
                "topic": instance.get("topic", ""),
                "scenario_type": "malicious",
                "model": args.model,
            })
    
    return responses

def get_model_completion(prompt, args):
    """Get completion from the appropriate model."""
    global total_input_tokens, total_output_tokens
    if args.org_name == 'openai':
        client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'), organization=args.org_id)
        try:
            response = completion_with_backoff(
            client, 
            model=args.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=args.max_tokens,
            n=1, 
            temperature=args.temperature
            )
        except:
            response = completion_with_backoff(
            client, 
            model=args.model,
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=args.max_tokens,
            n=1
            )
        total_input_tokens += response.usage.prompt_tokens
        total_output_tokens += response.usage.completion_tokens
        return response.choices[0].message.content
    elif args.org_name == 'anthropic':
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
        return response.content[0].text
    elif args.org_name == 'together':
        client = Together()
        messages = [{"role": "user", "content": prompt}]
            
        response = client.chat.completions.create(
            model=args.model,
            messages=messages,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p
        )
        return response.choices[0].message.content
    elif args.org_name == 'huggingface':
        # Initialize the model and tokenizer if not already initialized
        if not hasattr(get_model_completion, 'model'):
            print(f"Loading model {args.model}...")
            
            if args.use_vllm:
                print("Using vLLM for faster inference...")
                get_model_completion.model = LLM(
                    model=args.model,
                    tensor_parallel_size=args.tensor_parallel_size,
                    trust_remote_code=True,
                    dtype="float16"
                )
                get_model_completion.tokenizer = get_model_completion.model.get_tokenizer()
                get_model_completion.sampling_params = SamplingParams(
                    temperature=args.temperature,
                    top_p=args.top_p,
                    max_tokens=args.max_tokens
                )
            else:
                get_model_completion.tokenizer = AutoTokenizer.from_pretrained(args.model)
                # Configure quantization
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    bnb_4bit_compute_dtype=torch.float16
                )
                
                get_model_completion.model = AutoModelForCausalLM.from_pretrained(
                    args.model,
                    device_map="auto",
                    torch_dtype=torch.float16,
                    quantization_config=quantization_config
                )
            print("Model loaded successfully!")
        
        # Generate response
        if args.use_vllm:
            outputs = get_model_completion.model.generate(
                prompt,
                sampling_params=get_model_completion.sampling_params
            )
            response_text = outputs[0].outputs[0].text
        else:
            # Tokenize the input
            inputs = get_model_completion.tokenizer(prompt, return_tensors="pt").to(get_model_completion.model.device)
            
            # Generate response
            with torch.no_grad():
                outputs = get_model_completion.model.generate(
                    **inputs,
                    max_new_tokens=args.max_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    do_sample=True,
                    pad_token_id=get_model_completion.tokenizer.eos_token_id
                )
            
            # Decode the generated text
            generated_text = get_model_completion.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Remove the input prompt from the generated text
            response_text = generated_text[len(prompt):].strip()
        
        # Get exact token counts
        input_tokens = len(get_model_completion.tokenizer.encode(prompt))
        output_tokens = len(get_model_completion.tokenizer.encode(response_text))
        total_input_tokens += input_tokens
        total_output_tokens += output_tokens
        
        return response_text

def classify_response(instance, instruction_template, args):
    """Classify a model response using the instruction template."""
    global total_input_tokens, total_output_tokens
    
    # Prepare the prompt with the current scenario
    prompt = instruction_template.replace("{$prompt}", instance['prompt']).replace("{$response}", instance['response'])
    
    # Always use gpt-4o for classification
    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'), organization=args.org_id)
    response = completion_with_backoff(
        client, 
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=args.max_tokens,
        n=1, 
        temperature=args.temperature
    )
    total_input_tokens += response.usage.prompt_tokens
    total_output_tokens += response.usage.completion_tokens
    response_content = response.choices[0].message.content
    
    response_content = response_content.replace("```json\n", "").replace("\n```", "").strip()
    # Parse the response
    try:
        classification = json.loads(response_content)
        # Combine original instance with classification
        combined = {**instance, **classification}
        return combined
    except Exception as e:
        print(f"Error parsing classification for query {instance['prompt']}: {e}")
        print(f"Response content: {response_content}")
        return instance  # Return original instance if classification fails

def main():
    args = parser.parse_args()
    global total_input_tokens, total_output_tokens
    # Parse models from comma-separated string
    models = [m.strip() for m in args.model.split(',')]
    
    # Read instruction template for classification
    with open(args.instruction_template, 'r') as f:
        instruction_template = f.read()
    
    # Process each model
    for model in models:
        print(f"\nProcessing with model: {model}")
        args.model = model  # Set current model in args
        output_file = get_output_filenames(args.input_file, model)
        
        # Check if file exists and get confirmation
        if not check_and_confirm_overwrite(output_file):
            continue
        
        # Store all results
        all_results = []
        
        # Process scenarios
        with open(args.input_file, 'r') as fin, \
             open(output_file, 'w') as fout:
                 
            for line in tqdm(fin):
                instance = json.loads(line.strip())
                responses = model_response(instance, args)
                
                # Process each response
                for response in responses:
                    # Classify the response and combine with original data
                    combined_result = classify_response(response, instruction_template, args)
                    all_results.append(combined_result)
                    fout.write(json.dumps(combined_result) + '\n')
                
                fout.flush()
        
        try: 
            in_cost_per_token, out_cost_per_token = MODEL_COSTS[model]
            estimated_cost = in_cost_per_token * total_input_tokens + out_cost_per_token * total_output_tokens
        except KeyError:
            print(f"Model {model} not found in MODEL_COSTS")
            estimated_cost = "N/A"
        
        print(f"\nProcessing complete for model: {model}")
        print(f"Input file: {args.input_file}")
        print(f"Results saved to: {output_file}")
        print(f"Total input tokens: {total_input_tokens}")
        print(f"Total output tokens: {total_output_tokens}")
        if estimated_cost != "N/A":
            print(f"Estimated cost: ${estimated_cost:.4f}")
        else:
            print(f"Unknown estimated cost")
        
        # Reset token counters for next model
        total_input_tokens = 0
        total_output_tokens = 0

if __name__ == "__main__":
    main()