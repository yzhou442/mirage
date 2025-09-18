#!/usr/bin/env python3
"""
vLLM offline performance benchmark for Qwen3 model comparison with MPK.
Uses the same prompt as demo_hopper for fair performance comparison.
"""

import torch
import time
import argparse
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


def create_prompt():
    """Create the same prompt used in demo_hopper for fair comparison."""
    # code_text = """import numpy as np
    #             import matplotlib.pyplot as plt

    #             # Calculate the average
    #             average_throughput = np.mean(tokens_per_sec_arr)
    #             print(f"Average Throughput: {average_throughput} tokens/sec")

    #             # Plotting the histogram
    #             plt.hist(tokens_per_sec_arr, bins=20, color='blue', edgecolor='black', alpha=0.7)
    #             plt.title('Histogram of Throughput Values')
    #             plt.xlabel('Tokens per Second')
    #             plt.ylabel('Frequency')
    #             plt.axvline(average_throughput, color='red', linestyle='dashed', linewidth=1)
    #             plt.text(average_throughput*0.9, max(plt.ylim())*0.9, f'Average: {average_throughput:.2f}', color = 'red')
    #             plt.show()
    #             """
    # question = "Can you please change x axis to start from 0"
    # prompt = code_text + "\n" + question
    prompt = "Give me a short introduction to large language model."
    
    # Format as chat messages like in demo_hopper
    messages = [
        {
            "role": "system",
            "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
        },
        {"role": "user", "content": prompt},
    ]
    
    return messages


def benchmark_vllm(model_name: str, output_length: int = 512, warmup_runs: int = 3):
    """Benchmark vLLM performance with the specified model."""
    print(f"Initializing vLLM with model: {model_name}")
    
    # Initialize vLLM engine
    llm = LLM(
        model=model_name,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        max_model_len=4096,
    )
    
    # Initialize tokenizer for chat template
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Create prompt
    messages = create_prompt()
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    
    # Tokenize to get prompt length
    tokenized = tokenizer(text, return_tensors="pt")
    prompt_length = tokenized.input_ids.shape[1]
    
    print(f"Prompt length: {prompt_length} tokens")
    
    # Set sampling parameters
    sampling_params = SamplingParams(
        temperature=0.0,  # Deterministic generation for consistent comparison
        max_tokens=output_length,
        stop_token_ids=[tokenizer.eos_token_id] if hasattr(tokenizer, 'eos_token_id') else None,
    )
    
    # Warmup runs
    print(f"Running {warmup_runs} warmup iterations...")
    for i in range(warmup_runs):
        _ = llm.generate([text], sampling_params)
        print(f"Warmup {i+1}/{warmup_runs} completed")
    
    # Benchmark run
    print("Starting benchmark run...")
    torch.cuda.synchronize()
    start_time = time.time()
    
    outputs = llm.generate([text], sampling_params)
    
    torch.cuda.synchronize()
    end_time = time.time()
    
    # Calculate metrics
    total_time = (end_time - start_time) * 1000  # Convert to milliseconds
    output_text = outputs[0].outputs[0].text
    # generated_tokens = len(tokenizer.encode(output_text))
    generated_tokens = len(outputs[0].outputs[0].token_ids)
    
    per_token_time = total_time / generated_tokens if generated_tokens > 0 else 0
    tokens_per_second = generated_tokens / (total_time / 1000) if total_time > 0 else 0
    
    # Print results
    print("\n" + "="*60)
    print("vLLM BENCHMARK RESULTS")
    print("="*60)
    print(f"Model: {model_name}")
    print(f"Prompt length: {prompt_length} tokens")
    print(f"Generated tokens: {generated_tokens}")
    print(f"Total generation time: {total_time:.2f} ms")
    print(f"Per-token time: {per_token_time:.2f} ms/token")
    print(f"Throughput: {tokens_per_second:.2f} tokens/sec")
    print("="*60)
    
    # Print sample output (first 200 chars)
    print(f"\nGenerated text preview:")
    print(f"{output_text[:200]}{'...' if len(output_text) > 200 else ''}")
    
    return {
        'model': model_name,
        'prompt_length': prompt_length,
        'generated_tokens': generated_tokens,
        'total_time_ms': total_time,
        'per_token_time_ms': per_token_time,
        'tokens_per_second': tokens_per_second,
        'output_text': output_text
    }


def main():
    parser = argparse.ArgumentParser(description="vLLM benchmark for Qwen3 model")
    parser.add_argument(
        "--model", 
        type=str, 
        default="Qwen/Qwen3-8B",
        help="Model name or path (default: Qwen/Qwen3-8B)"
    )
    parser.add_argument(
        "--output-length",
        type=int,
        default=512,
        help="Maximum output length in tokens (default: 512)"
    )
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=3,
        help="Number of warmup runs (default: 3)"
    )
    
    args = parser.parse_args()
    
    try:
        results = benchmark_vllm(
            model_name=args.model,
            output_length=args.output_length,
            warmup_runs=args.warmup_runs
        )
        
        # Save results to file
        import json
        output_file = f"vllm_benchmark_results_{int(time.time())}.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_file}")
        
    except Exception as e:
        print(f"Error during benchmark: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
