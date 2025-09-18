#!/usr/bin/env python3
import time, json, argparse
import torch
import sglang as sgl
from transformers import AutoTokenizer

def create_messages():
    return [
        {"role": "system",
         "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
        {"role": "user",
         "content": "Give me a short introduction to large language model."},
    ]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="Qwen/Qwen3-8B",
                    help="HF repo or local path, e.g. Qwen/Qwen3-8B or Qwen/Qwen3-8B-Instruct")
    ap.add_argument("--output-length", type=int, default=512)
    ap.add_argument("--warmup-runs", type=int, default=1)
    # to manually specify precision, add: --dtype bfloat16 / float16; if not specified, use default
    ap.add_argument("--dtype", type=str, default=None,
                    help="Optional: bfloat16 / float16 / float32")
    args = ap.parse_args()

    # 1) start offline engine
    #    no extra optional parameters, reduce compatibility issues; if needed, pass dtype="bfloat16"/"float16"
    engine_kwargs = dict(model_path=args.model)
    if args.dtype:  # bfloat16 / float16
        engine_kwargs["dtype"] = args.dtype
    llm = sgl.Engine(**engine_kwargs)  # offline engine, built-in continuous batch processing, paged attention optimization

    # 2) use HF tokenizer to apply chat template
    tok = AutoTokenizer.from_pretrained(args.model)
    text = tok.apply_chat_template(create_messages(), tokenize=False, add_generation_prompt=True)

    # 3) warmup
    sampling = {"temperature": 0.0, "max_new_tokens": args.output_length}
    for _ in range(args.warmup_runs):
        _ = llm.generate([text], sampling)

    # 4) time benchmark
    if torch.cuda.is_available(): torch.cuda.synchronize()
    t0 = time.time()
    outs = llm.generate([text], sampling) 
    if torch.cuda.is_available(): torch.cuda.synchronize()
    t1 = time.time()

    out_text = outs[0]["text"]
    gen_tokens = len(tok(out_text, add_special_tokens=False).input_ids)
    total_ms = (t1 - t0) * 1000.0
    tps = gen_tokens / (t1 - t0) if (t1 - t0) > 0 else 0.0

    print("\n" + "="*60)
    print("SGLANG OFFLINE BENCHMARK (Engine.generate)")
    print("="*60)
    print(f"Model: {args.model}")
    print(f"Generated tokens: {gen_tokens}")
    print(f"Total generation time: {total_ms:.2f} ms")
    print(f"Per-token time: {total_ms / gen_tokens:.2f} ms/token")
    print(f"Throughput: {tps:.2f} tokens/sec")
    print("="*60)
    print("\nGenerated text preview:")
    print(out_text[:200] + ("..." if len(out_text) > 200 else ""))

    with open(f"sglang_bench_{int(time.time())}.json", "w") as f:
        json.dump({
            "model": args.model,
            "generated_tokens": gen_tokens,
            "total_time_ms": total_ms,
            "tokens_per_second": tps,
            "output_text": out_text,
        }, f, indent=2)

if __name__ == "__main__":
    main()
