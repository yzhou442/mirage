from models.modeling_qwen3 import Qwen3ForCausalLM
from transformers import AutoTokenizer, AutoConfig
from safetensors.torch import load_model
import torch
import torch.distributed as dist
import argparse
import os

# print limitation
torch.set_printoptions(profile="full")

def grid_for_rmsnorm_linear_layer(size):
    # 96 and 64 are enough to cover all Qwen3 model? Please update the method
    # if you meet any incompatibility.
    if size % 96 == 0:
        return 96
    elif size % 64 == 0:
        return 64
    
# Return the largest factor of m that is less than or equal to n
# This is used to determine the grid size
def max_factor_leq_n(m: int, n: int) -> int:
    max_factor = 1
    i = 1
    while i * i <= m:
        if m % i == 0:
            if i <= n:
                max_factor = max(max_factor, i)
            if m // i <= n:
                max_factor = max(max_factor, m // i)
        i += 1
    return max_factor

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-mirage", action="store_true", help="Use Mirage kernels")
    parser.add_argument("--max-num-batched-tokens", default=8, type=int, help="Max number of tokens in a batch")
    parser.add_argument("--max-num-batched-requests", default=1, type=int, help="Max number of requests in a batch")
    parser.add_argument("--page-size", default=4096, type=int, help="Page size")
    parser.add_argument("--max-num-pages", default=16, type=int, help="Max num pages")
    parser.add_argument("--output-dir", default="output", help="Output files directory")
    parser.add_argument("--trace-name", default="qwen3", help="Perfetto trace output name")
    parser.add_argument(
        "--profiling", action="store_true", help="Use Profiler to generate trace"
    )
    # lookahead or promptlookup
    parser.add_argument(
        "--spec-decode",
        default=None,
        choices=["promptlookup", "lookahead"],
        help="Enable speculative decoding with 'lookahead' or 'promptlookup' mode.",
    )
    parser.add_argument(
        "--ngram-size",
        default=3,
        type=int,
        help="Ngram size for lookahead spec decode",
    )
    parser.add_argument(
        "--max-seq-length",
        default=1,
        type=int,
        help="Max sequence length for lookahead spec decode",
    )
    parser.add_argument(
        "--spec-length",
        default=3,
        type=int,
        help="Spec length for lookahead spec decode",
    )

    parser.add_argument("--model-path", type=str, default=None, help="Path to a local model (necessary for multi-GPU demo)")
    parser.add_argument(
        "--model", type=str, default='Qwen/Qwen3-8B', help="Model path on hugging face"
    )
    args = parser.parse_args()
    try:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        world_size = comm.Get_size()
        rank = comm.Get_rank()
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12355"
    except ImportError:
        world_size = 1
        rank = 0

    if world_size > 1:
        dist.init_process_group(backend="nccl", init_method="env://")
    global print
    if rank != 0:
        print = lambda *_, **__: None

    print("Input arguments:", args)
    print(f"world_size({world_size}) rank({rank})")
    model_name = args.model
    torch.set_default_dtype(torch.bfloat16)

    torch.cuda.set_device(rank)
    with torch.device("cuda"):
        if args.model_path is not None:
            # load model locally (necessary for multi-GPU case)
            print(f"Load model from model path: {args.model_path}")
            config = AutoConfig.from_pretrained(args.model_path)
            # model = Qwen3ForCausalLM(config, world_size, args.max_num_pages, args.page_size)
            # load_model(
            #     model, f"{args.model_path}/model{rank}-mp{world_size}.safetensors"
            # )
            model = Qwen3ForCausalLM.from_pretrained(args.model_path, world_size, max_num_pages=args.max_num_pages, page_size=args.page_size).to("cuda")
            tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        else:
            model = Qwen3ForCausalLM.from_pretrained(model_name, world_size=1, max_num_pages=args.max_num_pages, page_size=args.page_size).to("cuda")
            tokenizer = AutoTokenizer.from_pretrained(model_name)

    total_num_requests = 1
    # get all model weight tensors
    tokens = torch.full((total_num_requests, args.max_seq_length), 0, dtype=torch.long, device="cuda")

    # prompt = "Give me a short introduction to large language model."
    # This prompt is copied from https://github.com/apoorvumang/prompt-lookup-decoding/blob/main/demo-pld.ipynb
    code_text = """import numpy as np
                import matplotlib.pyplot as plt

                # Calculate the average
                average_throughput = np.mean(tokens_per_sec_arr)
                print(f"Average Throughput: {average_throughput} tokens/sec")

                # Plotting the histogram
                plt.hist(tokens_per_sec_arr, bins=20, color='blue', edgecolor='black', alpha=0.7)
                plt.title('Histogram of Throughput Values')
                plt.xlabel('Tokens per Second')
                plt.ylabel('Frequency')
                plt.axvline(average_throughput, color='red', linestyle='dashed', linewidth=1)
                plt.text(average_throughput*0.9, max(plt.ylim())*0.9, f'Average: {average_throughput:.2f}', color = 'red')
                plt.show()
                """
    question = "Can you please change x axis to start from 0"
    prompt = code_text + "\n" + question
    messages = [
        {
            "role": "system",
            "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
        },
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    # for r in range(total_num_requests):
    #     for i in range(model_inputs.input_ids.shape[-1]):
            # tokens[r, i] = model_inputs.input_ids[0, i]
    prompt_lengths = torch.full((total_num_requests,), 8, dtype=torch.int, device="cuda")
    positions = torch.arange(32768).unsqueeze(0).to(model.device)
    position_embeddings = model.model.rotary_emb(positions)

    # get all model weight tensors
    input_tokens = torch.full((args.max_num_batched_tokens, 1), 0, dtype=torch.long, device="cuda")
    output_tokens = torch.full((args.max_num_batched_tokens, 1), 0, dtype=torch.long, device="cuda")
    prev_pos = 0

    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
        enable_timing=True
    )
    step = torch.full((total_num_requests, ), 0, dtype=torch.int32, device="cuda")
    num_new_tokens = torch.full((total_num_requests, ), 1, dtype=torch.int32, device="cuda")

    if args.use_mirage:
        import mirage as mi

        hidden_size = model.config.hidden_size
        intermediate_size = model.config.intermediate_size
        # pad vocab_size to facilitate task graph creation
        lm_head_weight = torch.cat(
            (
                model.lm_head.weight,
                torch.full(
                    (153600 - model.config.vocab_size, hidden_size), 0, device="cuda"
                ),
            ),
            0,
        )
        assert lm_head_weight.stride()[0] == hidden_size
        vocab_size = 153600
        num_q_heads = model.config.num_attention_heads
        num_kv_heads = model.config.num_key_value_heads
        num_local_q_heads = num_q_heads // world_size
        num_local_kv_heads = num_kv_heads // world_size
        head_dim = model.config.head_dim
        fused_outdim_1 = (num_q_heads + 2 * num_kv_heads) * head_dim
        fused_outdim_2 = 2 * intermediate_size

        if args.profiling:
            profiler_tensor = torch.zeros(
                3000 * 128, dtype=torch.uint64, device="cuda"
            ).contiguous()
        else:
            profiler_tensor = None
            
        spec_decode_config = mi.speculative.spec_decode_class(
            args.spec_decode,
            ngram_size=args.ngram_size,
            spec_length=args.spec_length,
        )
            
        num_workers, num_schedulers = mi.get_configurations_from_gpu(rank)
        print("num_workers: ", num_workers)
        print("num_schedulers: ", num_schedulers)
        qo_indptr_buffer = torch.empty(
            args.max_num_batched_requests + 1, dtype=torch.int32, device="cuda")
        paged_kv_indptr_buffer = torch.empty(
            args.max_num_batched_requests + 1, dtype=torch.int32, device="cuda")
        paged_kv_indices_buffer = torch.empty(
            args.max_num_pages, dtype=torch.int32, device="cuda")
        paged_kv_last_page_len_buffer = torch.empty(
            args.max_num_batched_requests, dtype=torch.int32, device="cuda")
        mpk = mi.PersistentKernel(
            mode="offline",
            world_size=world_size,
            mpi_rank=rank,
            num_workers=num_workers,
            num_local_schedulers=num_schedulers,
            num_remote_schedulers=0,
            max_seq_length=args.max_seq_length,
            max_num_batched_requests=args.max_num_batched_requests,
            max_num_batched_tokens=args.max_num_batched_tokens,
            max_num_pages=args.max_num_pages,
            page_size=args.page_size,
            eos_token_id=model.config.eos_token_id,
            meta_tensors={
                "step": step,
                "tokens": tokens,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "num_new_tokens": num_new_tokens,
                "prompt_lengths": prompt_lengths,
                "qo_indptr_buffer": qo_indptr_buffer,
                "paged_kv_indptr_buffer": paged_kv_indptr_buffer,
                "paged_kv_indices_buffer": paged_kv_indices_buffer,
                "paged_kv_last_page_len_buffer": paged_kv_last_page_len_buffer,
            },
            profiler_tensor=profiler_tensor,
            trace_name=args.trace_name,
            spec_decode_config=spec_decode_config,
            use_cutlass_kernel=False,
        )
        
        # x_torch = torch.full((8, 4096), 0.5, dtype=torch.bfloat16, device="cuda")
        # w_torch = torch.full((4096, 4096), 0.1, dtype=torch.bfloat16, device="cuda")
        # attn_out_torch = torch.full((8, 4096), 0.1, dtype=torch.bfloat16, device="cuda")
        # attn_proj_out_torch = torch.full((8, 4096), 0.0, dtype=torch.bfloat16, device="cuda")

        x_torch = torch.randn((8, 4096), dtype=torch.bfloat16, device="cuda")
        w_torch = torch.randn((4096, 4096), dtype=torch.bfloat16, device="cuda")
        attn_out_torch = torch.randn((8, 4096), dtype=torch.bfloat16, device="cuda")
        attn_proj_out_torch = torch.randn((8, 4096), dtype=torch.bfloat16, device="cuda")

        x = mpk.attach_input(torch_tensor=x_torch, name="input_x")
        w = mpk.attach_input(torch_tensor=w_torch, name="layer_0_qkv_proj")
        attn_out = mpk.attach_input(torch_tensor=attn_out_torch, name="layer_0_attn_out")
        attn_proj_out = mpk.attach_input(torch_tensor=attn_proj_out_torch, name="layer_0_attn_proj_out")

        attn_proj_out = x
        mpk.splitk_linear_layer(
            input=attn_out,
            weight=w,
            output=attn_proj_out,
            grid_dim=(64, 1, 1),
            block_dim=(256, 1, 1),
        )

        print("id(x):", id(x))
        print("id(attn_proj_out):", id(attn_proj_out))

        results = mpk.kn_graph.generate_task_graph(num_gpus=world_size, my_gpu_id=rank)
        with open(f"task_graph_{rank}.json", "w") as f:
            f.write(results["json_file"])
        with open(f"kernel_{rank}.cu", "w") as f:
            f.write(results["cuda_code"])

        mpk.compile(output_dir=args.output_dir)




        qo_indptr_buffer_2 = qo_indptr_buffer.clone()
        paged_kv_indptr_buffer_2 = paged_kv_indptr_buffer.clone()
        paged_kv_indices_buffer_2 = paged_kv_indices_buffer.clone()
        paged_kv_last_page_len_buffer_2 = paged_kv_last_page_len_buffer.clone()
        # ref with original paged attention layer
        mpk2 = mi.PersistentKernel(
            mode="offline",
            world_size=world_size,
            mpi_rank=rank,
            num_workers=num_workers,
            num_local_schedulers=num_schedulers,
            num_remote_schedulers=0,
            max_seq_length=args.max_seq_length,
            max_num_batched_requests=args.max_num_batched_requests,
            max_num_batched_tokens=args.max_num_batched_tokens,
            max_num_pages=args.max_num_pages,
            page_size=args.page_size,
            eos_token_id=model.config.eos_token_id,
            meta_tensors={
                "step": step.clone(),
                "tokens": tokens.clone(),
                "input_tokens": input_tokens.clone(),
                "output_tokens": output_tokens.clone(),
                "num_new_tokens": num_new_tokens.clone(),
                "prompt_lengths": prompt_lengths.clone(),
                "qo_indptr_buffer": qo_indptr_buffer_2,
                "paged_kv_indptr_buffer": paged_kv_indptr_buffer_2,
                "paged_kv_indices_buffer": paged_kv_indices_buffer_2,
                "paged_kv_last_page_len_buffer": paged_kv_last_page_len_buffer_2,
            },
            profiler_tensor=profiler_tensor,
            trace_name=args.trace_name + "_96",
            spec_decode_config=spec_decode_config,
            use_cutlass_kernel=False
        )

        x_torch_2 = x_torch.clone()
        w_torch_2 = w_torch.clone()
        attn_out_torch_2 = attn_out_torch.clone()
        attn_proj_out_torch_2 = attn_proj_out_torch.clone()

        x_2 = mpk2.attach_input(torch_tensor=x_torch_2, name="input_x_2")
        w_2 = mpk2.attach_input(torch_tensor=w_torch_2, name="layer_0_qkv_proj_2")
        attn_out_2 = mpk2.attach_input(torch_tensor=attn_out_torch_2, name="layer_0_attn_out_2")
        attn_proj_out_2 = mpk2.attach_input(torch_tensor=attn_proj_out_torch_2, name="layer_0_attn_proj_out_2")

        mpk2.linear_with_residual_layer(
            input=attn_out_2,
            weight=w_2,
            residual=x_2,
            output=attn_proj_out_2,
            grid_dim=(1, 1, 1),
            block_dim=(256, 1, 1),
        )
        mpk2.compile(output_dir=args.output_dir + "_2")

    starter.record()
    mpk()
    ender.record()
    torch.cuda.synchronize()
    run_time = starter.elapsed_time(ender)

    starter.record()
    mpk2()
    ender.record()
    torch.cuda.synchronize()
    run_time_96 = starter.elapsed_time(ender)

    close_flag = torch.allclose(x_torch, attn_proj_out_torch_2, rtol=1e-2, atol=1e-2)
    print("close check:", close_flag)

    print("first 10 elements of attn_proj_out_torch:")
    print(attn_proj_out_torch[0][:10])
    print("first 10 elements of x_torch:")
    print(x_torch[0][:10])

    print("x_torch is attn_proj_out_torch?", x_torch is attn_proj_out_torch)
    print("first 10 elements of attn_proj_out_torch_2:")
    print(attn_proj_out_torch_2[0][:10])

    # print("all elements of x:")
    # print(x_torch)



    if world_size > 1:
        dist.destroy_process_group()
