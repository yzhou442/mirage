import argparse
import threading
import queue
from typing import Callable, Optional, Tuple, Dict, Any
from transformers import AutoTokenizer

import torch
import logging
import os
import time

from mirage.mpk.mpk import MPK, MPKMetadata, MirageModelConfig


class StageWorker(threading.Thread):
    def __init__(
        self,
        *,
        name: str,
        thread_id: int,
        # build config
        model: str,
        max_seq_length: int,
        max_num_batched_tokens: int,
        max_num_batched_requests: int,
        page_size: int,
        max_num_pages: int,
        profiling: bool,
        trace_name: Optional[str],
        output_dir: Optional[str],
        use_cutlass_kernel: bool,
        max_sm_num: int,
        num_workers: int,
        num_schedulers: int,
        # logging config
        log_dir: str,
        log_level: str,
        cpu_stream_buffer: torch.Tensor = None,
        in_queue: "queue.Queue[Optional[Tuple[int, str]]]",
        out_queue: "queue.Queue[Optional[Tuple[int, str]]]",
        stream_queue: "queue.Queue[Optional[Tuple[int, str]]]" = None,
        transform: Optional[Callable[[str], str]] = None,
    ) -> None:
        super().__init__(name=name)
        # Config
        self.model_name = model
        self.thread_id = thread_id
        self.max_seq_length = max_seq_length
        self.max_num_batched_tokens = max_num_batched_tokens
        self.max_num_batched_requests = max_num_batched_requests
        self.page_size = page_size
        self.max_num_pages = max_num_pages
        self.profiling = profiling
        self.trace_name = trace_name
        self.output_dir = output_dir
        self.use_cutlass_kernel = use_cutlass_kernel
        self.max_sm_num = max_sm_num

        # Queues and transform
        self.in_queue = in_queue
        self.out_queue = out_queue
        self.stream_queue = stream_queue
        self.transform = transform
        self.cpu_stream_buffer = cpu_stream_buffer

        # CUDA stream
        self.stream = torch.cuda.Stream()

        # Logging setup
        os.makedirs(log_dir, exist_ok=True)
        self.logger = logging.getLogger(f"worker.{name}")
        # avoid duplicate handlers if instantiated once
        if not self.logger.handlers:
            file_path = os.path.join(log_dir, f"{name}.log")
            fh = logging.FileHandler(file_path, encoding="utf-8")
            fmt = logging.Formatter("%(asctime)s %(name)s [%(levelname)s] %(message)s")
            fh.setFormatter(fmt)
            self.logger.addHandler(fh)
            level_val = getattr(logging, (log_level or "INFO").upper(), logging.INFO)
            self.logger.setLevel(level_val)
            self.logger.propagate = False

        # Placeholders for MPK and meta tensors
        self.mpk: Optional[MPK] = None
        self.step: Optional[torch.Tensor] = None
        self.tokens: Optional[torch.Tensor] = None

        # Keep references to all meta tensors for lifecycle management
        self._prompt_lengths: Optional[torch.Tensor] = None
        self._input_tokens: Optional[torch.Tensor] = None
        self._output_tokens: Optional[torch.Tensor] = None
        self._num_new_tokens: Optional[torch.Tensor] = None
        self._qo_indptr_buffer: Optional[torch.Tensor] = None
        self._paged_kv_indptr_buffer: Optional[torch.Tensor] = None
        self._paged_kv_indices_buffer: Optional[torch.Tensor] = None
        self._paged_kv_last_page_len_buffer: Optional[torch.Tensor] = None
        self._profiler_tensor: Optional[torch.Tensor] = None
        self.num_workers = num_workers
        self.num_schedulers = num_schedulers
        self.start_time = None
        

        # Build immediately
        self._build_mpk()

    def _build_mpk(self) -> None:
        total_num_requests = 1

        self.tokens = torch.full(
            (total_num_requests, self.max_seq_length), 0, dtype=torch.long, device="cuda"
        )
        self._prompt_lengths = torch.full(
            (total_num_requests,), 0, dtype=torch.int, device="cuda"
        )
        self._input_tokens = torch.full(
            (self.max_num_batched_tokens, 1), 0, dtype=torch.long, device="cuda"
        )
        self._output_tokens = torch.full(
            (self.max_num_batched_tokens, 1), 0, dtype=torch.long, device="cuda"
        )
        self.step = torch.full((total_num_requests,), 0, dtype=torch.int32, device="cuda")
        self._num_new_tokens = torch.full((total_num_requests,), 1, dtype=torch.int32, device="cuda")

        self._qo_indptr_buffer = torch.zeros(
            self.max_num_batched_requests + 1, dtype=torch.int32, device="cuda"
        )
        self._paged_kv_indptr_buffer = torch.zeros(
            self.max_num_batched_requests + 1, dtype=torch.int32, device="cuda"
        )
        self._paged_kv_indices_buffer = torch.zeros(
            self.max_num_pages, dtype=torch.int32, device="cuda"
        )
        self._paged_kv_last_page_len_buffer = torch.zeros(
            self.max_num_batched_requests, dtype=torch.int32, device="cuda"
        )

        self._profiler_tensor = (
            torch.zeros(3000 * 128, dtype=torch.uint64, device="cuda").contiguous()
            if self.profiling
            else None
        )

        mirage_model_config = MirageModelConfig(with_lm_head=True)

        mpk_metadata = MPKMetadata(
            mode="offline",
            thread_id=self.thread_id,
            total_num_requests=total_num_requests,
            num_remote_schedulers=0,
            max_seq_length=self.max_seq_length,
            max_num_batched_requests=self.max_num_batched_requests,
            max_num_batched_tokens=self.max_num_batched_tokens,
            max_num_pages=self.max_num_pages,
            page_size=self.page_size,
            weight_from_model=True,
            model_name=self.model_name,
            step=self.step,
            tokens=self.tokens,
            input_tokens=self._input_tokens,
            output_tokens=self._output_tokens,
            num_new_tokens=self._num_new_tokens,
            prompt_lengths=self._prompt_lengths,
            qo_indptr_buffer=self._qo_indptr_buffer,
            paged_kv_indptr_buffer=self._paged_kv_indptr_buffer,
            paged_kv_indices_buffer=self._paged_kv_indices_buffer,
            paged_kv_last_page_len_buffer=self._paged_kv_last_page_len_buffer,
            model_config=mirage_model_config,
            profiling=self.profiling,
            profiler_tensor=self._profiler_tensor,
            cpu_stream_buffer=self.cpu_stream_buffer,
            trace_name=self.trace_name,
            spec_decode=None,
            spec_decode_config=None,
            use_cutlass_kernel=self.use_cutlass_kernel,
            max_sm_num=self.max_sm_num,
            num_workers=self.num_workers,
            num_schedulers=self.num_schedulers,
        )
        self.logger.info(f"max_sm_num: {self.max_sm_num}")

        self.mpk = MPK(mpk_metadata)
        self.mpk.build()
        self.mpk.compile(output_dir=self.output_dir)
        
    def timing_from_start(self) -> float:
        if self.start_time is None:
            self.start_time = time.time()
            return 0.0
        return time.time() - self.start_time

    def run(self) -> None:
        self.logger.info(f"Worker [{self.name}] started")
        while True:
            time.sleep(0.001)  # Prevent busy waiting
            item = self.in_queue.get()
            if item is None:
                # Propagate termination downstream and exit.
                self.logger.info("Shutdown signal received; forwarding and exiting")
                self.out_queue.put(None)
                if self.thread_id == 1:
                    self.stream_queue.put(None)
                self.in_queue.task_done()
                break

            req_id, text = item
            # self.logger.info("recv req_id=%d input=%s", req_id, text)
            # prompt = self.transform(text) if self.transform else text
            if self.thread_id == 0:
                prompt = """
                Please write a detailed explanation and implementation of the GeMM (General Matrix Multiplication) algorithm in CUDA.
                """
            else:
                prompt = """
                Can you write me a detailed GeMM kernel in CUDA and provide step by step explanation of the code?
                """

            # print(f"Agent{self.thread_id} Dealing with req_id={req_id} input={text}")
            self.logger.info(f"Agent{self.thread_id} Dealing with req_id={req_id} input")
            # with torch.cuda.stream(self.stream):
            self.logger.info(f"Agent{self.thread_id} Clearing buffers")
            self.mpk.clear_buffers()
            # if self.thread_id == 1:
            #     self.stream_queue.put((req_id, ""))  # signal stream worker to start    
            self.logger.info(f"Agent{self.thread_id} Loading new request")
            self.mpk.load_new_request(prompt)
            self.logger.info(f"Agent{self.thread_id} Initializing request function")
            self.mpk.init_request_func()
            self.logger.info(f"Agent{self.thread_id} Running MPK")
            if req_id == 0 and self.thread_id == 0:
                input("Press Enter to start MPK...")
            # start_time = time.time()
            self.mpk(logger=self.logger)
            # time_elapsed = time.time() - start_time
            self.logger.info(f"Agent{self.thread_id} MPK finished")

            # Ensure this request finished on this stage's stream
            # self.stream.synchronize()
            # self.logger.info(f"Agent{self.thread_id} Stream synchronized")

            # # Decode the single-request result
            # try:
            #     cur_step = int(self.step[0].item())
            #     generated_ids = self.tokens[0, : cur_step + 1]
            #     output = self.mpk.decode(generated_ids)
            # except Exception as e:
            #     self.logger.error(f"Error decoding output: {e}, cur_step: {cur_step}, generated_ids: {generated_ids}")
            #     output = ""
            
            # try:
            #     output = output.split("</think>")[1].strip()
            # except IndexError:
            #     # warn
            #     self.logger.warning("No </think> found in output")
            #     output = output

            # self.logger.info(f"send req_id={req_id} time={self.timing_from_start()} ms, output:{output}")
            # print(self.step, self._prompt_lengths, time_elapsed)
            self.out_queue.put((req_id, ""))
            self.in_queue.task_done()
        self.logger.info("worker stopped")


def stream_output(cpu_stream_buffer: torch.Tensor, tokenizer, in_queue: queue.Queue) -> None:
    request_idx = 0
    while True:
        # item = in_queue.get()
        # if item is None:
        #     print(f"Streaming worker received shutdown signal; exiting")
        #     in_queue.task_done()
        #     break
        # request_idx, _ = item
        print(f"Streaming output tokens for request {request_idx}:")
        for token_idx in range(cpu_stream_buffer.size(0)-55):
            token = cpu_stream_buffer[token_idx].item()
            count_iter = 0
            while token == -1:
                time.sleep(0.01)
                token = cpu_stream_buffer[token_idx].item()
                # count_iter += 1
                # if count_iter > 500:  # timeout after 5 seconds
                #     print("\nStreaming timeout.")
                #     return
            if token == tokenizer.eos_token_id:
                break
            print(tokenizer.decode([token]), end="", flush=True)
        print(f"\nEnd of streamed output for request {request_idx}.")
        break


def main() -> None:
    parser = argparse.ArgumentParser(description="Three-stage MPK agent pipeline demo")
    # parser.add_argument("--model", type=str, default="Qwen/Qwen3-1.7B", help="HF model id")
    parser.add_argument("--max-seq-length", type=int, default=4096)
    parser.add_argument("--max-num-batched-tokens", type=int, default=8)
    parser.add_argument("--max-num-batched-requests", type=int, default=1)
    parser.add_argument("--page-size", type=int, default=4096)
    parser.add_argument("--max-num-pages", type=int, default=16)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--trace-name", type=str, default="")
    parser.add_argument("--profiling", action="store_true")
    parser.add_argument(
        "--no-use-cutlass-kernel",
        action="store_false",
        dest="use_cutlass_kernel",
        default=True,
        help="Disable cutlass kernel variant",
    )
    parser.add_argument("--num-requests", type=int, default=5)
    parser.add_argument("--log-dir", type=str, default="./logs", help="Directory to store per-worker logs")
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level for worker files",
    )
    parser.add_argument(
        "--prompt-prefix",
        type=str,
        default="Give me a short introduction to large language model.",
        help="Base text of stage-1 prompts",
    )
    args = parser.parse_args() 

    # Stage transforms
    def tfm1(text: str) -> str:
        return f""

    def tfm2(text: str) -> str:
        return f""

    def tfm3(text: str) -> str:
        return f""
    
    tfs = [tfm1, tfm2, tfm3]

    # max_sm_num = 36
    max_sm_num = 40
    
    models = ["Qwen/Qwen3-1.7B", "Qwen/Qwen3-8B"]
    num_workers = [16, 64]
    num_schedulers = [8, 8]
    queues = [queue.Queue(maxsize=8) for _ in range(len(models) + 1)]
    stream_queues = [queue.Queue(maxsize=8) for _ in range(len(models))]
    kwargs_list = []
    workers = []
    stream_workers = []
    cpu_stream_buffer_list = [torch.full((args.max_seq_length, ), -1, dtype=torch.long, pin_memory=True, device="cpu") for _ in range(len(models))]
    # Workers (each owns its MPK and tensors)
    tokenizer = AutoTokenizer.from_pretrained(models[0])
    for i in range(len(models)):
        kwargs = dict(
            thread_id=i,
            model=models[i],
            max_seq_length=args.max_seq_length,
            max_num_batched_tokens=args.max_num_batched_tokens,
            max_num_batched_requests=args.max_num_batched_requests,
            page_size=args.page_size,
            max_num_pages=args.max_num_pages,
            profiling=args.profiling,
            trace_name=args.trace_name,
            output_dir=args.output_dir,
            use_cutlass_kernel=args.use_cutlass_kernel,
            max_sm_num=max_sm_num,
            num_workers=num_workers[i],
            num_schedulers=num_schedulers[i],
            log_dir=args.log_dir,
            log_level=args.log_level,
            cpu_stream_buffer=cpu_stream_buffer_list[i],
        )
        # kwargs_list.append(kwargs)
        w = StageWorker(name=f"Stage-{i+1}", in_queue=queues[i], out_queue=queues[i+1], stream_queue=stream_queues[i], transform=tfs[i], **kwargs)
        workers.append(w)
        if i == 1:
            stream_w = threading.Thread(target=stream_output, args=(cpu_stream_buffer_list[i], tokenizer, stream_queues[i]), name=f"Stream-Stage-{i+1}")
            stream_workers.append(stream_w)
    # w1 = StageWorker(name="Stage-1", in_queue=q_in, out_queue=q_12, transform=tfm1, **common_kwargs)
    # w2 = StageWorker(name="Stage-2", in_queue=q_12, out_queue=q_23, transform=tfm2, **common_kwargs)
    # w3 = StageWorker(name="Stage-3", in_queue=q_23, out_queue=q_out, transform=tfm3, **common_kwargs)

    for w in workers:
        w.start()
    for sw in stream_workers:
        sw.start()
        
    q_in = queues[0]
    q_out = queues[-1]

    # Enqueue requests to stage-1
    for i in range(args.num_requests):
        base_prompt = f"{args.prompt_prefix} [请求ID={i}]"
        q_in.put((i, base_prompt))

    # Close input and propagate sentinels through pipeline
    q_in.put(None)

    # Collect results
    results: Dict[int, str] = {}
    finished = 0
    while True:
        item = q_out.get()
        if item is None:
            finished += 1
            q_out.task_done()
            break
        req_id, text = item
        results[req_id] = text
        q_out.task_done()

    # Drain and join
    for w in workers:
        w.join()
    for sw in stream_workers:
        sw.join()
    # # Print ordered outputs
    # for i in range(args.num_requests):
    #     out_text = results.get(i, "<missing>")
    #     print(f"===== Request {i} =====")
    #     print(out_text)


if __name__ == "__main__":
    main()

 