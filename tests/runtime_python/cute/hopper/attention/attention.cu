#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include "cute/tensor.hpp"
#include "cutlass/pipeline/pipeline.hpp"
#include "prefill_sm90.cuh"
#include "epilogue.cuh"
#include "kernel_traits.cuh"
#include "mainloop.cuh"
#include "mainloop_mma.cuh"
#include "tile_scheduler.cuh"
#include "utils.cuh"
#include "../../../../../include/mirage/persistent_kernel/tasks/hopper/tma_3d.cuh"
#include "../../../../../include/mirage/persistent_kernel/tasks/hopper/tma_2d.cuh"

using namespace cute;
using namespace cutlass;
using bfloat16 = cutlass::bfloat16_t;
using kernel::MaskMode;
using kernel::AttentionKernelTraits;
using kernel::PrefillWithKVCacheKernel;
using kernel::BatchPrefillTileScheduler;
using kernel::BatchPrefillPersistentTileScheduler;
using kernel::CollectiveEpilogue;
using kernel::CollectiveMainloop;
using kernel::get_gmem_layout;
using kernel::get_lse_gmem_layout;

// Define missing macros
#ifndef FLASHINFER_CUDA_CALL
#define FLASHINFER_CUDA_CALL(func) \
  { \
    cudaError_t status = (func); \
    if (status != cudaSuccess) { \
      printf("CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(status)); \
    } \
  }
#endif

// Multitoken Paged Attention cute implementation
template <typename T,
          int NUM_QO_HEADS,
          int NUM_KV_HEADS,
          int KV_CACHE_STRIDE,
          int QKV_STRIDE,
          int O_STRIDE,
          int HEAD_DIM,
          int MAX_SEQ_LEN,
          int PAGE_SIZE,
          typename TMA_Q,
          typename TMA_KV,
          typename TMA_PAGED_KV,
          typename TMA_OUTPUT,
          int MAX_TOKENS = 8>
__device__ __forceinline__ void multitoken_paged_attention_hopper_cute_impl(
    const TMA_Q &tma_q,
    const TMA_KV &tma_k,
    const TMA_KV &tma_v,
    const TMA_PAGED_KV &tma_paged_k_cache,
    const TMA_PAGED_KV &tma_paged_v_cache,
    const TMA_OUTPUT &tma_output,
    void *paged_k_cache_ptr,
    void *paged_v_cache_ptr,
    int const *qo_indptr_buffer_ptr,
    int const *paged_kv_indptr_buffer_ptr,
    int const *paged_kv_indices_buffer_ptr,
    int const *paged_kv_last_page_len_buffer_ptr,
    int request_id,
    bool qk_norm,
    bool rope,
    void const *q_norm_weight_ptr,
    void const *k_norm_weight_ptr,
    void const *cos_ptr,
    void const *sin_ptr,
    float q_eps,
    float k_eps,
    void *output_ptr,
    void *qkv_ptr) {
  // TODO: Implement the actual attention computation here
  // This is a placeholder implementation
}

// Multitoken Paged Attention cute
template <typename T,
          int NUM_QO_HEADS,
          int NUM_KV_HEADS,
          int KV_CACHE_STRIDE,
          int QKV_STRIDE,
          int O_STRIDE,
          int HEAD_DIM,
          int MAX_SEQ_LEN,
          int PAGE_SIZE,
          typename TMA_Q,
          typename TMA_KV,
          typename TMA_PAGED_KV,
          typename TMA_OUTPUT,
          int MAX_TOKENS = 8>
__global__ void multitoken_paged_attention_wrapper_hopper(
    const __grid_constant__ TMA_Q tma_q,
    const __grid_constant__ TMA_KV tma_k,
    const __grid_constant__ TMA_KV tma_v,
    const __grid_constant__ TMA_PAGED_KV tma_paged_k_cache,
    const __grid_constant__ TMA_PAGED_KV tma_paged_v_cache,
    const __grid_constant__ TMA_OUTPUT tma_output,
    void *paged_k_cache_ptr,
    void *paged_v_cache_ptr,
    int const *qo_indptr_buffer_ptr,
    int const *paged_kv_indptr_buffer_ptr,
    int const *paged_kv_indices_buffer_ptr,
    int const *paged_kv_last_page_len_buffer_ptr,
    int request_id,
    bool qk_norm,
    bool rope,
    void const *q_norm_weight_ptr,
    void const *k_norm_weight_ptr,
    void const *cos_ptr,
    void const *sin_ptr,
    float q_eps,
    float k_eps,
    void *output_ptr,
    void *qkv_ptr) {

  multitoken_paged_attention_hopper_cute_impl<T,
                                         NUM_QO_HEADS,
                                         NUM_KV_HEADS,
                                         KV_CACHE_STRIDE,
                                         QKV_STRIDE,
                                         O_STRIDE,
                                         HEAD_DIM,
                                         MAX_SEQ_LEN,
                                         PAGE_SIZE,
                                         TMA_Q,
                                         TMA_KV,
                                         TMA_PAGED_KV,
                                         TMA_OUTPUT,
                                         MAX_TOKENS>(
      tma_q,
      tma_k,
      tma_v,
      tma_paged_k_cache,
      tma_paged_v_cache,
      tma_output,
      paged_k_cache_ptr,
      paged_v_cache_ptr,
      qo_indptr_buffer_ptr,
      paged_kv_indptr_buffer_ptr,
      paged_kv_indices_buffer_ptr,
      paged_kv_last_page_len_buffer_ptr,
      request_id,
      qk_norm,
      rope,
      q_norm_weight_ptr,
      k_norm_weight_ptr,
      cos_ptr,
      sin_ptr,
      q_eps,
      k_eps,
      output_ptr,
      qkv_ptr);
}

template <typename T,
          int NUM_QO_HEADS,
          int NUM_KV_HEADS,
          int KV_CACHE_STRIDE,
          int QKV_STRIDE,
          int O_STRIDE,
          int HEAD_DIM,
          int MAX_SEQ_LEN,
          int PAGE_SIZE,
          int MAX_TOKENS = 8>
void launch_multitoken_paged_attention_hopper(
    void *qkv_ptr,
    void *paged_k_cache_ptr,
    void *paged_v_cache_ptr,
    void *output_ptr,
    int const *qo_indptr_buffer_ptr,
    int const *paged_kv_indptr_buffer_ptr,
    int const *paged_kv_indices_buffer_ptr,
    int const *paged_kv_last_page_len_buffer_ptr,
    int request_id,
    bool qk_norm,
    bool rope,
    void const *q_norm_weight_ptr,
    void const *k_norm_weight_ptr,
    void const *cos_ptr,
    void const *sin_ptr,
    float q_eps,
    float k_eps) {
  dim3 grid_dim(1, 1, 1);
  dim3 block_dim(256, 1, 1);
  size_t smem_size = 224 * 1024;

  constexpr int B = 3;
  constexpr int M = 3;
  constexpr int S = 3;
  constexpr int TMA_CP_SIZE = 64;
  constexpr int KV_TILE_SIZE = 64;
  constexpr int prompt_len = 8;
  constexpr int num_tokens = 8;

  constexpr int NUM_PAGES = 100;
  constexpr int TAIL_PAGE_SIZE = prompt_len % PAGE_SIZE;

  using TMA_Q =
      kernel::tma::tma_3d<bfloat16,
                          B,
                          M,
                          S,
                          num_tokens,
                          (NUM_QO_HEADS + 2 * NUM_KV_HEADS),
                          HEAD_DIM,
                          num_tokens,
                          NUM_QO_HEADS,
                          TMA_CP_SIZE,
                          (NUM_QO_HEADS + 2 * NUM_KV_HEADS) * HEAD_DIM,
                          HEAD_DIM,
                          1,
                          1,
                          (HEAD_DIM + TMA_CP_SIZE - 1) / TMA_CP_SIZE,
                          num_tokens * NUM_QO_HEADS * TMA_CP_SIZE,
                          true>;

  using TMA_KV =
      kernel::tma::tma_3d<bfloat16,
                          B,
                          M,
                          S,
                          num_tokens,
                          (NUM_QO_HEADS + 2 * NUM_KV_HEADS),
                          HEAD_DIM,
                          num_tokens,
                          NUM_KV_HEADS,
                          TMA_CP_SIZE,
                          (NUM_QO_HEADS + 2 * NUM_KV_HEADS) * HEAD_DIM,
                          HEAD_DIM,
                          1,
                          1,
                          (HEAD_DIM + TMA_CP_SIZE - 1) / TMA_CP_SIZE,
                          num_tokens * NUM_KV_HEADS * TMA_CP_SIZE, // skip number of rows between current 64 cols and next 64 cols
                          true>;

  using TMA_PAGED_KV_CACHE =
      kernel::tma::tma_3d<bfloat16,
                          B,
                          M,
                          S,
                          NUM_PAGES,
                          PAGE_SIZE,
                          HEAD_DIM,
                          1,
                          KV_TILE_SIZE,
                          TMA_CP_SIZE,
                          PAGE_SIZE * HEAD_DIM,
                          HEAD_DIM,
                          1,
                          1,
                          (HEAD_DIM + TMA_CP_SIZE - 1) / TMA_CP_SIZE,
                          KV_TILE_SIZE * TMA_CP_SIZE,
                          true>;

  using TMA_OUTPUT =
      kernel::tma::tma_2d<bfloat16,
                          3,
                          3,
                          3,
                          num_tokens * NUM_QO_HEADS,
                          HEAD_DIM,
                          num_tokens * NUM_QO_HEADS,
                          TMA_CP_SIZE,
                          HEAD_DIM,
                          1,
                          1,
                          (HEAD_DIM + TMA_CP_SIZE - 1) / TMA_CP_SIZE,
                          num_tokens * NUM_QO_HEADS * TMA_CP_SIZE,
                          true>;

  // bfloat16 *__restrict__ qkv_ptr_bf16 = static_cast<bfloat16 *>(qkv_ptr);

  TMA_Q tma_q(qkv_ptr);
  TMA_KV tma_k(qkv_ptr);
  TMA_KV tma_v(qkv_ptr);
  TMA_PAGED_KV_CACHE tma_paged_k_cache(paged_k_cache_ptr);
  TMA_PAGED_KV_CACHE tma_paged_v_cache(paged_v_cache_ptr);
  TMA_OUTPUT tma_output(output_ptr);

  cudaFuncSetAttribute(
      multitoken_paged_attention_wrapper_hopper<T,
                                                NUM_QO_HEADS,
                                                NUM_KV_HEADS,
                                                KV_CACHE_STRIDE,
                                                QKV_STRIDE,
                                                O_STRIDE,
                                                HEAD_DIM,
                                                MAX_SEQ_LEN,
                                                PAGE_SIZE,
                                                TMA_Q,
                                                TMA_KV,
                                                TMA_PAGED_KV_CACHE,
                                                TMA_OUTPUT,
                                                num_tokens>,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      smem_size);

#ifndef MIRAGE_PROFILE_HOPPER
  multitoken_paged_attention_wrapper_hopper<T,
                                            NUM_QO_HEADS,
                                            NUM_KV_HEADS,
                                            KV_CACHE_STRIDE,
                                            QKV_STRIDE,
                                            O_STRIDE,
                                            HEAD_DIM,
                                            MAX_SEQ_LEN,
                                            PAGE_SIZE,
                                            TMA_Q,
                                            TMA_KV,
                                            TMA_PAGED_KV_CACHE,
                                            TMA_OUTPUT,
                                            num_tokens>
      <<<grid_dim, block_dim, smem_size>>>(tma_q,
                                           tma_k,
                                           tma_v,
                                           tma_paged_k_cache,
                                           tma_paged_v_cache,
                                           tma_output,
                                           paged_k_cache_ptr,
                                           paged_v_cache_ptr,
                                           qo_indptr_buffer_ptr,
                                           paged_kv_indptr_buffer_ptr,
                                           paged_kv_indices_buffer_ptr,
                                           paged_kv_last_page_len_buffer_ptr,
                                           request_id,
                                           qk_norm,
                                           rope,
                                           q_norm_weight_ptr,
                                           k_norm_weight_ptr,
                                           cos_ptr,
                                           sin_ptr,
                                           q_eps,
                                           k_eps,
                                           output_ptr,
                                           qkv_ptr);
#else

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  constexpr int WARMUP_RUNS = 16;
  constexpr int BENCHMARK_RUNS = 1000;

  printf("=== Multitoken Paged Attention Kernel Performance Profiling ===\n");

  for (int i = 0; i < WARMUP_RUNS; i++) {
    multitoken_paged_attention_wrapper_hopper<T,
                                              NUM_QO_HEADS,
                                              NUM_KV_HEADS,
                                              KV_CACHE_STRIDE,
                                              QKV_STRIDE,
                                              O_STRIDE,
                                              HEAD_DIM,
                                              MAX_SEQ_LEN,
                                              PAGE_SIZE,
                                              TMA_Q,
                                              TMA_KV,
                                              TMA_PAGED_KV_CACHE,
                                              TMA_OUTPUT,
                                              num_tokens>
        <<<grid_dim, block_dim, smem_size>>>(tma_q,
                                             tma_k,
                                             tma_v,
                                             tma_paged_k_cache,
                                             tma_paged_v_cache,
                                             tma_output,
                                             paged_k_cache_ptr,
                                             paged_v_cache_ptr,
                                             qo_indptr_buffer_ptr,
                                             paged_kv_indptr_buffer_ptr,
                                             paged_kv_indices_buffer_ptr,
                                             paged_kv_last_page_len_buffer_ptr,
                                             request_id,
                                             qk_norm,
                                             rope,
                                             q_norm_weight_ptr,
                                             k_norm_weight_ptr,
                                             cos_ptr,
                                             sin_ptr,
                                             q_eps,
                                             k_eps,
                                             output_ptr,
                                             qkv_ptr);
  }
  cudaDeviceSynchronize();

  printf("Running %d benchmark iterations...\n", BENCHMARK_RUNS);

  float *iteration_times = new float[BENCHMARK_RUNS];
  float total_time_ms = 0.0f;

  for (int i = 0; i < BENCHMARK_RUNS; i++) {
    cudaEventRecord(start);
    multitoken_paged_attention_wrapper_hopper<T,
                                              NUM_QO_HEADS,
                                              NUM_KV_HEADS,
                                              KV_CACHE_STRIDE,
                                              QKV_STRIDE,
                                              O_STRIDE,
                                              HEAD_DIM,
                                              MAX_SEQ_LEN,
                                              PAGE_SIZE,
                                              TMA_Q,
                                              TMA_KV,
                                              TMA_PAGED_KV_CACHE,
                                              TMA_OUTPUT,
                                              num_tokens>
        <<<grid_dim, block_dim, smem_size>>>(tma_q,
                                             tma_k,
                                             tma_v,
                                             tma_paged_k_cache,
                                             tma_paged_v_cache,
                                             tma_output,
                                             paged_k_cache_ptr,
                                             paged_v_cache_ptr,
                                             qo_indptr_buffer_ptr,
                                             paged_kv_indptr_buffer_ptr,
                                             paged_kv_indices_buffer_ptr,
                                             paged_kv_last_page_len_buffer_ptr,
                                             request_id,
                                             qk_norm,
                                             rope,
                                             q_norm_weight_ptr,
                                             k_norm_weight_ptr,
                                             cos_ptr,
                                             sin_ptr,
                                             q_eps,
                                             k_eps,
                                             output_ptr,
                                             qkv_ptr);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float iteration_time_ms;
    cudaEventElapsedTime(&iteration_time_ms, start, stop);

    iteration_times[i] = iteration_time_ms;
    total_time_ms += iteration_time_ms;
  }

  float avg_time_ms = total_time_ms / BENCHMARK_RUNS;

  printf("\n=== Multitoken Paged Attention Performance Results ===\n");
  printf("Configuration:\n");
  printf("  NUM_QO_HEADS=%d, NUM_KV_HEADS=%d, HEAD_DIM=%d\n",
         NUM_QO_HEADS,
         NUM_KV_HEADS,
         HEAD_DIM);
  printf("  MAX_SEQ_LEN=%d, PAGE_SIZE=%d, MAX_TOKENS=%d\n",
         MAX_SEQ_LEN,
         PAGE_SIZE,
         MAX_TOKENS);
  printf("  Average: %.3f ms\n", avg_time_ms);

  printf("===============================\n");

  delete[] iteration_times;
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
#endif
}


template <typename KernelTraits,
          bool LEFT_SLIDING_WINDOW,
          bool CAUSAL,
          bool SAME_SCHEDULE_FOR_ALL_HEADS,
          typename Params,
          bool MULTIITEMSCORING = false>
cudaError_t
    BatchPrefillWithPagedKVCacheKernelTraitsDispatched(Params &params,
                                                       cudaStream_t stream) {
  using DTypeQ = typename KernelTraits::DTypeQ;
  using DTypeKV = typename KernelTraits::DTypeKV;
  using DTypeO = typename KernelTraits::DTypeO;
  using IdType = typename KernelTraits::IdType;

  using CollectiveMainloop =
      kernel::CollectiveMainloop<typename Params::AdditionalParams,
                               KernelTraits,
                               CAUSAL>;
  using CollectiveEpilogue = kernel::CollectiveEpilogue<KernelTraits>;
  using Scheduler =
      std::conditional_t<SAME_SCHEDULE_FOR_ALL_HEADS,
                         BatchPrefillTileScheduler<IdType>,
                         BatchPrefillPersistentTileScheduler<IdType>>;

  typename CollectiveMainloop::Params mainloop_params =
      CollectiveMainloop::to_underlying_arguments(
          {params.q_ptr,
           get_gmem_layout(params.nnz_qo,
                           params.num_qo_heads,
                           KernelTraits::HEAD_DIM_QK,
                           params.q_stride_n,
                           params.q_stride_h), // layout_Q
           params.k_ptr,
           // NOTE(Zihao): nnz was useless here, we can just pass 0
           get_gmem_layout(/*nnz=*/0,
                           params.num_kv_heads,
                           KernelTraits::HEAD_DIM_QK,
                           params.k_stride_n,
                           params.k_stride_h), // layout_K
           params.v_ptr,
           get_gmem_layout(/*nnz=*/0,
                           params.num_kv_heads,
                           KernelTraits::HEAD_DIM_VO,
                           params.v_stride_n,
                           params.v_stride_h), // layout_V
           params.window_left,
           params.additional_params});
  typename CollectiveEpilogue::Params epilogue_params =
      CollectiveEpilogue::to_underlying_arguments({
          params.o_ptr,
          get_gmem_layout(params.nnz_qo,
                          params.num_qo_heads,
                          KernelTraits::HEAD_DIM_VO,
                          params.o_stride_n,
                          params.o_stride_h), // layout_O
          params.lse_ptr,
          get_lse_gmem_layout(params.nnz_qo, params.num_qo_heads), // layout_LSE
      });

  typename Scheduler::Arguments scheduler_args = {
      params.work_indptr,
      params.head_indices,
      params.qo_tile_indices,
      params.qo_indptr,
      params.kv_indptr,
      params.qo_lens,
      params.kv_lens,
      params.batch_indices,
      cutlass::FastDivmod(params.num_qo_heads / params.num_kv_heads),
      params.num_qo_heads};
  typename Scheduler::Params scheduler_params =
      Scheduler::to_underlying_arguments(scheduler_args);

  // Get the ptr to kernel function.
  auto kernel = (void *)PrefillWithKVCacheKernel<CollectiveMainloop,
                                                 CollectiveEpilogue,
                                                 KernelTraits,
                                                 LEFT_SLIDING_WINDOW,
                                                 CAUSAL,
                                                 Scheduler,
                                                 MULTIITEMSCORING>;
  int smem_size = sizeof(typename KernelTraits::SharedStorage);
  FLASHINFER_CUDA_CALL(cudaFuncSetAttribute(
      kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

  int device;
  cudaGetDevice(&device);
  int multiprocessor_count;
  FLASHINFER_CUDA_CALL(cudaDeviceGetAttribute(
      &multiprocessor_count, cudaDevAttrMultiProcessorCount, device));
  dim3 grid_dims =
      Scheduler::get_grid_dim(scheduler_args, multiprocessor_count);
  static constexpr int ctaSize = KernelTraits::NUM_WARPS * 32;
  dim3 block_dims(ctaSize);
  void *args[] = {&mainloop_params, &epilogue_params, &scheduler_params};
  FLASHINFER_CUDA_CALL(
      cudaLaunchKernel(kernel, grid_dims, block_dims, args, smem_size, stream));

  return cudaSuccess;
}

template <uint32_t HEAD_DIM_QK,
          uint32_t HEAD_DIM_VO,
          MaskMode MASK_MODE,
          bool LEFT_SLIDING_WINDOW,
          bool SAME_SCHEDULE_FOR_ALL_HEADS,
          typename AttentionVariant,
          typename Params>
cudaError_t BatchPrefillWithPagedKVCacheDispatched(Params &params,
                                                   bool enable_pdl,
                                                   cudaStream_t stream) {
  static_assert(HEAD_DIM_VO == 64 || HEAD_DIM_VO == 128 || HEAD_DIM_VO == 256);
  if (MASK_MODE == MaskMode::kCustom) {
    return cudaErrorNotSupported; // Not supported yet.
  }
  constexpr bool CAUSAL = MASK_MODE == MaskMode::kCausal;
  constexpr bool MULTIITEMSCORING = MASK_MODE == MaskMode::kMultiItemScoring;
  if constexpr (HEAD_DIM_QK == HEAD_DIM_VO) {
    if constexpr (HEAD_DIM_VO == 64) {
      // NOTE(Zihao): CTA_KV not tuned for HEAD_DIM == 64, need to optimize
      // later
      BatchPrefillWithPagedKVCacheKernelTraitsDispatched<
          AttentionKernelTraits</*USE_TMA_LOAD_KV=*/false,
                                HEAD_DIM_QK,
                                HEAD_DIM_VO,
                                /*CTA_Q_=*/192,
                                /*CTA_KV_=*/96,
                                /*NUM_STAGES_=*/2,
                                typename Params::DTypeQ,
                                typename Params::DTypeKV,
                                typename Params::DTypeO,
                                typename Params::IdType,
                                AttentionVariant>,
          LEFT_SLIDING_WINDOW,
          CAUSAL,
          SAME_SCHEDULE_FOR_ALL_HEADS,
          Params,
          MULTIITEMSCORING>(params, stream);
    } else if constexpr (HEAD_DIM_VO == 128) {
      BatchPrefillWithPagedKVCacheKernelTraitsDispatched<
          AttentionKernelTraits</*USE_TMA_LOAD_KV=*/false,
                                HEAD_DIM_QK,
                                HEAD_DIM_VO,
                                /*CTA_Q_=*/128,
                                /*CTA_KV_=*/96,
                                /*NUM_STAGES_=*/2,
                                typename Params::DTypeQ,
                                typename Params::DTypeKV,
                                typename Params::DTypeO,
                                typename Params::IdType,
                                AttentionVariant>,
          LEFT_SLIDING_WINDOW,
          CAUSAL,
          SAME_SCHEDULE_FOR_ALL_HEADS,
          Params,
          MULTIITEMSCORING>(params, stream);
    } else {
      // HEAD_DIM == 256;
      // NOTE(Zihao): CTA_KV not tuned for HEAD_DIM == 256, need to optimize
      // later
      BatchPrefillWithPagedKVCacheKernelTraitsDispatched<
          AttentionKernelTraits</*USE_TMA_LOAD_KV=*/false,
                                HEAD_DIM_QK,
                                HEAD_DIM_VO,
                                /*CTA_Q_=*/128,
                                /*CTA_KV_=*/32,
                                /*NUM_STAGES_=*/2,
                                typename Params::DTypeQ,
                                typename Params::DTypeKV,
                                typename Params::DTypeO,
                                typename Params::IdType,
                                AttentionVariant>,
          LEFT_SLIDING_WINDOW,
          CAUSAL,
          SAME_SCHEDULE_FOR_ALL_HEADS,
          Params,
          MULTIITEMSCORING>(params, stream);
    }
  } else {
    return cudaErrorNotSupported;
  }
  cudaError_t status = cudaGetLastError();
  return status;
};

void multitoken_paged_attention_hopper(
    torch::Tensor qkv,
    torch::Tensor paged_k_cache,
    torch::Tensor paged_v_cache,
    torch::Tensor output,
    torch::Tensor qo_indptr_buffer,
    torch::Tensor paged_kv_indptr_buffer,
    torch::Tensor paged_kv_indices_buffer,
    torch::Tensor paged_kv_last_page_len_buffer,
    int request_id,
    bool qk_norm,
    bool rope,
    torch::optional<torch::Tensor> q_norm_weight = torch::nullopt,
    torch::optional<torch::Tensor> k_norm_weight = torch::nullopt,
    torch::optional<torch::Tensor> cos = torch::nullopt,
    torch::optional<torch::Tensor> sin = torch::nullopt,
    float q_eps = 0.0f,
    float k_eps = 0.0f) {
  void *qkv_ptr = qkv.data_ptr();
  void *paged_k_cache_ptr = paged_k_cache.data_ptr();
  void *paged_v_cache_ptr = paged_v_cache.data_ptr();
  void *output_ptr = output.data_ptr();
  int const *qo_indptr_buffer_ptr = qo_indptr_buffer.data_ptr<int>();
  int const *paged_kv_indptr_buffer_ptr =
      paged_kv_indptr_buffer.data_ptr<int>();
  int const *paged_kv_indices_buffer_ptr =
      paged_kv_indices_buffer.data_ptr<int>();
  int const *paged_kv_last_page_len_buffer_ptr =
      paged_kv_last_page_len_buffer.data_ptr<int>();

  void const *q_norm_weight_ptr = qk_norm ? q_norm_weight->data_ptr() : nullptr;
  void const *k_norm_weight_ptr = qk_norm ? k_norm_weight->data_ptr() : nullptr;
  void const *cos_ptr = rope ? cos->data_ptr() : nullptr;
  void const *sin_ptr = rope ? sin->data_ptr() : nullptr;
  int const qo_heads = 4;
  int const kv_heads = 1;
  int const head_dim = 128;
  int const qkv_stride = (qo_heads + 2 * kv_heads) * head_dim;
  assert(qkv_stride == qkv.stride(0));
  int const kv_stride = head_dim * kv_heads;
  assert(kv_stride == paged_k_cache.stride(1));
  int const o_stride = head_dim * qo_heads;
  int const page_size = 4096;
  int const max_seq_len = 512;

  launch_multitoken_paged_attention_hopper<bfloat16,
                                           qo_heads,
                                           kv_heads,
                                           kv_stride,
                                           qkv_stride,
                                           o_stride,
                                           head_dim,
                                           max_seq_len,
                                           page_size>(
      qkv_ptr,
      paged_k_cache_ptr,
      paged_v_cache_ptr,
      output_ptr,
      qo_indptr_buffer_ptr,
      paged_kv_indptr_buffer_ptr,
      paged_kv_indices_buffer_ptr,
      paged_kv_last_page_len_buffer_ptr,
      request_id,
      qk_norm,
      rope,
      q_norm_weight_ptr,
      k_norm_weight_ptr,
      cos_ptr,
      sin_ptr,
      q_eps,
      k_eps);

  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    printf("CUDA kernel launch error: %s\n", cudaGetErrorString(err));
  }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("multitoken_paged_attention_hopper", &multitoken_paged_attention_hopper, "Multitoken Paged Attention Hopper");
}