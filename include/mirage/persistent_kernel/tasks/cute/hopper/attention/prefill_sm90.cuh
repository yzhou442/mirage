/* Copyright 2025 CMU
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
// borrowed from flashinfer:
// https://github.com/flashinfer-ai/flashinfer/blob/bbb57add5affe44e5df87ecd2c97656108ef1330/include/flashinfer/attention/hopper/prefill_sm90.cuh

#include <cuda.h>
#include <cuda_device_runtime_api.h>
#include <cutlass/arch/reg_reconfig.h>
#include <cutlass/array.h>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/numeric_types.h>

#include <type_traits>
#include <vector>

#include "../../cutlass_utils.cuh"
#include "../../exception.h"
#include "../mask.cuh"
#include "cute/tensor.hpp"
#include "cutlass/pipeline/pipeline.hpp"
#include "epilogue.cuh"
#include "kernel_traits.cuh"
#include "mainloop.cuh"
#include "mainloop_mma.cuh"
#include "tile_scheduler.cuh"
#include "utils.cuh"

namespace kernel {

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
      SparseCollectiveMainloop<typename Params::AdditionalParams,
                               KernelTraits,
                               CAUSAL,
                               MULTIITEMSCORING>;
  using CollectiveEpilogue = CollectiveEpilogue<KernelTraits>;
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
           params.kv_indices,
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

} // namespace kernel