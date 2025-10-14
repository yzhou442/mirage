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

#include "cutlass_utils.cuh"
// #include "../../exception.h"
// #include "../mask.cuh"
#include "cute/tensor.hpp"
#include "cutlass/pipeline/pipeline.hpp"
#include "epilogue.cuh"
#include "kernel_traits.cuh"
#include "mainloop.cuh"
#include "mainloop_mma.cuh"
#include "tile_scheduler.cuh"
#include "utils.cuh"

namespace kernel {

template <typename CollectiveMainloop,
          typename CollectiveEpilogue,
          typename Ktraits,
          bool LEFT_SLIDING_WINDOW,
          bool CAUSAL,
          typename TileScheduler,
          bool MULTIITEMSCORING = false>
__global__ void
    __launch_bounds__(Ktraits::NUM_WARPS *cutlass::NumThreadsPerWarp, 1)
        PrefillWithKVCacheKernel(
            CUTE_GRID_CONSTANT
            typename CollectiveMainloop::Params const mainloop_params,
            CUTE_GRID_CONSTANT
            typename CollectiveEpilogue::Params const epilogue_params,
            CUTE_GRID_CONSTANT
            typename TileScheduler::Params const scheduler_params) {
  using DTypeQ = typename Ktraits::DTypeQ;
  using DTypeKV = typename Ktraits::DTypeKV;
  using DTypeO = typename Ktraits::DTypeO;
  using DTypeQKAccum = typename Ktraits::DTypeQKAccum;
  using TileShape_QKD = typename Ktraits::TileShape_QKD;
  using TileShape_PDV = typename Ktraits::TileShape_PDV;
  using AttentionVariant = typename Ktraits::AttentionVariant;

  static constexpr int NUM_MMA_THREADS = Ktraits::NUM_MMA_THREADS;
  static constexpr int NUM_COPY_THREADS = cutlass::NumThreadsPerWarpGroup;
  static constexpr int CTA_Q = Ktraits::CTA_Q;
  static constexpr int CTA_KV = Ktraits::CTA_KV;

  static constexpr bool use_tma_load_kv = CollectiveMainloop::USE_TMA_LOAD_KV;

  using MainloopPipeline = typename CollectiveMainloop::MainloopPipeline;
  using PipelineParams = typename MainloopPipeline::Params;
  using PipelineState = typename MainloopPipeline::PipelineState;

  extern __shared__ char shared_memory[];
  auto &shared_storage =
      *reinterpret_cast<typename Ktraits::SharedStorage *>(shared_memory);

  int const lane_predicate = cute::elect_one_sync();
  int const warp_idx = cutlass::canonical_warp_idx_sync();

  // Issue Tma Descriptor Prefetch from a single thread
  if (warp_idx == 0 && lane_predicate) {
    CollectiveMainloop::prefetch_tma_descriptors(mainloop_params);
    CollectiveEpilogue::prefetch_tma_descriptors(epilogue_params);
  }

  // Obtain warp index
  int const warp_group_thread_idx =
      threadIdx.x % cutlass::NumThreadsPerWarpGroup;

  PipelineParams pipeline_params;
  int warp_group_idx = cutlass::canonical_warp_group_idx();
  pipeline_params.role = warp_group_idx == 0
                             ? MainloopPipeline::ThreadCategory::Producer
                             : MainloopPipeline::ThreadCategory::Consumer;
  if constexpr (use_tma_load_kv) {
    pipeline_params.is_leader = warp_group_thread_idx == 0;
    pipeline_params.num_consumers = NUM_MMA_THREADS;
  } else {
    pipeline_params.producer_arv_count = NUM_COPY_THREADS;
    pipeline_params.consumer_arv_count = NUM_MMA_THREADS;
  }

  if (warp_idx == 0 && lane_predicate) {
    shared_storage.barrier_Q.init(/*num_threads=*/1);
    shared_storage.barrier_O.init(/*num_threads=*/1);
  }
  // We're counting on pipeline_k to call cutlass::arch::fence_barrier_init();
  MainloopPipeline pipeline_k = [&] {
    if constexpr (use_tma_load_kv) {
      pipeline_params.transaction_bytes =
          CollectiveMainloop::TmaTransactionBytesK;
      return MainloopPipeline(shared_storage.pipeline_k,
                              pipeline_params,
                              /*cluster_shape=*/Shape<_1, _1, _1>{});
    } else {
      return MainloopPipeline(shared_storage.pipeline_k, pipeline_params);
    }
  }();

  MainloopPipeline pipeline_v = [&] {
    if constexpr (use_tma_load_kv) {
      pipeline_params.transaction_bytes =
          CollectiveMainloop::TmaTransactionBytesV;
      return MainloopPipeline(shared_storage.pipeline_v,
                              pipeline_params,
                              /*cluster_shape=*/Shape<_1, _1, _1>{});
    } else {
      return MainloopPipeline(shared_storage.pipeline_v, pipeline_params);
    }
  }();

  CollectiveMainloop collective_mainloop;
  CollectiveEpilogue collective_epilogue;

  // We need this to guarantee that the Pipeline init is visible to all
  // producers and consumer blocks in the Cluster
  __syncthreads();

  uint32_t *maybe_prefix_len_ptr = nullptr;
  if constexpr (has_maybe_prefix_len_ptr_v<
                    decltype(mainloop_params.additional_params)>) {
    maybe_prefix_len_ptr =
        mainloop_params.additional_params.maybe_prefix_len_ptr;
  }
  uint16_t *maybe_token_pos_in_items_ptr = nullptr;
  if constexpr (has_maybe_token_pos_in_items_ptr_v<
                    decltype(mainloop_params.additional_params)>) {
    maybe_token_pos_in_items_ptr =
        mainloop_params.additional_params.maybe_token_pos_in_items_ptr;
  }
  uint32_t token_pos_in_items_len = 0;
  if constexpr (has_token_pos_in_items_len_v<
                    decltype(mainloop_params.additional_params)>) {
    token_pos_in_items_len =
        mainloop_params.additional_params.token_pos_in_items_len;
  }
  uint16_t *maybe_max_item_len_ptr = nullptr;
  if constexpr (has_maybe_max_item_len_ptr_v<
                    decltype(mainloop_params.additional_params)>) {
    maybe_max_item_len_ptr =
        mainloop_params.additional_params.maybe_max_item_len_ptr;
  }

  if (warp_group_idx == 0) { // Producer
    if constexpr (use_tma_load_kv) {
      cutlass::arch::warpgroup_reg_dealloc<Ktraits::NUM_WARPS == 12 ? 24
                                                                    : 32>();
    } else {
      cutlass::arch::warpgroup_reg_dealloc<72>();
    }

    int warp_idx_in_warpgroup =
        __shfl_sync(0xffffffff, (threadIdx.x / 32) % 4, 0);
    if (!use_tma_load_kv || warp_idx_in_warpgroup == 0) { // Load Q, K, V
      PipelineState smem_pipe_write_k =
          cutlass::make_producer_start_state<MainloopPipeline>();
      PipelineState smem_pipe_write_v =
          cutlass::make_producer_start_state<MainloopPipeline>();

      int work_idx = 0;

      TileScheduler scheduler;
      for (auto work_tile_info = scheduler.get_initial_work(scheduler_params);
           work_tile_info.is_valid(scheduler_params);
           work_tile_info =
               scheduler.template get_next_work</*is_producer=*/true>(
                   scheduler_params, work_tile_info)) {
        auto block_coord = work_tile_info.get_block_coord(scheduler_params);
        auto [q_tile_idx,
              qo_head_idx,
              kv_head_idx,
              qo_indptr,
              kv_indptr,
              qo_len,
              kv_len,
              batch_idx] = block_coord;

        if (q_tile_idx * CTA_Q >= qo_len) {
          continue;
        }
        int num_kv_tiles = collective_mainloop.get_num_kv_tiles(
            mainloop_params, q_tile_idx, qo_len, kv_len);
        if (num_kv_tiles <= 0) {
          scheduler.prefetch_next_work(scheduler_params, work_tile_info);
          scheduler.broadcast_next_work(work_tile_info);
          continue;
        }
        int num_kv_tiles_outside_items_window = 0;
        int num_kv_tiles_prefix = 0;
        if constexpr (MULTIITEMSCORING) {
          auto prefix_len = __ldg(maybe_prefix_len_ptr + batch_idx);
          auto max_item_len = __ldg(maybe_max_item_len_ptr + batch_idx);
          auto valid_items_window_len =
              std::max(0, q_tile_idx * CTA_Q + kv_len - qo_len - max_item_len);
          num_kv_tiles_outside_items_window = valid_items_window_len / CTA_KV;
          num_kv_tiles_prefix = cute::ceil_div(prefix_len, CTA_KV);
        }
        if constexpr (MULTIITEMSCORING) {
          collective_mainloop.load<LEFT_SLIDING_WINDOW>(
              mainloop_params,
              pipeline_k,
              pipeline_v,
              smem_pipe_write_k,
              smem_pipe_write_v,
              shared_storage,
              scheduler,
              scheduler_params,
              work_tile_info,
              block_coord,
              work_idx,
              num_kv_tiles_outside_items_window,
              num_kv_tiles_prefix);
        } else {
          collective_mainloop.template load<LEFT_SLIDING_WINDOW>(
              mainloop_params,
              pipeline_k,
              pipeline_v,
              smem_pipe_write_k,
              smem_pipe_write_v,
              shared_storage,
              scheduler,
              scheduler_params,
              work_tile_info,
              block_coord,
              work_idx);
        }
        ++work_idx;
      }
      collective_mainloop.load_tail(
          pipeline_k, pipeline_v, smem_pipe_write_k, smem_pipe_write_v);
    }
  } else { // Consumer
    if constexpr (use_tma_load_kv) {
      cutlass::arch::warpgroup_reg_alloc<Ktraits::NUM_WARPS == 12 ? 240
                                                                  : 160>();
    } else {
      cutlass::arch::warpgroup_reg_alloc<Ktraits::NUM_WARPS == 12 ? 216
                                                                  : 144>();
    }

    TileScheduler scheduler;
    // Initialize matmul objects.
    typename Ktraits::TiledMmaPV tiled_mma_pv;

    PipelineState smem_pipe_read_k, smem_pipe_read_v;
    // We don't need separate variables smem_pipe_release_k and
    // smem_pipe_release_v (like in Cutlass's gemm) because the read and release
    // pipeline states are always the same.

    CollectiveMainloop::WarpScheduler::mma_init();
    scheduler.init_consumer();

    int work_idx = 0;
    CUTLASS_PRAGMA_NO_UNROLL
    for (auto work_tile_info = scheduler.get_initial_work(scheduler_params);
         work_tile_info.is_valid(scheduler_params);
         work_tile_info =
             scheduler.template get_next_work</*is_producer=*/false>(
                 scheduler_params, work_tile_info)) {
      // Attention output (GEMM-II) accumulator.
      Tensor tOrO =
          partition_fragment_C(tiled_mma_pv, select<0, 1>(TileShape_PDV{}));

      auto block_coord = work_tile_info.get_block_coord(scheduler_params);
      auto [q_tile_idx,
            qo_head_idx,
            kv_head_idx,
            qo_indptr,
            kv_indptr,
            qo_len,
            kv_len,
            batch_idx] = block_coord;

      AttentionVariant variant(mainloop_params, block_coord);
      auto attention_updater =
          variant.template GetAttentionUpdater<2 *
                                               (2 * CTA_Q / NUM_MMA_THREADS)>();

      if (q_tile_idx * CTA_Q >= qo_len) {
        continue;
      }
      int num_kv_tiles = collective_mainloop.get_num_kv_tiles(
          mainloop_params, q_tile_idx, qo_len, kv_len);
      if (num_kv_tiles <=
          0) { // We exit early and write 0 to gO and -inf to gLSE.
        collective_epilogue.store_zero(epilogue_params,
                                       shared_storage,
                                       threadIdx.x - NUM_COPY_THREADS,
                                       block_coord);
        continue;
      }

      int swa_begin_kv_tile_idx = 0;
      int swa_end_kv_tile_idx = -1;
      if constexpr (LEFT_SLIDING_WINDOW) {
        swa_begin_kv_tile_idx = get_swa_begin_kv_tile_idx<CTA_Q, CTA_KV>(
            mainloop_params.window_left, q_tile_idx, qo_len, kv_len);
        swa_end_kv_tile_idx = get_swa_end_kv_tile_idx<CTA_Q, CTA_KV>(
            mainloop_params.window_left, q_tile_idx, qo_len, kv_len);
      }

      uint32_t prefix_len = 0;
      uint16_t *token_pos_in_items = nullptr;
      if constexpr (MULTIITEMSCORING) {
        prefix_len = __ldg(maybe_prefix_len_ptr + batch_idx);
        token_pos_in_items =
            maybe_token_pos_in_items_ptr + batch_idx * token_pos_in_items_len;
      }
      int num_kv_tiles_outside_items_window = 0;
      int num_kv_tiles_prefix = 0;
      if constexpr (MULTIITEMSCORING) {
        auto prefix_len = __ldg(maybe_prefix_len_ptr + batch_idx);
        auto max_item_len = __ldg(maybe_max_item_len_ptr + batch_idx);
        auto valid_items_window_len =
            std::max(0, q_tile_idx * CTA_Q + kv_len - qo_len - max_item_len);
        num_kv_tiles_outside_items_window = valid_items_window_len / CTA_KV;
        num_kv_tiles_prefix = cute::ceil_div(prefix_len, CTA_KV);
      }
      mma_f16<Ktraits,
              /*LEFT_SLIDING_WINDOW=*/LEFT_SLIDING_WINDOW,
              CAUSAL,
              MULTIITEMSCORING,
              CollectiveMainloop::WarpScheduler>(
          mainloop_params,
          variant,
          pipeline_k,
          pipeline_v,
          smem_pipe_read_k,
          smem_pipe_read_v,
          tOrO,
          attention_updater,
          num_kv_tiles,
          swa_begin_kv_tile_idx,
          swa_end_kv_tile_idx,
          threadIdx.x - NUM_COPY_THREADS,
          work_idx,
          q_tile_idx,
          shared_storage,
          qo_len,
          kv_len,
          qo_head_idx,
          kv_head_idx,
          prefix_len,
          token_pos_in_items,
          num_kv_tiles_outside_items_window,
          num_kv_tiles_prefix);
      collective_epilogue.store(epilogue_params,
                                tOrO,
                                attention_updater.get_lse(),
                                shared_storage,
                                tiled_mma_pv,
                                threadIdx.x - NUM_COPY_THREADS,
                                block_coord);

      ++work_idx;
    }
    collective_epilogue.store_tail();
  }
}

enum class MaskMode {
  kNone = 0U,    // No mask
  kCausal = 1U,  // Causal mask
  kCustom = 2U,  // Custom mask
  kMultiItemScoring = 3U,
};

} // namespace kernel