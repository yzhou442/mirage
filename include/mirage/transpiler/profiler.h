/*
 * Copyright (c) 2025 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <cstdint>
#include <string>
#include <utility>

#ifdef MIRAGE_ENABLE_PROFILER
#define MIRAGE_ENABLE_PROFILER_ADDITIONAL_FUNC_PARAMS , void* profiler_buffer
#define MIRAGE_ENABLE_PROFILER_ADDITIONAL_FUNC_PARAMS_ARGS , profiler_buffer
#else
#define MIRAGE_ENABLE_PROFILER_ADDITIONAL_FUNC_PARAMS
#define MIRAGE_ENABLE_PROFILER_ADDITIONAL_FUNC_PARAMS_ARGS
#endif

namespace mirage {
namespace transpiler {

class Profiler {
public:
  static std::pair<std::string, std::string> get_profiling_ptr();
};

// constants
#define PROFILER_CONSTANTS_DECL \
  constexpr uint32_t EVENT_IDX_SHIFT = 2; \
  constexpr uint32_t BLOCK_IDX_SHIFT = 14; \
  \
  constexpr uint32_t EVENT_BEGIN = 0x0; \
  constexpr uint32_t EVENT_END = 0x1; \
  constexpr uint32_t EVENT_INSTANT = 0x2;


// helper functions 
#define PROFILER_HELPER_FUNCTIONS_DECL \
  __device__ __forceinline__ uint32_t get_block_idx() { \
    return (blockIdx.z * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x; \
  } \
  \
  __device__ __forceinline__ uint32_t get_num_blocks() { \
    return gridDim.x * gridDim.y * gridDim.z; \
  } \
  \
  __device__ __forceinline__ uint32_t get_thread_idx() { \
    return (threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x; \
  } \
  \
  __device__ __forceinline__ uint32_t encode_tag(uint32_t block_idx, uint32_t event_idx, \
                                               uint32_t event_type) { \
    return (block_idx << BLOCK_IDX_SHIFT) | (event_idx << EVENT_IDX_SHIFT) | event_type; \
  } \
  \
  __device__ __forceinline__ uint32_t get_timestamp() { \
    volatile uint32_t ret; \
    asm volatile("mov.u32 %0, %globaltimer_lo;" : "=r"(ret)); \
    return ret; \
  }

// ProfilerEntry structure
#define PROFILER_ENTRY_DECL \
  struct ProfilerEntry { \
    union { \
      struct { \
        uint32_t nblocks; \
        uint32_t ngroups; \
      }; \
      struct { \
        uint32_t tag; \
        uint32_t delta_time; \
      }; \
      uint64_t raw; \
    }; \
  };

#ifdef MIRAGE_ENABLE_PROFILER
#define PROFILER_ADDITIONAL_FUNC_PARAMS , void* profiler_buffer
#define PROFILER_ADDITIONAL_PARAMS_SETTER profiler_buffer_ptr = static_cast<uint64_t*>(profiler_buffer);

#define PROFILER_INCLUDE_ALL_DECL \
  PROFILER_CONSTANTS_DECL \
  PROFILER_HELPER_FUNCTIONS_DECL \
  PROFILER_ENTRY_DECL


#define PROFILER_CLOSURE_PARAMS_DECL \
  uint64_t* profiler_write_ptr;      \
  uint32_t profiler_write_stride;    \
  uint32_t profiler_entry_tag_base;  \
  bool profiler_write_thread_predicate;

#define PROFILER_PARAMS_DECL uint64_t* profiler_buffer_ptr;

#define PROFILER_INIT(profiler_buffer,                     \
                      write_thread_predicate)                                                   \
  volatile ProfilerEntry entry;                                                                 \
  if (get_thread_idx() == 0) {                                          \
    entry.nblocks = get_num_blocks();                                                           \
    profiler_buffer[0] = entry.raw;                                                      \
  }                                                                                             \
  profiler_write_ptr =                                                                  \
      profiler_buffer + 1 + get_block_idx();                    \
  profiler_write_stride = get_num_blocks();                                \
  profiler_entry_tag_base = encode_tag(get_block_idx(), 0, 0); \
  profiler_write_thread_predicate = write_thread_predicate;

#define PROFILER_EVENT_START(event)                                                  \
  if (profiler_write_thread_predicate) {                                              \
    entry.tag =                                                                               \
        profiler_entry_tag_base | ((uint32_t)event << EVENT_IDX_SHIFT) | EVENT_BEGIN; \
    entry.delta_time = get_timestamp();                                                       \
    *profiler_write_ptr = entry.raw;                                                  \
    profiler_write_ptr += profiler_write_stride;                              \
  }                                                                                           \
  __threadfence_block();

#define PROFILER_EVENT_END(event)                                                  \
  __threadfence_block();                                                                    \
  if (profiler_write_thread_predicate) {                                            \
    entry.tag =                                                                             \
        profiler_entry_tag_base | ((uint32_t)event << EVENT_IDX_SHIFT) | EVENT_END; \
    entry.delta_time = get_timestamp();                                                     \
    *profiler_write_ptr = entry.raw;                                                \
    profiler_write_ptr += profiler_write_stride;                            \
  }

#define PROFILER_EVENT_INSTANT(event)                                                  \
  __threadfence_block();                                                                        \
  if (profiler_write_thread_predicate) {                                                \
    entry.tag =                                                                                 \
        profiler_entry_tag_base | ((uint32_t)event << EVENT_IDX_SHIFT) | EVENT_INSTANT; \
    entry.delta_time = get_timestamp();                                                         \
    *profiler_write_ptr = entry.raw;                                                    \
  }                                                                                             \
  __threadfence_block();

#else

#define PROFILER_ADDITIONAL_FUNC_PARAMS
#define PROFILER_ADDITIONAL_PARAMS_SETTER

#define PROFILER_INCLUDE_ALL_DECL
#define PROFILER_CLOSURE_PARAMS_DECL
#define PROFILER_PARAMS_DECL
#define PROFILER_INIT(profiler_buffer, write_thread_predicate)
#define PROFILER_EVENT_START(event)
#define PROFILER_EVENT_END(event)
#define PROFILER_EVENT_INSTANT(event)

#endif


} // namespace transpiler
} // namespace mirage
