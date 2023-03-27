/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "xla/service/gpu/runtime/topk_kernel.h"

#include "xla/stream_executor/gpu/gpu_stream.h"
#include "xla/stream_executor/gpu/gpu_types.h"

namespace xla::gpu {

using ::tensorflow::se::gpu::GpuStreamHandle;
using ::xla::runtime::ffi::FfiStatus;
using ::xla::runtime::ffi::PrimitiveType;

namespace internal {

// Per thread buffer:
constexpr size_t kNumStaging = 3;
__host__ __device__ constexpr size_t PerThreadElements(size_t k) {
  return k * kNumStaging;
}

constexpr size_t NumThreads(size_t n, size_t k) {
  // const int batch_length = n / batch_size;
  return std::min(1024UL, n / k);
}

template <typename T, typename V>
struct Descending {
  class KVT {
   public:
    __device__ KVT() = default;
    __device__ KVT& operator=(const KVT&) = default;
    __device__ KVT& operator=(KVT&&) = default;
    __device__ KVT(const KVT&) = default;
    __device__ KVT(KVT&&) = default;

    __device__ KVT(T k, V v) : k_(k), v_(v) {}
    __forceinline__ __device__ void Write(T* key, uint32_t* value) const {
      *key = k_;
      *value = v_;
    }

   private:
    T k_;
    V v_;
    friend class Descending<T, V>;
  } __attribute__((packed));

  static_assert(sizeof(T) + sizeof(V) == sizeof(KVT));

  __device__ __forceinline__ static constexpr bool Gt(const KVT& lhs,
                                                      const KVT& rhs) {
    return lhs.k_ == rhs.k_ ? lhs.v_ < rhs.v_ : lhs.k_ > rhs.k_;
  }
};

template <>
struct Descending<Eigen::bfloat16, uint16_t> {
  using T = Eigen::bfloat16;
  using V = uint16_t;
  class KVT {
   public:
    __device__ KVT() = default;
    __device__ KVT& operator=(const KVT&) = default;
    __device__ KVT& operator=(KVT&&) = default;
    __device__ KVT(const KVT&) = default;
    __device__ KVT(KVT&&) = default;

    __device__ KVT(T k, V v) {
      memcpy(&kv_, &k, sizeof(T));
      kv_ ^= 0x8000;
      kv_ <<= 16;
      kv_ = kv_ | (0xffff - v);
    }
    __forceinline__ __device__ void Write(T* key, uint32_t* value) const {
      uint16_t tmp = (kv_ >> 16) ^ 0x8000;
      memcpy(key, &tmp, sizeof(tmp));
      *value = 0xffff - (kv_ & 0xffff);
    }

   private:
    uint32_t kv_;
    friend class Descending<T, V>;
  } __attribute__((packed));

  static_assert(sizeof(uint32_t) == sizeof(KVT));
  __device__ __forceinline__ static constexpr bool Gt(const KVT& lhs,
                                                      const KVT& rhs) {
    // UB-safe bitcast. clang knows how to handle this.
    uint32_t ilhs = 0;
    uint32_t irhs = 0;
    memcpy(&irhs, &rhs, sizeof(irhs));
    memcpy(&ilhs, &lhs, sizeof(ilhs));
    return lhs.kv_ > rhs.kv_;
  }
};

template <typename KT,
          template <typename T, typename V> class Traits = Descending>
class TopK {
 public:
  // TODO(doak): Choose VT based on value type.
  using VT = uint16_t;
  using Trait = Traits<KT, VT>;
  using KVT = typename Trait::KVT;

  __device__ TopK(KVT* buffer, int k) : buffer_(buffer), k_(k) {}

  __device__ void Init(KT* keys, int n) {
    for (int i = 0; i < n; ++i) {
      int offset = i * blockDim.x + threadIdx.x;
      Assign(i, 0, KVT(keys[offset], static_cast<VT>(offset)));
    }
    Sort(0);
  }

  __device__ void Push(KT* keys, int n, int start) {
    for (int i = 0; i < n; ++i) {
      int offset = start + i * blockDim.x + threadIdx.x;
      Assign(i, 1, KVT(keys[offset], static_cast<VT>(offset)));
    }
    Sort(1);
    MergeLocal();
  }

  __device__ void Dump(KT* keys, uint32_t* values) {
    MergeThreadBuffers();
    if (threadIdx.x != 0) return;
    for (int i = 0; i < k_; ++i) {
      KVT kv = GetKV(i, source_);
      kv.Write(&keys[i], &values[i]);
    }
  }

  int k() const { return k_; }

 private:
  __device__ __forceinline__ int Index(int logical_index, int bank,
                                       int thread_id) {
    int all_threads = blockDim.x * k_;
    int idx = bank * all_threads + logical_index * blockDim.x + thread_id;
    return idx;
  }

  __device__ __forceinline__ KVT GetKV(int i, int bank) {
    return buffer_[Index(i, bank, threadIdx.x)];
  }

  __device__ __forceinline__ KVT GetKV(int i, int bank, int thread_id) {
    return buffer_[Index(i, bank, thread_id)];
  }

  __device__ __forceinline__ void Assign(int i, int bank, const KVT& kv) {
    buffer_[Index(i, bank, threadIdx.x)] = kv;
  }

  __device__ void MergeThreadBuffers() {
    // log2(num_threads)
    const size_t num_iterations = (63 - __clzll(blockDim.x));
    // At each iteration, we want all indexes 2**i to steal from the next
    // unstolen index,
    __syncthreads();
    for (int i = 0; i < num_iterations; ++i) {
      // bitmask of size i, 2*(i+1) -1.
      const uint64_t mask = (1 << (i + 1)) - 1;
      if ((threadIdx.x & mask) != 0) continue;
      int victim = threadIdx.x + (1 << i);
      MergeFrom(victim);
      __syncthreads();
    }
  }

  // Merges buffers staging[1] and staging[source_] into the remainder buffer;
  __device__ void MergeLocal() {
    int lhs_bank = source_;
    // Ping pong between 0 and 2;
    source_ = (source_ + 2) % 4;
    int dst_bank = source_;
    int lhs_idx = 0;
    int rhs_idx = 0;
    for (int i = 0; i < k_; ++i) {
      // Branchless merge.
      const KVT lhs = GetKV(lhs_idx, lhs_bank);
      const KVT rhs = GetKV(rhs_idx, 1);
      const bool cmp = Trait::Gt(lhs, rhs);
      Assign(i, dst_bank, cmp ? lhs : rhs);
      lhs_idx += cmp;
      rhs_idx += !cmp;
    }
  }

  // Merges buffers staging[1] and staging[source_] into the remainder buffer;
  // source_ runs in lockstep with all threads.
  __device__ void MergeFrom(int remote) {
    int src_bank = source_;
    // Ping pong between 0 and 2;
    source_ = (source_ + 2) % 4;
    int dst_bank = source_;
    int lhs_idx = 0;
    int rhs_idx = 0;
    for (int i = 0; i < k_; ++i) {
      // Branchless merge.
      // TODO: This can implemented with one load per iteration.
      const KVT lhs = GetKV(lhs_idx, src_bank);
      const KVT rhs = GetKV(rhs_idx, src_bank, remote);
      const bool cmp = Trait::Gt(lhs, rhs);
      Assign(i, dst_bank, cmp ? lhs : rhs);
      lhs_idx += cmp;
      rhs_idx += !cmp;
    }
  }

  __device__ void Sort(int bank) {
    // TODO(doak): Use optimal sorting networks, see:
    // https://github.com/scandum/fluxsort
    //
    // This horrible sort is enough to beat
    //
    for (int i = 0; i < k_; ++i) {
      KVT largest = GetKV(i, bank);
      int largest_idx = i;
      for (int j = i + 1; j < k_; ++j) {
        KVT right = GetKV(j, bank);
        const bool cmp = Trait::Gt(largest, right);
        largest = cmp ? largest : right;
        largest_idx = cmp ? largest_idx : j;
      }
      KVT t = GetKV(largest_idx, bank);
      KVT m = GetKV(i, bank);
      Assign(largest_idx, bank, m);
      Assign(i, bank, t);
    }
  }

  int source_ = 0;
  KVT* buffer_;
  int k_;
};

template <typename T, typename KVT = typename TopK<T>::KVT>
__global__ void Run(T* data, int slice_size, T* result, uint32_t* result_idxs,
                    int k, KVT* shbuff) {
  // Each thread holds an array of K elements.
  KVT* block_local_shbuff =
      &shbuff[blockIdx.x * blockDim.x * PerThreadElements(k)];
  TopK<T> top_k(block_local_shbuff, k);
  T* block_data = &data[blockDim.x * slice_size * blockIdx.x];
  top_k.Init(block_data, k);
  for (int i = k; i < slice_size; i += k) {
    top_k.Push(block_data, k, blockDim.x * i);
  }
  top_k.Dump(&result[k * blockIdx.x], &result_idxs[k * blockIdx.x]);
}

template <typename T, typename KVT = typename TopK<T>::KVT>
struct TopkArgs {
  TopkArgs(GpuStreamHandle stream, PrimitiveType dtype, T* data,
           size_t num_elements, T* top_elements, uint32_t* top_indices,
           size_t k, size_t batch_size, KVT* scratch_buffer)
      : stream(stream),
        dtype(dtype),
        data(data),
        num_elements(num_elements),
        top_elements(top_elements),
        top_indices(top_indices),
        k(k),
        batch_size(batch_size),
        scratch_buffer(scratch_buffer) {}

  template <typename T2, typename KVT2 = typename TopK<T2>::KVT>
  TopkArgs<T2, KVT2> Convert() const {
    return TopkArgs<T2, KVT2>(stream, dtype, static_cast<T2*>(data),
                              num_elements, static_cast<T2*>(top_elements),
                              top_indices, k, batch_size,
                              static_cast<KVT2*>(scratch_buffer));
  }

  GpuStreamHandle stream;
  PrimitiveType dtype;
  T* data;
  size_t num_elements;
  T* top_elements;
  uint32_t* top_indices;
  size_t k;
  size_t batch_size;
  KVT* scratch_buffer;
};

template <typename T>
FfiStatus TypedTopK(TopkArgs<T> args) {
  int num_threads = NumThreads(args.num_elements, args.k);
  int slice_size = args.num_elements / num_threads;
  void* kernel = reinterpret_cast<void*>(&Run<T>);
  int blocks_per_grid = args.batch_size;
  void* kernel_args[] = {&args.data,        &slice_size, &args.top_elements,
                         &args.top_indices, &args.k,     &args.scratch_buffer};
  cudaError_t launch_status =
      cudaLaunchKernel(kernel, blocks_per_grid, num_threads, kernel_args,
                       /*sharedMem=*/0, args.stream);
  if (launch_status != cudaSuccess)
    return FfiStatus::Internal("Failed to launch kernel");
  return FfiStatus::Ok();
}

}  // namespace internal

size_t NumScratchElements(size_t n, size_t k, size_t batch_size) {
  return internal::NumThreads(n, k) * internal::PerThreadElements(k) *
         batch_size;
}

FfiStatus RunTopk(GpuStreamHandle stream, PrimitiveType dtype, void* data,
                  size_t num_elements, void* top_elements,
                  uint32_t* top_indices, size_t k, size_t batch_size,
                  void* scratch) {
  VLOG(2) << "TopK: " << PrimitiveTypeToString(dtype) << ", n: " << num_elements
          << ", k: " << k << ", bs: " << batch_size;
  auto args = internal::TopkArgs<void, void>(stream, dtype, data, num_elements,
                                             top_elements, top_indices, k,
                                             batch_size, scratch);
  switch (dtype) {
    case PrimitiveType::F32:
      return internal::TypedTopK(args.Convert<float>());
    case PrimitiveType::BF16:
      return internal::TypedTopK(args.Convert<Eigen::bfloat16>());
    default:
      return FfiStatus::Internal("GpuTopK not implemented for this dtype");
  }
}

}  // namespace xla::gpu
