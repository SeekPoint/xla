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

#include "xla/service/gpu/runtime/topk.h"

#include <stdint.h>

#include <cstddef>
#include <memory>
#include <string>

#include "absl/container/flat_hash_set.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/dfs_hlo_visitor_with_default.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/runtime/ffi/ffi_api.h"
#include "xla/runtime/ffi/ffi_c_api.h"
#include "xla/service/gpu/runtime/topk_kernel.h"
#include "xla/service/tuple_util.h"
#include "xla/status.h"

namespace xla::gpu {

// StatusOr<size_t>
StatusOr<HloInstruction*> SmallBufferOptimization(
    HloCustomCallInstruction* topk) {
  constexpr size_t KiB = 1024;
  Shape data_shape = topk->operand(0)->shape();
  auto supported_dtypes = {F32, BF16};
  if (!absl::c_linear_search(supported_dtypes, data_shape.element_type()))
    return InvalidArgument("Invalid Dtype");
  // We only support topk of the shape [x] or [batch, x].
  if (data_shape.dimensions_size() > 2)
    return InvalidArgument("Invalid input dimensions");
  const bool has_batch = data_shape.dimensions_size() == 2;
  const size_t n = data_shape.dimensions(has_batch ? 1 : 0);
  const size_t k = topk->shape().tuple_shapes(0).dimensions(has_batch ? 1 : 0);
  const size_t batch_size = has_batch ? data_shape.dimensions(0) : 1;
  // TODO: Write better errors.
  if (k > 128) return InvalidArgument("k is too large");
  if (n > 64 * KiB) return InvalidArgument("Input buffer too big");
  if (n < 128) return InvalidArgument("Input buffer too small");
  if (n % k != 0) return InvalidArgument("k must divide n");
  if (n & (n - 1)) return InvalidArgument("Input has to be a power of 2");
  Shape new_shape = topk->shape();
  // This buffer needs to hold NumScratchElements() {K,V} objects. The biggest
  // they can be is 32-bit each, so we allocate a buffer of 2*N U32.
  const int n_scratch = NumScratchElements(n, k, batch_size) * 2;
  ShapeUtil::AppendShapeToTuple(ShapeUtil::MakeShape(U32, {n_scratch}),
                                &new_shape);
  HloComputation* comp = topk->parent();
  HloInstruction* new_topk =
      comp->AddInstruction(HloInstruction::CreateCustomCall(
          new_shape, topk->operands(),
          // We don't need the original to_apply, but keeping it around allows
          // us to round-trip this CustomCall on tests.
          topk->to_apply(), "GpuTopK",
          /*opaque=*/"", CustomCallApiVersion::API_VERSION_TYPED_FFI));
  return TupleUtil::ExtractPrefix(new_topk, 2);
}

class SpecializeTopkVisitor : public DfsHloRewriteVisitor {
 public:
  Status HandleCustomCall(HloInstruction* inst) override {
    HloCustomCallInstruction* topk = DynCast<HloCustomCallInstruction>(inst);
    if (topk == nullptr || topk->custom_call_target() != "TopK") {
      return OkStatus();
    }
    DCHECK_GE(topk->operand_count(), 1);

    if (auto small_topk = SmallBufferOptimization(topk); small_topk.ok()) {
      return ReplaceInstruction(topk, *small_topk);
    } else {
      LOG(ERROR) << "Small TopK optimization doesn't match: "
                 << small_topk.status();
    }

    return OkStatus();
  }
};

StatusOr<bool> SpecializeTopk::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  return SpecializeTopkVisitor().RunOnModule(module, execution_threads);
}

namespace ffi = ::xla::runtime::ffi;

struct TopkFfiModule : ffi::StatelessModule {
  explicit TopkFfiModule(const XLA_FFI_Api* api)
      : StatelessModule(api, "TopkFfiModule", {{"GpuTopK", FFI_TopK}}) {}

  XLA_FFI_DEFINE_FUNCTION(FFI_TopK, TopK,
                          ffi::Ffi::Binding()
                              .Stream<se::gpu::GpuStreamHandle>()
                              .Arg<ffi::StridedBufferArg>()
                              .Arg<ffi::StridedBufferArg>()
                              .Arg<ffi::StridedBufferArg>()
                              .Arg<ffi::StridedBufferArg>());

  static ffi::FfiStatus TopK(se::gpu::GpuStreamHandle stream,
                             ffi::StridedBufferArg data,
                             ffi::StridedBufferArg top_elements,
                             ffi::StridedBufferArg indices,
                             ffi::StridedBufferArg scratch_elements) {
    // TODO(doak): Better validate these arguments.
    if (data.sizes.size() > 2)
      return ffi::FfiStatus::InvalidArgument("Invalid input shape");
    if (indices.dtype != ffi::PrimitiveType::S32)
      return ffi::FfiStatus::InvalidArgument("Indices should be S32");
    const bool has_batch = data.sizes.size() == 2;
    const size_t batch_size = has_batch ? data.sizes[0] : 1;
    const size_t n = has_batch ? data.sizes[1] : data.sizes[0];
    const size_t k = has_batch ? top_elements.sizes[1] : top_elements.sizes[0];
    return RunTopk(stream, data.dtype, data.data, n, top_elements.data,
                   static_cast<uint32_t*>(indices.data), k, batch_size,
                   scratch_elements.data);
  }
};

XLA_REGISTER_FFI_MODULE(std::make_unique<TopkFfiModule>(GetXlaFfiApi()));

}  // namespace xla::gpu
