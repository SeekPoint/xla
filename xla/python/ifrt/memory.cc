/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "xla/python/ifrt/memory.h"

#include <optional>
#include <string>
#include <utility>

#include "absl/container/node_hash_set.h"

namespace xla {
namespace ifrt {

namespace {

// Global state that keeps a stable copy of memory kind strings for `MemoryKind`
// instances.
struct MemoryKindsSet {
  absl::Mutex mu;
  absl::node_hash_set<std::string> memory_kinds_set ABSL_GUARDED_BY(mu);
};

}  // namespace

MemoryKind::MemoryKind(std::optional<absl::string_view> memory_kind) {
  static auto* const global_set = new MemoryKindsSet();
  absl::MutexLock lock(&global_set->mu);
  auto it = global_set->memory_kinds_set.find(*memory_kind);
  if (it == global_set->memory_kinds_set.end()) {
    memory_kind_ =
        *global_set->memory_kinds_set.insert(std::string(*memory_kind)).first;
  } else {
    memory_kind_ = *it;
  }
}

std::string MemoryKind::DebugString() const {
  if (memory_kind_.has_value()) {
    return std::string(*memory_kind_);
  }
  return "(default)";
}

}  // namespace ifrt
}  // namespace xla
