/**
 * Copyright (c) Glow Contributors. See CONTRIBUTORS file.
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

#include "BackendTestUtils.h"

#include "glow/ExecutionEngine/ExecutionEngine.h"
#include "glow/Graph/Graph.h"
#include "glow/IR/IR.h"
#include "glow/IR/IRBuilder.h"
#include "glow/IR/Instrs.h"
#include "glow/Optimizer/GraphOptimizer/GraphOptimizer.h"
#include "glow/Quantization/Base/Base.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

#include <functional>
#include <numeric>

using namespace glow;

/// Helper to test ReluSimple using \p DTy.
template <typename DataType>
static void testReluSimple(glow::PlaceholderBindings &bindings,
                           glow::Module &mod, glow::Function *F,
                           glow::ExecutionEngine &EE, ElemKind DTy) {
  auto *in = mod.createPlaceholder(DTy, {7}, "in", false);
  auto *relu = F->createRELU("relu", in);
  auto *save = F->createSave("relu", relu);
  auto *result = bindings.allocate(save->getPlaceholder());

  bindings.allocate(in)->getHandle<DataType>() = {0, -1, -2, -3, 4, 5, 6};

  EE.compile(CompilationMode::Infer);
  EE.run(bindings);

  auto resultH = result->getHandle<DataType>();

  for (size_t i = 0; i < 7; i++) {
    if (i < 4) {
      EXPECT_EQ(resultH.raw(i), static_cast<DataType>(0));
    } else {
      EXPECT_EQ(resultH.raw(i), static_cast<DataType>(i));
    }
  }
}

/// Verify that the RELU operator works correctly for Float.
TEST_P(OperatorTest, ReluSimple_Float) {
  CHECK_IF_ENABLED();

  testReluSimple<float>(bindings_, mod_, F_, EE_, ElemKind::FloatTy);
}

/// Verify that the RELU operator works correctly for Float16.
TEST_P(OperatorTest, ReluSimple_Float16) {
  CHECK_IF_ENABLED();
  testReluSimple<float16_t>(bindings_, mod_, F_, EE_, ElemKind::Float16Ty);
}
