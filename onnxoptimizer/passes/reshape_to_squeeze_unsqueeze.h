/* SPDX-License-Identifier: Apache-2.0 */

#pragma once

#include <vector>

#include "onnxoptimizer/pass.h"
#include "onnxoptimizer/passes/pass_util.h"
#include "onnxoptimizer/passes/logging.h"

namespace ONNX_NAMESPACE {
namespace optimization {

struct ReplaceReshapeWithSqueezeUnsqueeze final : public PredicateBasedPass {
  explicit ReplaceReshapeWithSqueezeUnsqueeze()
      : PredicateBasedPass(PassType::Nop,
                           PassEfficiency::Partial,
                           PassOptimizationType::Compute) {}

  std::string getPassName() const override {
    return "replace_reshape_with_squeeze_unsqueeze";
  }

  static bool IsAxesAnAttr(const Graph &graph) {
    const int opset_version = getOpsetVersion(graph);
    return opset_version <= 12 && opset_version != 0;
  }

  bool patternMatchPredicate(Node *node) override {


    
    if (node->kind() != kReshape)
      return false;


    if (node->inputs().size() != 2)
      return false;

    // Shape must be constant
    if (!IsConstantTensor(node, 1))
      return false;
    
    

    const auto &old_shape = node->inputs()[0]->sizes();

    if (old_shape.empty())
      return false;
    
    // Ensure fully known shape
    for (const auto &d : old_shape)
      if (!d.is_int)
        return false;

    return true;
  }

  // ------------------------------------------------------------
  // Compute inserted axes for UNSQUEEZE
  // ------------------------------------------------------------
  static bool computeUnsqueezeAxes(
      const std::vector<Dimension> &old_shape,
      const std::vector<int64_t> &new_shape,
      std::vector<int64_t> &axes) {
    axes.clear();

    size_t i = 0, j = 0;
    while (i < old_shape.size() && j < new_shape.size()) {
      if (!old_shape[i].is_int)
        return false;

      int64_t old_dim = old_shape[i].dim;
      int64_t new_dim = new_shape[j];

      if (new_dim == old_dim) {
        ++i;
        ++j;
      } else if (new_dim == 1) {
        axes.push_back(j);
        ++j;
      } else {
        return false;
      }
    }

    if (i != old_shape.size())
      return false;

    while (j < new_shape.size()) {
      if (new_shape[j] != 1)
        return false;
      axes.push_back(j);
      ++j;
    }

    return !axes.empty();
  }

  // ------------------------------------------------------------
  // Compute removed axes for SQUEEZE
  // ------------------------------------------------------------
  static bool computeSqueezeAxes(
      const std::vector<Dimension> &old_shape,
      const std::vector<int64_t> &new_shape,
      std::vector<int64_t> &axes) {
    axes.clear();

    size_t i = 0, j = 0;
    while (i < old_shape.size()) {
      if (!old_shape[i].is_int)
        return false;

      int64_t old_dim = old_shape[i].dim;
      int64_t new_dim = (j < new_shape.size() ? new_shape[j] : -1);

      if (old_dim == 1) {
        if (j >= new_shape.size() || new_dim != 1) {
          axes.push_back(i);
          ++i;
        } else {
          ++i;
          ++j;
        }
      } else {
        if (j >= new_shape.size() || new_dim != old_dim)
          return false;
        ++i;
        ++j;
      }
    }

    return (j == new_shape.size()) && !axes.empty();
  }

  // ------------------------------------------------------------
  // RUN TRANSFORM
  // ------------------------------------------------------------
  bool runTransform(Node *node,
                    Graph &graph,
                    NodeDestroyType &destroy_current) override {
    const auto &old_shape = node->inputs()[0]->sizes();

    const auto *shape_input = node->inputs()[1];
            

    const Tensor *shape_tensor = FetchConstantTensor(shape_input);
    if (!shape_tensor)
      return false;

    if (shape_tensor->elem_type() != TensorProto_DataType_INT64)
      return false;
    
    
    std::vector<int64_t> new_shape = ParseTensorData<int64_t>(shape_tensor);
    if (new_shape.empty())
      return false;

    std::vector<int64_t> axes;
    bool use_unsqueeze = false;

    if (new_shape.size() > old_shape.size()) {
      if (!computeUnsqueezeAxes(old_shape, new_shape, axes))
        return false;
      use_unsqueeze = true;

    } else if (new_shape.size() < old_shape.size()) {
      if (!computeSqueezeAxes(old_shape, new_shape, axes))
        return false;
      use_unsqueeze = false;

    } else {
      return false;
    }

    Node *new_node =
        graph.create(use_unsqueeze ? kUnsqueeze : kSqueeze, 1);

    new_node->addInput(node->inputs()[0]);
    new_node->output()->copyMetadata(node->output());
    new_node->insertAfter(node);

    if (IsAxesAnAttr(graph)) {
      new_node->is_(kaxes, std::move(axes));  // FIXED
    } else {
      Tensor t;
      t.elem_type() = TensorProto_DataType_INT64;
      t.sizes().push_back(axes.size());
      t.int64s() = axes;

      Value *axes_v = graph.addInitializerAndCreateValue(t);
      new_node->addInput(axes_v);
    }

    if (!tryReplacingAllUsesWith(node->output(), new_node->output())) {
      new_node->destroy();
      return false;
    }

    auto *shape_v = node->inputs()[1];
    if (shape_v->uses().size() == 0 && shape_v->node()->kind() == kConstant)
      shape_v->node()->destroy();
    destroy_current = NodeDestroyType::DestroyOne;
    return true;
  }
};

} // namespace optimization
} // namespace ONNX_NAMESPACE
