/* SPDX-License-Identifier: Apache-2.0 */
// EXPERIMENTAL pass: hoist common per-branch prefix before Split.

#pragma once

#include <algorithm>
#include <cstdint>
#include <vector>

#include "onnxoptimizer/pass.h"
#include "onnxoptimizer/passes/pass_util.h"
#include "onnxoptimizer/passes/tensor_util.h"

namespace ONNX_NAMESPACE {
namespace optimization {

struct HoistCommonPrefixThroughSplit final : public FullGraphBasedPass {
  explicit HoistCommonPrefixThroughSplit()
      : FullGraphBasedPass(
            PassType::Fuse,
            PassEfficiency::Partial,
            PassOptimizationType::Compute) {}

  std::string getPassName() const override {
    return "hoist_common_prefix_through_split";
  }

  PassAnalysisType getPassAnalysisType() const override {
    return PassAnalysisType::Empty;
  }

private:
  static bool isUnaryElementwise(Symbol k) {
    return k == kSigmoid || k == kTanh || k == kExp || k == kLog ||
           k == kSqrt || k == kabs || k == kNeg || k == kabs || k == kceil;
  }

  static bool isBinaryElementwise(Symbol k) {
    // keep only kAdd if you want to be stricter
    return k == kAdd || k == kSub || k == kMul || k == kDiv;
  }

  static bool isSupported(Symbol k) {
    return isUnaryElementwise(k) || isBinaryElementwise(k) ||
           k == kSqueeze || k == kUnsqueeze;
  }

  static int64_t normalizeAxis(int64_t axis, int64_t rank) {
    if (axis < 0) axis += rank;
    return axis;
  }

  static bool isStaticRank(const std::vector<Dimension>& s) {
    return !s.empty(); // rank known if sizes() not empty in this IR
  }

  static bool isStaticShape(const std::vector<Dimension>& s) {
    if (s.empty()) return false;
    for (const auto& d : s) if (!d.is_int) return false;
    return true;
  }

  static size_t otherIdx(size_t split_idx) { return split_idx == 0 ? 1 : 0; }

  static bool hasNoAttrs(Node* n) {
    return n->attributeNames().empty();
  }

  static bool getSplitInfo(Node* split, int64_t& axis, std::vector<int64_t>& sizes) {
    sizes.clear();
    axis = 0;
    if (split->hasAttribute(kaxis)) axis = split->i(kaxis);
    if (split->hasAttribute(ksplit)) {
      auto tmp = split->is(ksplit);
      sizes.assign(tmp.begin(), tmp.end());
    }
    return true;
  }

  // Extract axes list for squeeze/unsqueeze, supporting:
  // - attribute kaxes (older)
  // - input[1] constant tensor int64 (newer)
  static bool getAxesFromNode(Node* n, std::vector<int64_t>& axes_out) {
    axes_out.clear();

    if (n->hasAttribute(kaxes)) {
      auto tmp = n->is(kaxes);
      axes_out.assign(tmp.begin(), tmp.end());
      return true;
    }

    // opset 13+: axes is second input
    if (n->inputs().size() >= 2 && IsConstantTensor(n->inputs()[1])) {
      const Tensor* t = FetchConstantTensor(n->inputs()[1]);
      if (!t) return false;
      if (t->elem_type() != TensorProto_DataType_INT64) return false;
      axes_out = ParseTensorData<int64_t>(t);
      return !axes_out.empty();
    }

    return false;
  }

  static bool axesDoNotTouchSplitAxis(
      Node* split,
      int64_t split_axis_attr,
      Node* sq_or_unsq) {

    const auto& Xshape = split->inputs()[0]->sizes();
    //std::cout<<"Xshape size "<<Xshape.size()<<std::endl;
    if (!isStaticRank(Xshape)) return false;
    //std::cout<<"2.1"<<std::endl;
    const int64_t rank = static_cast<int64_t>(Xshape.size());
    int64_t split_axis = normalizeAxis(split_axis_attr, rank);
    if (split_axis < 0 || split_axis >= rank) return false;
    //std::cout<<"2.2"<<std::endl;
 
    std::vector<int64_t> axes;
    if (!getAxesFromNode(sq_or_unsq, axes)) return false;
    //std::cout<<"2.3"<<std::endl;
    for (int64_t a : axes) {
      int64_t na = normalizeAxis(a, rank);
      if (na == split_axis) return false;
    }
    return true;
  }


  static bool isBroadcastableTo(const std::vector<Dimension>& from,
    const std::vector<Dimension>& to) {
  // Require static shapes for safety (as you already do)
  if (!isStaticShape(from) || !isStaticShape(to)) return false;

  int64_t i = static_cast<int64_t>(from.size()) - 1;
  int64_t j = static_cast<int64_t>(to.size()) - 1;

  while (j >= 0) {
  const int64_t td = to[static_cast<size_t>(j)].dim;
  const int64_t fd = (i >= 0) ? from[static_cast<size_t>(i)].dim : 1;

  if (!(fd == 1 || fd == td)) return false;

  --i;
  --j;
  }

  // Any remaining leading dims in "from" must be 1
  while (i >= 0) {
  if (from[static_cast<size_t>(i)].dim != 1) return false;
  --i;
  }

  return true;
  }

  // Concatenate constant tensors along split axis, requiring each constant tensor shape
  // matches corresponding split output shape (your conservative rule).
  static bool buildConcatOfBranchConstants(
      Graph& graph,
      Node* split,
      int64_t axis,
      const std::vector<Value*>& branch_consts,
      Value*& concat_out) {

    const auto& split_outs = split->outputs();
    if (branch_consts.size() != split_outs.size()) return false;
    //std::cout << "branch_consts size " << branch_consts.size() << std::endl;
    for (size_t i = 0; i < branch_consts.size(); ++i) {
      const Tensor* t = FetchConstantTensor(branch_consts[i]);
      if (!t) return false;
      //std::cout << "t name " << t->name() << std::endl;
      const auto& cshape = branch_consts[i]->sizes();
      const auto& oshape = split_outs[i]->sizes();
      //std::cout << "cshape size " << cshape.size() << std::endl;
      //std::cout << "oshape size " << oshape.size() << std::endl;

      if (!isBroadcastableTo(cshape, oshape)) return false;
      //std::cout << "after broadcastability check " << std::endl;
      
      
    }

    Node* concat = graph.create(kConcat, /*num_outputs=*/1);
    for (auto* v : branch_consts) concat->addInput(v);



    if (axis>=0) axis-=split_outs[0]->sizes().size();
    concat->i_(kaxis, axis);
    concat->insertBefore(split);
    concat_out = concat->output();
    return true;
  }

  bool tryHoistOnce(Graph& graph, Node* split) {
    if (!split || split->kind() != Symbol(182)) return false;

    //std::cout<<"split->inputs().size() "<<split->inputs().size()<<std::endl;
    

    for (auto in : split->inputs())
        //std::cout<<"in->name() "<<in->uniqueName()<<std::endl;

    // if (split->inputs().size() != 1) return false;
    if (split->outputs().empty()) return false;

    //std::cout<<"split->outputs().size() "<<split->outputs().size()<<std::endl;

    int64_t axis_attr = 0;
    std::vector<int64_t> split_sizes;
    (void)getSplitInfo(split, axis_attr, split_sizes);

    const auto& outs = split->outputs();
    const size_t B = outs.size();

    // First node after each split output must be unique and linear (exactly one user).
    std::vector<Node*> branch_nodes(B, nullptr);
    std::vector<size_t> branch_in_idx(B, 0);

    //std::cout << "B " << B << std::endl;
    for (size_t i = 0; i < B; ++i) {
      const auto& uses = outs[i]->uses();
      //std::cout << "uses size " << uses.size() << std::endl;
      //std::cout << "outs[i].name " << uses[0].user->name() << std::endl;

      if (uses.size() != 1) return false;
      branch_nodes[i] = uses[0].user;
      branch_in_idx[i] = uses[0].offset;
      if (!branch_nodes[i]) return false;
      if (!isSupported(branch_nodes[i]->kind())) return false;
      if (branch_nodes[i]->outputs().size() != 1) return false;
    }

    // Must be the same operator kind across branches, and same split-fed input index.
    const Symbol K = branch_nodes[0]->kind();
    for (size_t i = 1; i < B; ++i) if (branch_nodes[i]->kind() != K) return false;
    for (size_t i = 1; i < B; ++i) if (branch_in_idx[i] != branch_in_idx[0]) return false;



    const size_t split_fed_idx = branch_in_idx[0];

    // Conservative: no attrs for elementwise ops.
    if (isUnaryElementwise(K) || isBinaryElementwise(K)) {
      for (auto* n : branch_nodes) if (!hasNoAttrs(n)) return false;
    }

    

    // Squeeze / Unsqueeze hoist (axes must not include split axis)
    if (K == kSqueeze || K == kUnsqueeze) {
      // Require split-fed is data input (input 0) for these ops.
      if (split_fed_idx != 0) return false;
      //std::cout<<"e uno"<<std::endl;
      // Require all branches have identical axes encoding (same attribute or same constant input value).
      // Simplest conservative rule: require axesDoNotTouchSplitAxis on all and also same axes list.
      std::vector<int64_t> ref_axes;
      if (!getAxesFromNode(branch_nodes[0], ref_axes)) return false;
      //std::cout<<"e due"<<std::endl;
      if (!axesDoNotTouchSplitAxis(split, axis_attr, branch_nodes[0])) return false;
      //std::cout<<"e tre"<<std::endl;
      for (size_t i = 1; i < B; ++i) {
        std::vector<int64_t> axes_i;
        if (!getAxesFromNode(branch_nodes[i], axes_i)) return false;
        if (axes_i != ref_axes) return false;
        if (!axesDoNotTouchSplitAxis(split, axis_attr, branch_nodes[i])) return false;
        //std::cout<<"e quattro"<<std::endl;

      }

      Node* hoisted = graph.create(K, /*num_outputs=*/1);
      hoisted->addInput(split->inputs()[0]);

      // Preserve axes in the same form as branch 0 (attribute if present, otherwise constant input).
      if (branch_nodes[0]->hasAttribute(kaxes)) {
        hoisted->is_(kaxes, std::move(ref_axes));
      } else {
        // If axes were encoded as input[1], reuse the same initializer value from branch 0.
        hoisted->addInput(branch_nodes[0]->inputs()[1]);
      }

      hoisted->insertBefore(split);
      split->replaceInput(0, hoisted->output());

      for (size_t i = 0; i < B; ++i) {
        if (!tryReplacingAllUsesWith(branch_nodes[i]->output(), split->outputs()[i]))
          return false;
      }
      for (auto* n : branch_nodes) n->destroy();
      return true;
    }

    // Unary elementwise hoist
    if (isUnaryElementwise(K)) {
      for (auto* n : branch_nodes) if (n->inputs().size() != 1) return false;

      Node* hoisted = graph.create(K, /*num_outputs=*/1);
      hoisted->addInput(split->inputs()[0]);
      hoisted->insertBefore(split);

      split->replaceInput(0, hoisted->output());

      for (size_t i = 0; i < B; ++i) {
        if (!tryReplacingAllUsesWith(branch_nodes[i]->output(), split->outputs()[i]))
          return false;
      }
      for (auto* n : branch_nodes) n->destroy();
      return true;
    }

    // Binary elementwise hoist (Add/Sub/Mul/Div) with concat-of-constants fallback
    if (isBinaryElementwise(K)) {
      for (auto* n : branch_nodes) if (n->inputs().size() != 2) return false;

      const size_t other = otherIdx(split_fed_idx);


      Value* ref_other = branch_nodes[0]->inputs()[other];
      bool all_same_other = true;
      for (size_t i = 1; i < B; ++i) {
        if (branch_nodes[i]->inputs()[other] != ref_other) { all_same_other = false; break; }
      }


      Value* hoisted_other = nullptr;

      if (all_same_other) {
        hoisted_other = ref_other;
      } else {
        // Require all other operands are constants; then concat them along split axis.
        std::vector<Value*> consts(B, nullptr);
        for (size_t i = 0; i < B; ++i) {
          Value* v = branch_nodes[i]->inputs()[other];
          if (!IsConstantTensor(v)) return false;
          consts[i] = v;
        }
        // Need static rank to normalize split axis; also need static shapes for safety in concat builder.
        const auto& Xshape = split->inputs()[0]->sizes();

        if (!isStaticRank(Xshape)) return false;
        int64_t rank = (int64_t)Xshape.size();
        int64_t ax = normalizeAxis(axis_attr, rank);
        if (ax < 0 || ax >= rank) return false;


        if (!buildConcatOfBranchConstants(graph, split, ax, consts, hoisted_other))
          return false;
      }

      Node* hoisted = graph.create(K, /*num_outputs=*/1);

      Value* old_split_in = split->inputs()[0];

      if (split_fed_idx == 0) {
        hoisted->addInput(old_split_in);
        hoisted->addInput(hoisted_other);
      } else {
        hoisted->addInput(hoisted_other);
        hoisted->addInput(old_split_in);
      }

      hoisted->insertBefore(split);

      Value* new_split_in = hoisted->output();

      // ---- SSA-safe: copyMetadata, then restore the original (fresh) name ----
      const std::string new_name = new_split_in->uniqueName();  // save hoisted output name

      // If your IR has copyMetadata, prefer it:
      new_split_in->copyMetadata(old_split_in);

      new_split_in->setUniqueName(new_name, /*rename=*/false);

      split->replaceInput(0, hoisted->output());

      for (size_t i = 0; i < B; ++i) {
        if (!tryReplacingAllUsesWith(branch_nodes[i]->output(), split->outputs()[i]))
          return false;
      }
      
      //std::cout << "tryHoistOnce " << std::endl;

      for (auto* n : branch_nodes) n->destroy();
      return true;
    }

    return false;
  }

  std::shared_ptr<PostPassAnalysis> runPass(Graph& graph) override {
    // Exactly one hoist per outer iteration; restart scan immediately after mutation.

    for (;;) {
      bool changed = false;

      for (auto it = graph.begin(); it != graph.end(); ++it) {
        Node* n = *it;
        
        if (!n || n->kind() != Symbol(182) ) continue;//Symbol(182) is the SPLIT node
        //std::cout << "n->name " << n->name() << std::endl;
        if (tryHoistOnce(graph, n)) {
          changed = true;
          break; // graph mutated: restart full scan
        }
      }

      if (!changed) break;
    }

    return std::make_shared<PostPassAnalysis>();
  }
};

} // namespace optimization
} // namespace ONNX_NAMESPACE
