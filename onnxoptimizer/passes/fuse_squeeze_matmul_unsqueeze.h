/* SPDX-License-Identifier: Apache-2.0 */

#pragma once

#include <vector>
#include <cstddef>

#include "onnxoptimizer/pass.h"
#include "onnxoptimizer/passes/pass_util.h"

namespace ONNX_NAMESPACE {
namespace optimization {

// /**
//  * Fuse pattern:
//  *
//  *   X --(1)--> Squeeze(axis=a) -> Y -> MatMul -> Z -> Unsqueeze(axis=a) -> W
//  *
//  * into:
//  *
//  *   X --(1)--> MatMul -> W
//  *
//  * i.e. remove the Squeeze / Unsqueeze around MatMul when:
//  *   - they use the same single axis,
//  *   - that axis is a statically known dimension == 1 on X,
//  *   - Squeeze output is used only by that MatMul,
//  *   - MatMul output is used only by that Unsqueeze.
//  *
//  * Shape correctness:
//  *   - We copy metadata (shape/type) from Unsqueeze's output to MatMul's
//  *     output, so downstream nodes keep seeing the original shape (e.g. 1x50x16).
//  */
// struct FuseSqueezeMatMulUnsqueeze final : public PredicateBasedPass {
//   explicit FuseSqueezeMatMulUnsqueeze()
//       : PredicateBasedPass(
//             PassType::Fuse,
//             PassEfficiency::Partial,
//             PassOptimizationType::Compute) {}

//   std::string getPassName() const override {
//     return "fuse_squeeze_matmul_unsqueeze";
//   }


  // static bool tryGetConstInt64Tensor(Value* v, Graph& graph, std::vector<int64_t>& out) {
  //   if (!v) return false;
  
  //   Tensor t;
  //   Node* def = v->node();
  
  //   // Case A: produced by Constant
  //   if (def && def->kind() == kConstant) {
  //     if (!def->hasAttribute(kvalue)) return false;
  //     t = def->t(kvalue);
  //   } else {
  //     // Case B: graph initializer (compile-time constant)
  //     auto it = graph.getInitializer(v->uniqueName());
  //     if (it == graph.initializers().end()) return false;
  //     t = *it;
  //   }
  
  //   // Must be INT64 for Squeeze/Unsqueeze axes input
  //   if (t.elem_type() != TensorProto_DataType_INT64) return false;
  
  //   // Compute number of elements (handles scalar and 1-D)
  //   size_t numel = 1;
  //   for (auto d : t.sizes()) {
  //     if (d < 0) return false;
  //     numel *= static_cast<size_t>(d);
  //   }
  
  //   out.resize(numel);
  
  //   // data<int64_t>() is the safest accessor because it works even if the tensor uses raw_data
  //   const int64_t* p = t.data<int64_t>();
  //   if (!p) return false;
  
  //   std::copy(p, p + numel, out.begin());
  //   return true;
  // }
  
//   // Unified "get axes" helper:
//   //  - prefer attribute kaxes if present
//   //  - otherwise expect a constant tensor in input(1)



//   bool patternMatchPredicate(Node* node) override {
//     // Start only from Squeeze nodes; we’ll check the full pattern in runTransform.
//     return node->kind() == kSqueeze;
//   }

//   bool runTransform(Node* squeeze,
//                     Graph& graph,
//                     NodeDestroyType& destroy_current) override {
//     destroy_current = NodeDestroyType::DestroyZero;

//     if (squeeze->kind() != kSqueeze)
//       return false;
//     if (squeeze->outputs().size() != 1)
//       return false;
    

//     std::cout<<" Trasformer of squeeze Squeeze "<<squeeze->name()<<std::endl;
    
    
//     // ---- 1) Get Squeeze axis, ensure exactly one axis ----
//     std::vector<int64_t> sq_axes;
//     if (!getAxes(squeeze, graph, sq_axes))
//       return false;
   
    
//     if (sq_axes.size() != 1)
//       return false;
//     int64_t axis = sq_axes[0];
    
//     const auto& in_shape = squeeze->inputs()[0]->sizes();

//     if (in_shape.empty())
//       return false;  // need some shape info to be safe

//     int64_t rank = static_cast<int64_t>(in_shape.size());

//     for (auto& a : sq_axes)
//       a=+rank;

//     // Normalize negative axis using input rank
    
  

    

//     if (axis < 0)
//       axis += rank;
//     if (axis < 0 || axis >= rank)
//       return false;
    
    
//     // Require that this dimension is statically 1
//     const auto& dim = in_shape[static_cast<size_t>(axis)];
//     if (!dim.is_int || dim.dim != 1)
//       return false;

//       // ---- 2) Squeeze output -> single MatMul use ----
//     Value* sq_out = squeeze->output();
//     const auto& sq_uses = sq_out->uses();
//     if (sq_uses.size() != 1)
//       return false;

//     Node* matmul = sq_uses[0].user;


//     size_t mm_input_idx = sq_uses[0].offset;

//     if (!matmul || matmul->kind() != kMatMul)
//       return false;
//     if (matmul->outputs().size() != 1)
//       return false;

//     // ---- 3) MatMul output -> single Unsqueeze use ----
//     Value* mm_out = matmul->output();
//     const auto& mm_uses = mm_out->uses();
//     if (mm_uses.size() != 1)
//       return false;
    
    

//     Node* unsqueeze = mm_uses[0].user;
//     std::vector<int64_t> un_axes;
//     if (!getAxes(unsqueeze, graph, un_axes))
//       return false;
//     if (un_axes.size() != 1)
//       return false;
    
//     for (auto& a : un_axes)
//       a=+rank;
    
    
//     int64_t un_axis = un_axes[0];

    

    
//     // Must reinsert the same axis
//     if (un_axis != sq_axes[0])
//       return false;

//     // ---- We now have: Squeeze(axis) -> MatMul -> Unsqueeze(axis) ----

//     // 4) Rewrite MatMul to take the original Squeeze input
//     // Squeeze always has the data tensor as input(0). (input(1) is axes in newer opsets.)
//     if (squeeze->inputs().empty() || squeeze->inputs()[0] == nullptr)
//       return false;

//     Value* orig_data = squeeze->inputs()[0];   // or squeeze->input(0) if you prefer

//     matmul->replaceInput(mm_input_idx, orig_data);

//     // 5) Preserve original outer shape:
//     //    copy metadata (shape / type) from Unsqueeze output to MatMul output.
//     Value* un_out = unsqueeze->output();
//     mm_out->copyMetadata(un_out);

//     // 6) Redirect all Unsqueeze consumers to MatMul output
//     if (!tryReplacingAllUsesWith(un_out, mm_out)) {
//       return false;
//     }

//     // 7) Remove Unsqueeze; Squeeze is marked for removal by the framework.
//     unsqueeze->destroy();
//     destroy_current = NodeDestroyType::DestroyOne;  // remove this Squeeze

//     return true;
//   }
// };

struct LiftAddBeforeUnsqueeze final : public PredicateBasedPass {
  explicit LiftAddBeforeUnsqueeze()
      : PredicateBasedPass(PassType::Fuse, PassEfficiency::Partial,
                           PassOptimizationType::Compute) {}

  std::string getPassName() const override {
    return "lift_add_before_unsqueeze";
  }

  bool patternMatchPredicate(Node* node) override {
    return node->kind() == kAdd;
  }

  static bool tryGetConstInt64Tensor(Value* v, Graph& graph, std::vector<int64_t>& out) {
    if (!v) return false;
  
    Tensor t;
    Node* def = v->node();
  
    // Case A: produced by Constant
    if (def && def->kind() == kConstant) {
      if (!def->hasAttribute(kvalue)) return false;
      t = def->t(kvalue);
    } else {
      // Case B: graph initializer (compile-time constant)
      auto it = graph.getInitializer(v->uniqueName());
      if (it == graph.initializers().end()) return false;
      t = *it;
    }
  
    // Must be INT64 for Squeeze/Unsqueeze axes input
    if (t.elem_type() != TensorProto_DataType_INT64) return false;
  
    // Compute number of elements (handles scalar and 1-D)
    size_t numel = 1;
    for (auto d : t.sizes()) {
      if (d < 0) return false;
      numel *= static_cast<size_t>(d);
    }
  
    out.resize(numel);
  
    // data<int64_t>() is the safest accessor because it works even if the tensor uses raw_data
    const int64_t* p = t.data<int64_t>();
    if (!p) return false;
  
    std::copy(p, p + numel, out.begin());
    return true;
  }

  // Reuse your getAxes(...) helper (attribute or const input)
  static bool getAxes(Node* n, Graph& graph, std::vector<int64_t>& axes) {
    if (n->hasAttribute(kaxes)) {
      axes = n->is(kaxes);
      return true;
    }
  
    // opset >= 13 style: axes as 2nd input
    if (n->inputs().size() < 2 || n->inputs()[1] == nullptr) return false;
    return tryGetConstInt64Tensor(n->inputs()[1], graph, axes);
  }

  static int64_t normalizeAxis(int64_t axis, int64_t rank) {
    if (axis < 0) axis += rank;
    return axis;
  }

  static bool isStaticallyOne(const Dimension& d) {
    return d.is_int && d.dim == 1;
  }

  static bool isConstantValue(Value* v, Graph& graph) {
    if (!v) return false;
    if (Node* def = v->node()) {
      if (def->kind() == kConstant) return true;
    }
    return graph.getInitializer(v->uniqueName()) != graph.initializers().end();
  }

  bool runTransform(Node* add, Graph& graph,
                    NodeDestroyType& destroy_current) override {
    destroy_current = NodeDestroyType::DestroyZero;
    if (add->outputs().size() != 1) return false;
    if (add->inputs().size() != 2) return false;

    Value* in0 = add->inputs()[0];
    Value* in1 = add->inputs()[1];

    // Identify Unsqueeze operand
    Node* unsq = nullptr;
    Value* bias = nullptr;
    int unsq_input_idx_in_add = -1;

    if (in0 && in0->node() && in0->node()->kind() == kUnsqueeze) {
      unsq = in0->node(); bias = in1; unsq_input_idx_in_add = 0;
    } else if (in1 && in1->node() && in1->node()->kind() == kUnsqueeze) {
      unsq = in1->node(); bias = in0; unsq_input_idx_in_add = 1;
    } else {
      return false;
    }


    if (!isConstantValue(bias, graph))
      return false;



    if (unsq->outputs().size() != 1) return false;

    // Require Unsqueeze output used only by this Add (otherwise lifting changes other consumers)
    Value* unsq_out = unsq->output();
    if (unsq_out->uses().size() != 1 || unsq_out->uses()[0].user != add) {
      return false;
    }

    // Get axis (single)
    std::vector<int64_t> axes;
    if (!getAxes(unsq, graph, axes)) return false;
    if (axes.size() != 1) return false;

    // Shapes
    Value* z = unsq->inputs()[0];
    if (!z) return false;

    const auto& z_shape = z->sizes();
    const auto& u_shape = unsq_out->sizes();
    const auto& b_shape = bias ? bias->sizes() : std::vector<Dimension>{};

    if (z_shape.empty() || u_shape.empty()) return false;

    int64_t z_rank = (int64_t)z_shape.size();
    int64_t u_rank = (int64_t)u_shape.size();
    if (u_rank != z_rank + 1) return false;

    int64_t axis = normalizeAxis(axes[0], u_rank);
    if (axis < 0 || axis >= u_rank) return false;

    // Conservative broadcast check for bias:
    // If bias rank equals u_rank, require its dim at axis is 1 (so removing that axis doesn't change semantics).
    if (!b_shape.empty()) {
      int64_t b_rank = (int64_t)b_shape.size();
      if (b_rank == u_rank) {
        const auto& bd = b_shape[(size_t)axis];
        if (!isStaticallyOne(bd)) return false;
      } else if (b_rank > u_rank) {
        return false;
      }
      // If b_rank <= z_rank, it will broadcast to Z the same way it did to U (common bias=[M]).
      // If b_rank is between z_rank+1 and u_rank-1, reject for safety.
      if (b_rank > z_rank && b_rank != u_rank) return false;
    }

    // Create new Add: T = Add(Z, bias)
    std::vector<Value*> add_inputs;
    add_inputs.reserve(2);
    add_inputs.push_back(z);
    add_inputs.push_back(bias);

    // Create Add with explicit inputs and 1 output
    Node* lifted_add = graph.create(kAdd, add_inputs, /*num_outputs=*/1);


    
    auto* out = lifted_add->output();
    const std::string out_name = out->uniqueName();
    out->copyMetadata(z);                 // shape/type OK
    out->setUniqueName(out_name, false);  // restore unique name
    
    
    // Replace Unsqueeze input from Z to T
    unsq->replaceInput(0, lifted_add->output());
    
    // Preserve final metadata on Unsqueeze output (but do NOT overwrite its name)
    
    const std::string un_name = unsq_out->uniqueName();
    unsq_out->copyMetadata(add->output());
    unsq_out->setUniqueName(un_name, false);
    
    

    // Redirect all uses of old Add output to Unsqueeze output
    if (!tryReplacingAllUsesWith(add->output(), unsq_out)) return false;

    // Insert lifted_add before unsq to keep topological order
    lifted_add->insertBefore(unsq);

    // Destroy old Add (now dead). DCE can also handle it, but safe to destroy current.
    destroy_current = NodeDestroyType::DestroyOne;
    return true;
  }
};




} // namespace optimization
} // namespace ONNX_NAMESPACE
