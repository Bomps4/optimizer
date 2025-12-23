#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "onnxoptimizer/pass.h"

// ONNX IR types
#include "onnx/common/interned_strings.h"
#include "onnx/defs/schema.h"
#include "onnx/onnx_pb.h"

namespace ONNX_NAMESPACE {
namespace optimization {

// =======================
// C++ detection utilities
// =======================

template <class, class = void>
struct has_copyMetadata : std::false_type {};
template <class T>
struct has_copyMetadata<T, std::void_t<decltype(std::declval<T*>()->copyMetadata(std::declval<T*>()))>>
    : std::true_type {};

template <class, class = void>
struct has_node_f_setter : std::false_type {};
template <class T>
struct has_node_f_setter<T, std::void_t<decltype(std::declval<T*>()->f_(std::declval<ONNX_NAMESPACE::Symbol>(), 0.0f))>>
    : std::true_type {};

template <class, class = void>
struct has_node_i_setter : std::false_type {};
template <class T>
struct has_node_i_setter<T, std::void_t<decltype(std::declval<T*>()->i_(std::declval<ONNX_NAMESPACE::Symbol>(), 0))>>
    : std::true_type {};

template <class, class = void>
struct has_addAttribute : std::false_type {};
template <class T>
struct has_addAttribute<T, std::void_t<decltype(std::declval<T*>()->addAttribute(std::declval<ONNX_NAMESPACE::Symbol>()))>>
    : std::true_type {};

template <class NodeT>
inline void setFloatAttr(NodeT* n, const ONNX_NAMESPACE::Symbol& k, float v) {
  if (!n) return;
  if constexpr (has_node_f_setter<NodeT>::value) {
    // Common ONNX-IR builder API (most forks)
    n->f_(k, v);
  } else if constexpr (has_addAttribute<NodeT>::value) {
    // Fallback for forks that expose addAttribute
    auto* a = n->addAttribute(k);
    a->set_f(v);
  } else {
    // No supported way to set attributes in this build.
    // Leave unset (conservative).
  }
}


template <class NodeT>
inline void setIntAttr(NodeT* n, const ONNX_NAMESPACE::Symbol& k, float v) {
  if (!n) return;
  if constexpr (has_node_i_setter<NodeT>::value) {
    // Common ONNX-IR builder API (most forks)
    n->i_(k, v);
  } else if constexpr (has_addAttribute<NodeT>::value) {
    // Fallback for forks that expose addAttribute
    auto* a = n->addAttribute(k);
    a->set_i(v);
  } else {
    // No supported way to set attributes in this build.
    // Leave unset (conservative).
  }
}



template <class ValueT>
inline void tryCopyMetadata(ValueT* dst, ValueT* src) {
  if (!dst || !src) return;
  std::string dst_name = dst->uniqueName();

  if constexpr (has_copyMetadata<ValueT>::value) {
    // NOTE: To Avoid SSA name collisions, reset old unique name;
    dst->copyMetadata(src);
    dst->setUniqueName(dst_name);
  } else {
    // No metadata API available; skip.
  }
}

inline std::optional<int64_t> extract_int_dimension (const Dimension dimension) {
  if (dimension.is_int) {
    return dimension.dim;
  }
  return std::nullopt;
}

// -------------------------
// Helpers: scalar extraction
// -------------------------
// Read scalar directly from your Tensor wrapper (no TensorProto).

inline bool tensorIsScalarLike(const ONNX_NAMESPACE::Tensor& t) {
  if (t.sizes().empty()) return true;
  return (t.sizes().size() == 1 && t.sizes()[0] == 1);
}

inline bool readScalarFloatFromTensor(const ONNX_NAMESPACE::Tensor& t, float& out) {
  if (!tensorIsScalarLike(t)) return false;
  if (t.elem_num() != 1) return false;

  switch (t.elem_type()) {
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT: {
      const float* p = t.data<float>();
      if (!p) return false;
      out = p[0];
      return true;
    }
    case ONNX_NAMESPACE::TensorProto_DataType_DOUBLE: {
      const double* p = t.data<double>();
      if (!p) return false;
      out = static_cast<float>(p[0]);
      return true;
    }
    default:
      return false; // conservative
  }
}



// Constant scalar float extractor: Constant has attribute "value" of kind t (Tensor).
inline bool isConstFloatScalar(ONNX_NAMESPACE::Value* v, float& out) {
  if (!v) return false;
  auto* n = v->node();
  if (!n || (n->kind() != ONNX_NAMESPACE::kConstant && !IsConstantTensor(v)) ) return false;
  else { 
    
    if (IsConstantTensor(v)){
        //std::cout<<"isConstFloatScalar: IsConstantTensor"<<std::endl;
        const ONNX_NAMESPACE::Tensor* ten = FetchConstantTensor(v);
        return readScalarFloatFromTensor(std::move(*ten), out);
        }
    else {
        static const ONNX_NAMESPACE::Symbol kValue = ONNX_NAMESPACE::Symbol("value");
        if (!n->hasAttribute(kValue)) return false;
        if (n->kindOf(kValue) != ONNX_NAMESPACE::AttributeKind::t) return false;
        const ONNX_NAMESPACE::Tensor& ten = n->t(kValue);
        return readScalarFloatFromTensor(ten, out);
        }
    }
}

// -------------------------
// Optional mask extraction
// -------------------------

struct MaskInfo {
  ONNX_NAMESPACE::Value* scores_no_mask{nullptr};
  ONNX_NAMESPACE::Value* attn_mask{nullptr};
};

inline std::optional<MaskInfo> tryExtractAddMask(ONNX_NAMESPACE::Node* maybe_add) {
  if (!maybe_add || maybe_add->kind() != ONNX_NAMESPACE::kAdd) return std::nullopt;
  if (maybe_add->inputs().size() != 2) return std::nullopt;

  MaskInfo mi;

  // Default ordering: input0=scores, input1=mask
  mi.scores_no_mask = maybe_add->inputs()[0];
  mi.attn_mask      = maybe_add->inputs()[1];

  // If you want to be less permissive, wire your shape inference here.
  return mi;
}

// -------------------------
// Optional scale extraction
// -------------------------

inline std::optional<float> tryExtractScaleFromScoresProducer(ONNX_NAMESPACE::Value*& scores_in_out) {
  if (!scores_in_out) return std::nullopt;
  auto* n = scores_in_out->node();

  //std::cout<<"n "<<n->kind().toString()<<std::endl;

  if (!n) return std::nullopt;

  // A) Mul(qk, s) or Mul(s, qk)
  if (n->kind() == ONNX_NAMESPACE::kMul && n->inputs().size() == 2) {

    //std::cout<<"Mul "<<std::endl;
    float s = 0.0f;
    if (isConstFloatScalar(n->inputs()[0], s)) {
      //std::cout<<"first "<<std::endl;
      scores_in_out = n->inputs()[1];
      return s;
    }
    if (isConstFloatScalar(n->inputs()[1], s)) {
      //std::cout<<"second "<<std::endl;
      scores_in_out = n->inputs()[0];
      return s;
    }
  }

  // B) Div(qk, denom) => scale = 1/denom
  if (n->kind() == ONNX_NAMESPACE::kDiv && n->inputs().size() == 2) {
    float denom = 0.0f;
    if (isConstFloatScalar(n->inputs()[1], denom) && denom != 0.0f) {
      scores_in_out = n->inputs()[0];
      return 1.0f / denom;
    }
  }

  return std::nullopt;
}

// Pre-scale pattern:
// MatMul(Mul(Q,c), Transpose(Mul(K,c))) => scale=c*c
inline std::optional<float> tryExtractPreScaleFromQKMatMul(ONNX_NAMESPACE::Node* qk_mm,
                                                          ONNX_NAMESPACE::Value*& Q,
                                                          ONNX_NAMESPACE::Value*& K) {
  Q = nullptr;
  K = nullptr;

  if (!qk_mm || qk_mm->kind() != ONNX_NAMESPACE::kMatMul) return std::nullopt;
  if (qk_mm->inputs().size() != 2) return std::nullopt;

  auto* q_in = qk_mm->inputs()[0];
  auto* k_in = qk_mm->inputs()[1];
  if (!q_in || !k_in) return std::nullopt;

  // K side must be Transpose(...)
  ONNX_NAMESPACE::Node* k_t = k_in->node();
  if (k_t->inputs().size() != 1 || !k_t->inputs()[0]) return std::nullopt;

  ONNX_NAMESPACE::Value* k_pre_t = k_t->inputs()[0];

  auto* q_mul = q_in->node();
  auto* k_mul = k_pre_t->node();

  const bool q_is_mul = (q_mul && q_mul->kind() == ONNX_NAMESPACE::kMul && q_mul->inputs().size() == 2);
  const bool k_is_mul = (k_mul && k_mul->kind() == ONNX_NAMESPACE::kMul && k_mul->inputs().size() == 2);

  if (q_is_mul && k_is_mul) {
    float cq = 0.0f, ck = 0.0f;

    const bool q_c0 = isConstFloatScalar(q_mul->inputs()[0], cq);
    const bool q_c1 = isConstFloatScalar(q_mul->inputs()[1], cq);
    const bool k_c0 = isConstFloatScalar(k_mul->inputs()[0], ck);
    const bool k_c1 = isConstFloatScalar(k_mul->inputs()[1], ck);

    if ((q_c0 || q_c1) && (k_c0 || k_c1) && cq == ck) {
      Q = q_c0 ? q_mul->inputs()[1] : q_mul->inputs()[0];
      K = k_c0 ? k_mul->inputs()[1] : k_mul->inputs()[0];
      return cq * cq;
    }
  }

  // Fallback: no pre-scale match; still return bare Q/K (K is pre-transpose input).
  Q = q_in;
  K = k_pre_t;
  return std::nullopt;
}


inline void setPermAttr(ONNX_NAMESPACE::Node* t, const std::vector<int64_t> perm) {
  static const ONNX_NAMESPACE::Symbol kPerm = ONNX_NAMESPACE::Symbol("perm");
  t->is_(kPerm, std::move(perm));

}


inline bool findPermToMatchShape(const std::vector<int64_t>& in_shape,
  const std::vector<int64_t>& ref_shape,
  std::vector<int64_t>& perm_out) {
const size_t r = ref_shape.size();
if (in_shape.size() != r) return false;

perm_out.assign(r, -1);
std::vector<char> used(r, 0);

// Backtracking; rank is small (<=6 typically), so this is fast.
std::function<bool(size_t)> dfs = [&](size_t i) -> bool {
if (i == r) return true;

// Try all axes j in input that match desired dim ref_shape[i].
for (size_t j = 0; j < r; ++j) {


if (used[j]) continue;

//std::cout<<"i: "<<i<<std::endl;
//std::cout<<"corresponding j: "<<j<<std::endl;
//std::cout<<"reference shape "<<ref_shape[i]<<std::endl;
//std::cout<<"input shape "<<in_shape[j]<<std::endl;
//std::cout<<std::endl<<std::endl;

if (in_shape[j] != ref_shape[i]) continue;

used[j] = 1;
perm_out[i] = static_cast<int64_t>(j);
for (auto u:perm_out)
  //std::cout<<u<<" \n";

if (dfs(i + 1)) return true;
used[j] = 0;
perm_out[i] = -1;
}


return false;
};

return dfs(0);
}

// ---------- The function you asked for ----------
// Make transpose output shape match q_ref shape by changing perm.






inline bool adaptTransposetoAttention(ONNX_NAMESPACE::Value* q_ref,
       ONNX_NAMESPACE::Node* k_transpose,
       ONNX_NAMESPACE::Value*& K ) {
if (!q_ref || !k_transpose) return false;
if (k_transpose->kind() != ONNX_NAMESPACE::kTranspose) return false;
if (k_transpose->inputs().size() != 1 || !k_transpose->inputs()[0]) return false;

// Shapes from your inference pipeline
auto ref_shape = q_ref->sizes();
auto kin_shape = k_transpose->inputs()[0]->sizes();

std::vector<int64_t> ref_shape_int;
std::vector<int64_t> kin_shape_int;

for(auto d : ref_shape)
  if (d.is_int)
        ref_shape_int.push_back(d.dim);
  else
        return false; //not supported
        
for(auto d : kin_shape)
  if (d.is_int)
        kin_shape_int.push_back(d.dim);
  else 
        return false; //not supported



if (ref_shape_int.empty() || kin_shape_int.empty()) return false;
if (ref_shape_int.size() != kin_shape_int.size()) return false;

std::vector<int64_t> perm;
if (!findPermToMatchShape(kin_shape_int, ref_shape_int, perm)) {
return false; // cannot match by permutation alone (needs reshape/broadcast)
}



const int64_t rank = static_cast<int64_t>(perm.size());
if (rank < 2) return false;

// ---- CHECK if permutation is only the last 2 elements swapped
const bool is_last2_swap = perm[rank - 1] == rank - 2 && perm[rank - 2] == rank - 1;

if (!is_last2_swap) {
  // Already correct for Attention: do nothing
  // K should be the input to transpose (pre-transpose)
  setPermAttr(k_transpose, perm);
  K = k_transpose->outputs()[0];
  

}


return true;
}


// ===================================================================
// Pass: FuseAttention
// ===================================================================

struct FuseAttention final : public PredicateBasedPass {
  explicit FuseAttention()
      : PredicateBasedPass(PassType::Fuse, PassEfficiency::Complete, PassOptimizationType::Compute) {}

  std::string getPassName() const override { return "fuse_attention"; }

  bool patternMatchPredicate(ONNX_NAMESPACE::Node* n) override {
    if (!n || n->kind() != ONNX_NAMESPACE::kMatMul) return false;
    if (n->inputs().size() != 2) return false;
    if (n->outputs().size() != 1) return false;

    auto* probs = n->inputs()[0];
    if (!probs || !probs->node() || probs->node()->kind() != ONNX_NAMESPACE::kSoftmax) return false;
    if (probs->node()->inputs().size() != 1) return false;

    return true;
  }

  bool runTransform(ONNX_NAMESPACE::Node* out_mm,
                    ONNX_NAMESPACE::Graph& graph,
                    NodeDestroyType& destroy_current) override {
    destroy_current = NodeDestroyType::DestroyZero;

    // out = MatMul(probs, V)
    ONNX_NAMESPACE::Value* probs_v = out_mm->inputs()[0];
    ONNX_NAMESPACE::Value* V       = out_mm->inputs()[1];

    //std::cout<<"probs_v "<<probs_v->uniqueName()<<std::endl;

    if (!probs_v || !V) return false;

    ONNX_NAMESPACE::Node* softmax = probs_v->node();
    if (!softmax || softmax->kind() != ONNX_NAMESPACE::kSoftmax) return false;
    if (softmax->inputs().size() != 1) return false;
    
    //std::cout<<"softmax "<<std::endl;

    ONNX_NAMESPACE::Value* scores_v = softmax->inputs()[0];
    if (!scores_v) return false;

    // Optional mask: Add(scores, attn_mask)
    ONNX_NAMESPACE::Value* attn_mask = nullptr;
    if (scores_v->node() && scores_v->node()->kind() == ONNX_NAMESPACE::kAdd) {
      auto mi = tryExtractAddMask(scores_v->node());
      if (mi) {
        scores_v  = mi->scores_no_mask;
        attn_mask = mi->attn_mask;
      }
    }

    // Optional post-scale on scores
    std::optional<float> scale_attr;
    {
      ONNX_NAMESPACE::Value* maybe_scaled_scores = scores_v;
      auto post = tryExtractScaleFromScoresProducer(maybe_scaled_scores);
      if (post) {

        //std::cout<<"post "<<std::endl;
        scale_attr = post;
        scores_v = maybe_scaled_scores;
      }
    }

    // Require qk MatMul
    ONNX_NAMESPACE::Node* qk_mm = scores_v ? scores_v->node() : nullptr;

    //std::cout<<"qk_mm "<<std::endl;

    //std::cout<<"what node is this one "<<qk_mm->kind().toString()<<std::endl;

    if (!qk_mm || qk_mm->kind() != Symbol("MatMul")) return false;

    //std::cout<<"mm "<<std::endl;

    if (qk_mm->inputs().size() != 2) return false;

    //std::cout<<"mm sono 2 "<<std::endl;

    ONNX_NAMESPACE::Value* Q = nullptr;
    ONNX_NAMESPACE::Value* K = nullptr;

    // Optional pre-scale
    auto pre_scale = tryExtractPreScaleFromQKMatMul(qk_mm, Q, K);
    if (pre_scale && !scale_attr) scale_attr = pre_scale;

    // Validate K-side transpose
    //std::cout<<"Suppongo questa "<<std::endl;


    bool adaptable = adaptTransposetoAttention(qk_mm->inputs()[0],qk_mm->inputs()[1]->node(),K);

    if (!adaptable)
    {
      return false;
    }
    
    

    if (!qk_mm->inputs()[1] ||!qk_mm->inputs()[0]) return false;
    
    //std::cout<<"Suppongo questa "<<std::endl;


    
    if (!Q || !K) return false;

    //std::cout<<"K size transform "<<std::endl;

    // Create Attention node by name (not numeric Symbol)
    std::vector<ONNX_NAMESPACE::Value*> attn_inputs;
    attn_inputs.push_back(Q);
    attn_inputs.push_back(K);
    attn_inputs.push_back(V);
    if (attn_mask) attn_inputs.push_back(attn_mask);

    ONNX_NAMESPACE::Node* attn =
        graph.create(ONNX_NAMESPACE::Symbol("Attention"), attn_inputs, /*n_outputs=*/1);


    attn->insertBefore(out_mm);

    // Rewire SSA
    ONNX_NAMESPACE::Value* old_out = out_mm->output();
    ONNX_NAMESPACE::Value* new_out = attn->output();

    // Best-effort metadata transfer (only if your build supports it)
    tryCopyMetadata(new_out, old_out);

    if (old_out) {
      old_out->replaceAllUsesWith(new_out);
    }

    // Set optional "scale" attribute using the API available in your build
    if (scale_attr) {
      static const ONNX_NAMESPACE::Symbol kScale = ONNX_NAMESPACE::Symbol("scale");
      setFloatAttr(attn, kScale, *scale_attr);
    }


    const auto q_heads_dim = Q->sizes()[1];
    const auto kv_heads_dim = V->sizes()[1];

    const std::optional<int64_t> q_heads=extract_int_dimension(q_heads_dim);
    const std::optional<int64_t> kv_heads=extract_int_dimension(kv_heads_dim);


    if (!q_heads || !kv_heads)return false;
    

    const ONNX_NAMESPACE::Symbol kQNumHeads  = ONNX_NAMESPACE::Symbol("q_num_heads");
    const ONNX_NAMESPACE::Symbol kKVNumHeads = ONNX_NAMESPACE::Symbol("kv_num_heads");


    setIntAttr(attn, kQNumHeads,  *q_heads);
    setIntAttr(attn, kKVNumHeads, *kv_heads);


    // Destroy only the anchor node; DCE can clean up internal nodes if unused.
    destroy_current = NodeDestroyType::DestroyOne;
    return true;
  }
};

} // namespace optimization
} // namespace ONNX_NAMESPACE
