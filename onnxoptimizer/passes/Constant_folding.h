/* SPDX-License-Identifier: Apache-2.0 */
// EXPERIMENTAL

#pragma once

#include <string>
#include <vector>

#include "onnx/onnx_pb.h"
#include "onnxoptimizer/pass.h"
#include "onnxoptimizer/passes/pass_util.h"
#include "onnxoptimizer/passes/tensor_util.h"
#include <onnx/defs/schema.h>
#include <onnx/common/interned_strings.h>

#include "onnxruntime_cxx_api.h"

namespace ONNX_NAMESPACE {


    


namespace optimization {

  static onnx::TensorProto_DataType OrtToOnnxDType(ONNXTensorElementDataType t) {
    switch (t) {
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:   return onnx::TensorProto_DataType_FLOAT;
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:  return onnx::TensorProto_DataType_DOUBLE;
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:   return onnx::TensorProto_DataType_INT64;
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:   return onnx::TensorProto_DataType_INT32;
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:   return onnx::TensorProto_DataType_INT16;
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:    return onnx::TensorProto_DataType_INT8;
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:  return onnx::TensorProto_DataType_UINT64;
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:  return onnx::TensorProto_DataType_UINT32;
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:  return onnx::TensorProto_DataType_UINT16;
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:   return onnx::TensorProto_DataType_UINT8;
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:    return onnx::TensorProto_DataType_BOOL;
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16: return onnx::TensorProto_DataType_FLOAT16;
      default: throw std::runtime_error("Unsupported ORT dtype");
    }
}

static size_t OnnxDTypeSize(int dtype) {
    switch (dtype) {
      case onnx::TensorProto_DataType_BOOL:   return 1;
      case onnx::TensorProto_DataType_INT8:   return 1;
      case onnx::TensorProto_DataType_UINT8:  return 1;
      case onnx::TensorProto_DataType_INT16:  return 2;
      case onnx::TensorProto_DataType_UINT16: return 2;
      case onnx::TensorProto_DataType_INT32:  return 4;
      case onnx::TensorProto_DataType_UINT32: return 4;
      case onnx::TensorProto_DataType_INT64:  return 8;
      case onnx::TensorProto_DataType_UINT64: return 8;
      case onnx::TensorProto_DataType_FLOAT16:return 2;
      case onnx::TensorProto_DataType_FLOAT:  return 4;
      case onnx::TensorProto_DataType_DOUBLE: return 8;
      // add BFLOAT16 / COMPLEX64 / COMPLEX128 if you use them
      default: throw std::runtime_error("Unsupported TensorProto dtype");
    }
  }


 

struct OrtConstantFoldingSimple final : public FullGraphBasedPass {
  explicit OrtConstantFoldingSimple()
      : FullGraphBasedPass(PassType::Fuse,
                           PassEfficiency::Partial,
                           PassOptimizationType::Compute) {}

  std::string getPassName() const override {
    return "ort_constant_folding_simple";
  }

  PassAnalysisType getPassAnalysisType() const override {
    return PassAnalysisType::Empty;
  }

private:


  static onnx::Tensor OrtValueToOnnxTensor(const Ort::Value& v, const std::string& name) {

    if (!v.IsTensor()) throw std::runtime_error("Ort::Value is not a tensor");

    auto info = v.GetTensorTypeAndShapeInfo();
    const auto ort_et = static_cast<ONNXTensorElementDataType>(info.GetElementType());
    const auto onnx_dt = OrtToOnnxDType(ort_et);

    std::vector<int64_t> shape = info.GetShape();
    const size_t numel  = info.GetElementCount();
    const size_t nbytes = numel * OnnxDTypeSize(onnx_dt);

    onnx::Tensor t;

    if (onnx_dt == onnx::TensorProto_DataType_STRING) {
      throw std::runtime_error("STRING tensors require a different extraction path from ORT");
    }

    const void* src = v.GetTensorRawData(); // CPU-readable only

    // Allocate the raw backing store as a string and copy bytes into it.
    std::string raw;
    raw.resize(nbytes);
    std::memcpy(raw.data(), src, nbytes);

    // Store inside onnx::Tensor (deep copy / owned storage via string move)
    t.setName(name);
    t.elem_type()=static_cast<int>(onnx_dt);
    t.sizes()=shape;
    t.set_raw_data(std::move(raw));
    return t;

  }

  static onnx::TensorProto OnnxTensorToTensorProto(const onnx::Tensor& t,
                                      const std::string& name) {
  onnx::TensorProto tp;
  tp.set_name(name);

  const int dtype = static_cast<int>(t.elem_type());
  tp.set_data_type(dtype);

  if (dtype == onnx::TensorProto_DataType_STRING) {
    throw std::runtime_error("STRING tensors must be written via string_data, not raw_data");
  }

  for (auto d : t.sizes()) {
    tp.add_dims(static_cast<int64_t>(d));
  }

  // raw() already contains the byte payload as a std::string.
  // Copy it into the proto (or move if you can relinquish ownership).
  tp.set_raw_data(t.raw());   // copies

  return tp;
  }


  static bool isFoldCandidate(Node* n) {

    if (!n) return false;
    if (n->kind() == kConstant) return false;
    if (n->kind() == kIf || n->kind() == kLoop ) return false;
    if (n->outputs().size() != 1) return false;                 // keep it simple
    if (n->outputs()[0]->uses().empty()) return false;          // nothing to do
    if(n->inputs().size()==0) return false;
    for (Value* v : n->inputs()) {
      if (!IsConstantTensor(v)) return false;
    }

    return true;
  }

  static bool addInputAsInitializer(onnx::GraphProto* g, Value* v) {
    const Tensor* t = FetchConstantTensor(v);
    if (t) {
      onnx::TensorProto* tp = g->add_initializer();
      *tp = OnnxTensorToTensorProto(*t,v->uniqueName());
      return true;
    }
  
    if (v && v->node() && v->node()->kind() == kConstant) {
      Node* c = v->node();
      if (c->hasAttribute(kvalue)) {
        onnx::TensorProto* tp = g->add_initializer();
        *tp = OnnxTensorToTensorProto(c->t(kvalue),v->uniqueName());
        return true;
      }
    }
    return false;
  }
  
  static std::string unqual(const Symbol& s) {
    std::string str = s.toString();
    auto pos = str.rfind("::");
    if (pos != std::string::npos) str = str.substr(pos + 2);
    return str;
  }
  
  static bool emitNodeProto(Node* node, onnx::GraphProto* g) {
    auto* np = g->add_node();
    np->set_op_type(unqual(node->kind()));

    if (!node->domain().empty()) np->set_domain(node->domain());
  
    for (auto* in : node->inputs()) np->add_input(in->uniqueName());
    np->add_output(node->output()->uniqueName());

    
    for (onnx::Symbol a : node->attributeNames()) {
      // std::cout<<" the  symbol? "<<(uint32_t)(*it)<<std::endl;
      // ONNX_NAMESPACE::Symbol a = *it;
      auto* ap = np->add_attribute();
      ap->set_name(unqual(a));

      switch (node->kindOf(a)) {
        case AttributeKind::i:  ap->set_type(onnx::AttributeProto::INT); ap->set_i(node->i(a)); break;
        case AttributeKind::f:  ap->set_type(onnx::AttributeProto::FLOAT); ap->set_f(node->f(a)); break;
        case AttributeKind::s:  ap->set_type(onnx::AttributeProto::STRING); ap->set_s(node->s(a)); break;
        case AttributeKind::is: ap->set_type(onnx::AttributeProto::INTS); for (auto v : node->is(a)) ap->add_ints(v); break;
        case AttributeKind::fs: ap->set_type(onnx::AttributeProto::FLOATS); for (auto v : node->fs(a)) ap->add_floats(v); break;
        case AttributeKind::ss: ap->set_type(onnx::AttributeProto::STRINGS); for (const auto& v : node->ss(a)) ap->add_strings(v); break;
        case AttributeKind::t:  ap->set_type(onnx::AttributeProto::TENSOR); *ap->mutable_t() = OnnxTensorToTensorProto(node->t(a),node->t(a).name()); break;
        case AttributeKind::ts: ap->set_type(onnx::AttributeProto::TENSORS); for (const Tensor& tt : node->ts(a)) *ap->add_tensors() = OnnxTensorToTensorProto(tt,tt.name()); break;
        default: return false;
      }
    }
    return true;
  }

  static bool cloneNodeToOneNodeModel(Node* node,
                                      Graph& parent_graph,
                                      onnx::ModelProto& out_model,
                                      std::string& out_output_name) {
    out_model.Clear();

    out_model.set_ir_version(11);
    auto* opset = out_model.add_opset_import();
    opset->set_domain("");

    // 

    int latest = 23;

    opset->set_version(latest); // conservative: match parent
    

    onnx::GraphProto* g = out_model.mutable_graph();
    g->set_name("ort_cf_one_node");


    // Add initializers for all inputs (they are constant by construction).
    for (Value* in : node->inputs()) {
        if (!addInputAsInitializer(g, in)) return false;
    }

    if(!emitNodeProto(node, g)) return false;
    out_output_name = node->output()->uniqueName();

    // Mark node output as graph output
    onnx::ValueInfoProto* out_vi = g->add_output();
    out_vi->set_name(out_output_name);

    return true;
  }

  static bool ortRunOneNode(const onnx::ModelProto& model,
                            const std::string& out_name,
                            onnx::Tensor& out_tensor) {
    static Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "onnxoptimizer_cf");
    Ort::SessionOptions so;
    so.SetIntraOpNumThreads(1);
    so.SetInterOpNumThreads(1);
    so.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_DISABLE_ALL);

    std::string bytes;
    if (!model.SerializeToString(&bytes))
      return false;

    Ort::Session session(env, bytes.data(), bytes.size(), so);

    // No external inputs (everything is initializer)
    const char* output_names[] = { out_name.c_str() };
    auto outputs = session.Run(Ort::RunOptions{},
                               nullptr, nullptr, 0,
                               output_names, 1);

    if (outputs.size() != 1 || !outputs[0].IsTensor())
      return false;

    Ort::Value& v = outputs[0];
    
    out_tensor = OrtValueToOnnxTensor(v,out_name);

    return true;

  }

  static bool foldOne(Node* node, Graph& graph) {
    onnx::ModelProto one;
    std::string out_name;



    if (!cloneNodeToOneNodeModel(node, graph, one, out_name))
      return false;

    


    // SSA-safe replacement:
    Value* old_out = node->output();
    const std::string old_name = old_out->uniqueName();
    // Rename old output to a fresh temp name before creating initializer with old_name.
    old_out->setUniqueName(ONNX_NAMESPACE::to_string(graph.getNextUnique()), false);

    onnx::Tensor folded;
    if (!ortRunOneNode(one, old_name, folded))
      return false;
    
    Value* init_v = graph.addInitializerAndCreateValue(folded);
    init_v->setUniqueName(old_name, false);

    // Replace all uses of the old output (now temp-named) with initializer
    if (!tryReplacingAllUsesWith(old_out, init_v))
      return false;

    node->destroy();
    return true;
  }

  std::shared_ptr<PostPassAnalysis> runPass(Graph& graph) override {
    // Fixed-point: fold until nothing changes.
    for (;;) {
      bool changed = false;

      for (auto it = graph.begin(); it != graph.end(); ) {
        Node* n = *it;
        
        ++it;

        if (isFoldCandidate(n)) {


          if (foldOne(n, graph)) {
            changed = true;
          }
        }
      }

      if (!changed) break;
    }

    return std::make_shared<PostPassAnalysis>();
  }
};

} // namespace optimization
} // namespace ONNX_NAMESPACE
