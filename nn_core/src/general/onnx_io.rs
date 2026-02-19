use std::path::Path;
use std::sync::Arc;
use anyhow::{anyhow, Context};
use prost::Message;

use super::activation::ActivationType;
use super::layer::NNLayer;
use super::loss::LossType;
use super::network::Network;
use super::tensor::GpuTensor;

// ---------------------------------------------------------------------------
// Minimal ONNX protobuf types
// ---------------------------------------------------------------------------
//
// Field numbers are taken directly from onnx/onnx.proto (ONNX IR v9, opset 17).
// Only the fields we actually read or write are included; unknown fields are
// harmlessly ignored by prost on decode.

mod proto {
    use prost::Message;

    /// Corresponds to `onnx.TensorProto`.
    #[derive(Clone, PartialEq, Message)]
    pub struct TensorProto {
        /// Shape dimensions.
        #[prost(int64, repeated, tag = "1")]
        pub dims: Vec<i64>,

        /// Element type. 1 = FLOAT (f32).
        #[prost(int32, optional, tag = "2")]
        pub data_type: Option<i32>,

        /// Tensor name (matches the initializer name used in NodeProto inputs).
        #[prost(string, optional, tag = "8")]
        pub name: Option<String>,

        /// Raw little-endian bytes — preferred over `float_data` because it
        /// avoids protobuf packing ambiguities.
        #[prost(bytes = "vec", optional, tag = "9")]
        pub raw_data: Option<Vec<u8>>,
    }

    /// Attribute type discriminant used in `AttributeProto.type`.
    pub mod attr_type {
        pub const FLOAT: i32 = 1;
        pub const INT: i32 = 2;
    }

    /// Corresponds to `onnx.AttributeProto`.
    #[derive(Clone, PartialEq, Message)]
    pub struct AttributeProto {
        #[prost(string, optional, tag = "1")]
        pub name: Option<String>,

        /// Discriminates which value field is populated.
        #[prost(int32, optional, tag = "20")]
        pub r#type: Option<i32>,

        /// INT value (e.g. `transB`).
        #[prost(int64, optional, tag = "3")]
        pub i: Option<i64>,

        /// FLOAT value (e.g. LeakyRelu `alpha`).
        #[prost(float, optional, tag = "4")]
        pub f: Option<f32>,
    }

    /// Corresponds to `onnx.NodeProto`.
    #[derive(Clone, PartialEq, Message)]
    pub struct NodeProto {
        #[prost(string, repeated, tag = "1")]
        pub input: Vec<String>,

        #[prost(string, repeated, tag = "2")]
        pub output: Vec<String>,

        #[prost(string, optional, tag = "3")]
        pub name: Option<String>,

        #[prost(string, optional, tag = "4")]
        pub op_type: Option<String>,

        #[prost(message, repeated, tag = "5")]
        pub attribute: Vec<AttributeProto>,
    }

    /// Corresponds to `onnx.ValueInfoProto` (only the name is used here).
    #[derive(Clone, PartialEq, Message)]
    pub struct ValueInfoProto {
        #[prost(string, optional, tag = "1")]
        pub name: Option<String>,
    }

    /// Corresponds to `onnx.GraphProto`.
    #[derive(Clone, PartialEq, Message)]
    pub struct GraphProto {
        #[prost(message, repeated, tag = "1")]
        pub node: Vec<NodeProto>,

        #[prost(string, optional, tag = "2")]
        pub name: Option<String>,

        #[prost(message, repeated, tag = "5")]
        pub initializer: Vec<TensorProto>,

        #[prost(message, repeated, tag = "11")]
        pub input: Vec<ValueInfoProto>,

        #[prost(message, repeated, tag = "12")]
        pub output: Vec<ValueInfoProto>,
    }

    /// Corresponds to `onnx.OperatorSetIdProto`.
    #[derive(Clone, PartialEq, Message)]
    pub struct OperatorSetIdProto {
        #[prost(string, optional, tag = "1")]
        pub domain: Option<String>,

        #[prost(int64, optional, tag = "2")]
        pub version: Option<i64>,
    }

    /// Corresponds to `onnx.ModelProto`.
    #[derive(Clone, PartialEq, Message)]
    pub struct ModelProto {
        /// ONNX IR version. We write 9 (current as of ONNX 1.16).
        #[prost(int64, optional, tag = "1")]
        pub ir_version: Option<i64>,

        /// Opset declarations.
        #[prost(message, repeated, tag = "8")]
        pub opset_import: Vec<OperatorSetIdProto>,

        /// Computation graph.
        #[prost(message, optional, tag = "7")]
        pub graph: Option<GraphProto>,

        /// We embed the loss type here so `load_from_onnx` can reconstruct the
        /// Network exactly. Format: "loss=MSE" or "loss=CrossEntropy".
        #[prost(string, optional, tag = "6")]
        pub doc_string: Option<String>,
    }
}

// ---------------------------------------------------------------------------
// Helper: f32 slice → raw little-endian bytes and back
// ---------------------------------------------------------------------------

fn floats_to_bytes(floats: &[f32]) -> Vec<u8> {
    floats.iter().flat_map(|f| f.to_le_bytes()).collect()
}

fn bytes_to_floats(bytes: &[u8]) -> anyhow::Result<Vec<f32>> {
    if bytes.len() % 4 != 0 {
        return Err(anyhow!("Raw tensor bytes length {} is not a multiple of 4", bytes.len()));
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect())
}

// ---------------------------------------------------------------------------
// Activation ↔ ONNX op_type mapping
// ---------------------------------------------------------------------------

fn activation_to_op(act: ActivationType) -> &'static str {
    match act {
        ActivationType::ReLU      => "Relu",
        ActivationType::Sigmoid   => "Sigmoid",
        ActivationType::Tanh      => "Tanh",
        ActivationType::LeakyReLU(_) => "LeakyRelu",
        ActivationType::SoftMax   => "Softmax",
    }
}

fn op_to_activation(op: &str, attrs: &[proto::AttributeProto]) -> anyhow::Result<ActivationType> {
    match op {
        "Relu"      => Ok(ActivationType::ReLU),
        "Sigmoid"   => Ok(ActivationType::Sigmoid),
        "Tanh"      => Ok(ActivationType::Tanh),
        "Softmax"   => Ok(ActivationType::SoftMax),
        "LeakyRelu" => {
            let alpha = attrs
                .iter()
                .find(|a| a.name.as_deref() == Some("alpha"))
                .and_then(|a| a.f)
                .unwrap_or(0.01); // ONNX default
            Ok(ActivationType::LeakyReLU(alpha))
        }
        other => Err(anyhow!("Unsupported ONNX activation op_type: {other}")),
    }
}

fn loss_to_doc(loss: &LossType) -> &'static str {
    match loss {
        LossType::MSE          => "loss=MSE",
        LossType::CrossEntropy => "loss=CrossEntropy",
    }
}

fn doc_to_loss(doc: &str) -> anyhow::Result<LossType> {
    match doc {
        "loss=MSE"          => Ok(LossType::MSE),
        "loss=CrossEntropy" => Ok(LossType::CrossEntropy),
        other => Err(anyhow!("Unknown loss in ONNX doc_string: {other}")),
    }
}

// ---------------------------------------------------------------------------
// impl Network
// ---------------------------------------------------------------------------

impl Network {
    /// Serialize the trained network to an ONNX file at `path`.
    ///
    /// The resulting `.onnx` file is compatible with standard ONNX runtimes
    /// (e.g. ONNXRuntime, tract) for inference.  Note that because the network
    /// processes inputs as flat column-major vectors on the GPU, the ONNX graph
    /// uses `Gemm` with `transB=1` so external runtimes see the conventional
    /// (batch × input_size) → (batch × output_size) shape.
    pub fn save_to_onnx(&self, path: impl AsRef<Path>) -> anyhow::Result<()> {
        let mut nodes: Vec<proto::NodeProto> = Vec::new();
        let mut initializers: Vec<proto::TensorProto> = Vec::new();

        let n_layers = self.layers.len();

        // Name of the tensor feeding into the current layer.
        let mut prev_tensor = "input".to_string();

        for (i, layer) in self.layers.iter().enumerate() {
            let NNLayer::FullyConnectedLayer {
                size,
                prev_size,
                activation_type,
                weights,
                biases,
                ..
            } = layer;

            // ── Read weights and biases from the GPU ──────────────────────
            let w_data = weights.read_to_cpu(&self.device, &self.queue);
            let b_data = biases.read_to_cpu(&self.device, &self.queue);

            let w_name = format!("W_{i}");
            let b_name = format!("B_{i}");
            let gemm_out = format!("gemm_{i}");
            let is_last = i == n_layers - 1;
            let act_out = if is_last { "output".to_string() } else { format!("act_{i}") };

            // ── Initializer: weight tensor (output_size × input_size) ─────
            initializers.push(proto::TensorProto {
                dims: vec![*size as i64, *prev_size as i64],
                data_type: Some(1), // FLOAT
                name: Some(w_name.clone()),
                raw_data: Some(floats_to_bytes(&w_data)),
            });

            // ── Initializer: bias tensor (output_size,) ───────────────────
            initializers.push(proto::TensorProto {
                dims: vec![*size as i64],
                data_type: Some(1), // FLOAT
                name: Some(b_name.clone()),
                raw_data: Some(floats_to_bytes(&b_data)),
            });

            // ── Gemm node ─────────────────────────────────────────────────
            // Y = A * B^T + C  (transB=1 keeps weights in output×input order)
            nodes.push(proto::NodeProto {
                input: vec![prev_tensor.clone(), w_name, b_name],
                output: vec![gemm_out.clone()],
                name: Some(format!("Gemm_{i}")),
                op_type: Some("Gemm".to_string()),
                attribute: vec![
                    proto::AttributeProto {
                        name: Some("transB".to_string()),
                        r#type: Some(proto::attr_type::INT),
                        i: Some(1),
                        f: None,
                    },
                    proto::AttributeProto {
                        name: Some("alpha".to_string()),
                        r#type: Some(proto::attr_type::FLOAT),
                        f: Some(1.0),
                        i: None,
                    },
                    proto::AttributeProto {
                        name: Some("beta".to_string()),
                        r#type: Some(proto::attr_type::FLOAT),
                        f: Some(1.0),
                        i: None,
                    },
                ],
            });

            // ── Activation node ───────────────────────────────────────────
            let mut act_attrs = Vec::new();
            if let ActivationType::LeakyReLU(alpha) = activation_type {
                act_attrs.push(proto::AttributeProto {
                    name: Some("alpha".to_string()),
                    r#type: Some(proto::attr_type::FLOAT),
                    f: Some(*alpha),
                    i: None,
                });
            }
            if let ActivationType::SoftMax = activation_type {
                // axis = -1 (opset ≥ 13): apply softmax over last dimension.
                act_attrs.push(proto::AttributeProto {
                    name: Some("axis".to_string()),
                    r#type: Some(proto::attr_type::INT),
                    i: Some(-1),
                    f: None,
                });
            }

            nodes.push(proto::NodeProto {
                input: vec![gemm_out],
                output: vec![act_out.clone()],
                name: Some(format!("Act_{i}")),
                op_type: Some(activation_to_op(*activation_type).to_string()),
                attribute: act_attrs,
            });

            prev_tensor = act_out;
        }

        // ── Assemble graph ─────────────────────────────────────────────────
        let graph = proto::GraphProto {
            name: Some("neural_network".to_string()),
            node: nodes,
            initializer: initializers,
            input: vec![proto::ValueInfoProto { name: Some("input".to_string()) }],
            output: vec![proto::ValueInfoProto { name: Some("output".to_string()) }],
        };

        // ── Assemble model ─────────────────────────────────────────────────
        let model = proto::ModelProto {
            ir_version: Some(9),
            opset_import: vec![proto::OperatorSetIdProto {
                domain: Some(String::new()), // "" = default ONNX domain
                version: Some(17),
            }],
            graph: Some(graph),
            doc_string: Some(loss_to_doc(&self.loss_type).to_string()),
        };

        // ── Encode and write ───────────────────────────────────────────────
        let bytes = model.encode_to_vec();
        std::fs::write(path.as_ref(), bytes)
            .with_context(|| format!("Failed to write ONNX file: {}", path.as_ref().display()))?;

        Ok(())
    }

    /// Load a network that was previously saved with [`Network::save_to_onnx`].
    ///
    /// The `batch_size` argument is the batch size to use for training/inference
    /// after loading — it does not need to match the batch size used when the
    /// file was saved.
    ///
    /// The loss type and all layer activations are recovered from the file
    /// automatically.
    pub fn load_from_onnx(
        device_arc: Arc<wgpu::Device>,
        queue: wgpu::Queue,
        batch_size: u32,
        path: impl AsRef<Path>,
    ) -> anyhow::Result<Self> {
        // ── Read and decode ────────────────────────────────────────────────
        let bytes = std::fs::read(path.as_ref())
            .with_context(|| format!("Failed to read ONNX file: {}", path.as_ref().display()))?;

        let model = proto::ModelProto::decode(bytes.as_slice())
            .context("Failed to decode ONNX protobuf")?;

        let doc = model.doc_string.as_deref().unwrap_or("");
        let loss = doc_to_loss(doc)
            .with_context(|| format!("Could not determine loss type from doc_string: {doc:?}"))?;

        let graph = model.graph
            .ok_or_else(|| anyhow!("ONNX model has no graph"))?;

        // ── Index initializers by name ─────────────────────────────────────
        let initializers: std::collections::HashMap<&str, &proto::TensorProto> = graph
            .initializer
            .iter()
            .filter_map(|t| t.name.as_deref().map(|n| (n, t)))
            .collect();

        // ── Walk the node list ─────────────────────────────────────────────
        // We expect alternating pairs: Gemm → activation.
        // Collect (output_size, input_size, weights, biases, activation) per layer.

        struct LayerData {
            output_size: u32,
            input_size: u32,
            weights: Vec<f32>,
            biases: Vec<f32>,
            activation: ActivationType,
        }

        let mut layers_data: Vec<LayerData> = Vec::new();
        let mut node_iter = graph.node.iter().peekable();

        while let Some(gemm_node) = node_iter.next() {
            let op = gemm_node.op_type.as_deref().unwrap_or("");
            if op != "Gemm" {
                // Skip any leading non-Gemm nodes (e.g. if the graph contains
                // Reshape / Flatten nodes added by an external tool).
                continue;
            }

            // Gemm inputs: [A=prev_activation, B=weights, C=biases]
            let w_name = gemm_node.input.get(1)
                .ok_or_else(|| anyhow!("Gemm node missing weight input"))?
                .as_str();
            let b_name = gemm_node.input.get(2)
                .ok_or_else(|| anyhow!("Gemm node missing bias input"))?
                .as_str();

            let w_tensor = initializers.get(w_name)
                .ok_or_else(|| anyhow!("Initializer not found: {w_name}"))?;
            let b_tensor = initializers.get(b_name)
                .ok_or_else(|| anyhow!("Initializer not found: {b_name}"))?;

            let (output_size, input_size) = match w_tensor.dims.as_slice() {
                &[o, i] => (o as u32, i as u32),
                other => return Err(anyhow!("Unexpected weight dims: {other:?}")),
            };

            let w_bytes = w_tensor.raw_data.as_deref()
                .ok_or_else(|| anyhow!("Weight tensor {w_name} has no raw_data"))?;
            let b_bytes = b_tensor.raw_data.as_deref()
                .ok_or_else(|| anyhow!("Bias tensor {b_name} has no raw_data"))?;

            let weights = bytes_to_floats(w_bytes)
                .with_context(|| format!("Decoding weights {w_name}"))?;
            let biases = bytes_to_floats(b_bytes)
                .with_context(|| format!("Decoding biases {b_name}"))?;

            // The next node must be an activation.
            let act_node = node_iter.next()
                .ok_or_else(|| anyhow!("Expected activation node after Gemm, found end-of-graph"))?;
            let act_op = act_node.op_type.as_deref().unwrap_or("");
            let activation = op_to_activation(act_op, &act_node.attribute)?;

            layers_data.push(LayerData { output_size, input_size, weights, biases, activation });
        }

        if layers_data.is_empty() {
            return Err(anyhow!("No Gemm layers found in ONNX graph"));
        }

        // ── Rebuild topology and activation arrays ─────────────────────────
        // topology = [input_size, hidden_1_size, ..., output_size]
        let mut topology: Vec<u32> = Vec::with_capacity(layers_data.len() + 1);
        topology.push(layers_data[0].input_size);
        for ld in &layers_data {
            topology.push(ld.output_size);
        }

        let activations: Vec<ActivationType> = layers_data.iter()
            .map(|ld| ld.activation)
            .collect();

        // ── Create Network (weights will be randomised, then overwritten) ──
        let mut network = Network::new(
            device_arc,
            queue,
            batch_size,
            &topology,
            &activations,
            loss,
        )?;

        // ── Upload the saved weights to the GPU ────────────────────────────
        for (layer, ld) in network.layers.iter().zip(layers_data.iter()) {
            layer.upload_weights(&ld.weights, &ld.biases, &network.queue);
        }

        Ok(network)
    }
}