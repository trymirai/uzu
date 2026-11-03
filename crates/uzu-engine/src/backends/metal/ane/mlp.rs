//! A dense MLP whose prefill batches run on the Neural Engine.

use std::sync::Arc;

use crate::{
    backends::{
        common::{Backend, CommandBuffer, gpu_types::activation_type::ActivationType},
        metal::{
            Metal,
            ane::ffn::{AneFfn, AneLayer},
            context::MetalContext,
        },
    },
    config::{clipping::ClippingBounds, mlp::dense_mlp::DenseMLPConfig},
    encodable_block::mlp::{DenseMlp, Mlp, MlpEncodeError},
};

/// Runs prefill-sized batches on the ANE and everything else (decode, small batches, and every
/// batch once the ANE has failed) on the wrapped GPU MLP.
pub struct AneMlp {
    dense: DenseMlp<Metal>,
    ffn: Arc<AneFfn>,
    layer: AneLayer,
}

impl AneMlp {
    /// Wraps `dense` if the ANE can run it; otherwise returns it as is.
    pub fn wrap(
        context: &MetalContext,
        dense: DenseMlp<Metal>,
        config: &DenseMLPConfig,
    ) -> Box<dyn Mlp<Metal>> {
        let plain_silu = config.activation.act_type() == ActivationType::SILU
            && config.activation.custom_alpha().is_none()
            && config.gate_clipping == ClippingBounds::default()
            && config.up_clipping == ClippingBounds::default();
        if !plain_silu {
            return Box::new(dense);
        }
        let Some(weights) = dense.weights() else {
            return Box::new(dense);
        };
        let Some(ffn) = context.ane_ffn(&weights) else {
            return Box::new(dense);
        };
        let layer = match ffn.layer(context, &weights) {
            Ok(Some(layer)) => layer,
            Ok(None) => return Box::new(dense),
            Err(error) => {
                eprintln!("uzu: ANE layer setup failed, MLP stays on the GPU: {error}");
                return Box::new(dense);
            },
        };
        drop(weights);
        Box::new(Self {
            dense,
            ffn,
            layer,
        })
    }
}

impl Mlp<Metal> for AneMlp {
    fn encode(
        &self,
        input: <Metal as Backend>::ScratchBuffer,
        batch_dim: u32,
        command_buffer: &mut <<Metal as Backend>::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<<Metal as Backend>::ScratchBuffer, MlpEncodeError<Metal>> {
        if !self.ffn.accepts(batch_dim) {
            return self.dense.encode(input, batch_dim, command_buffer);
        }
        let weights = self.dense.weights().expect("weights were exposed when the MLP was wrapped");
        self.ffn.encode(&self.layer, &weights, &input, batch_dim, command_buffer).map_err(MlpEncodeError::Backend)
    }
}
