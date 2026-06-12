use crate::{array::Array, backends::common::Backend};

pub struct LayerActivationTrace<B: Backend> {
    pub inputs: Array<B>,
    pub pre_attention_norm: Array<B>,
    pub attention: Array<B>,
    pub post_attention_norm: Array<B>,
    pub mlp_inputs: Array<B>,
    pub pre_mlp_norm: Array<B>,
    pub mlp: Array<B>,
    pub post_mlp_norm: Array<B>,
    pub outputs: Array<B>,
}

pub struct ActivationTrace<B: Backend> {
    pub embedding_norm: Option<Array<B>>,
    pub layer_results: Box<[LayerActivationTrace<B>]>,
    pub output_norm: Array<B>,
    pub output_pooling: Option<Array<B>>,
    pub logits: Array<B>,
}

impl<B: Backend> ActivationTrace<B> {
    pub fn embedding_norm_mut(&mut self) -> &mut Array<B> {
        self.embedding_norm.as_mut().expect("embedding_norm is only available for classifier traces")
    }

    pub fn output_pooling_mut(&mut self) -> &mut Array<B> {
        self.output_pooling.as_mut().expect("output_pooling is only available for classifier traces")
    }
}

#[cfg(all(test, feature = "tracing"))]
#[path = "../../unit/forward_pass/traces/mod.rs"]
mod tests;
