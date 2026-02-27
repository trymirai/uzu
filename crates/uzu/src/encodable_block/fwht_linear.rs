//! Linear layer wrapped with optional pre/post FWHT transforms.

use crate::{
    backends::common::Backend,
    encodable_block::{EncodableBlock, EncodingParameters},
    forward_pass::state::ForwardPassState,
};

pub struct FwhtLinear<B: Backend> {
    pre_fwht: Option<Box<dyn EncodableBlock<B>>>,
    linear: Box<dyn EncodableBlock<B>>,
    post_fwht: Option<Box<dyn EncodableBlock<B>>>,
}

impl<B: Backend> FwhtLinear<B> {
    pub fn new(
        pre_fwht: Option<Box<dyn EncodableBlock<B>>>,
        linear: Box<dyn EncodableBlock<B>>,
        post_fwht: Option<Box<dyn EncodableBlock<B>>>,
    ) -> Self {
        Self {
            pre_fwht,
            linear,
            post_fwht,
        }
    }
}

impl<B: Backend> EncodableBlock<B> for FwhtLinear<B> {
    fn supports_shared_encoder(&self) -> bool {
        self.pre_fwht.as_ref().map_or(true, |b| b.supports_shared_encoder())
            && self.linear.supports_shared_encoder()
            && self.post_fwht.as_ref().map_or(true, |b| b.supports_shared_encoder())
    }

    fn encode_with_shared_encoder(
        &self,
        state: &mut ForwardPassState<B>,
        parameters: &EncodingParameters<B>,
        encoder: &mut B::ComputeEncoder,
    ) {
        if let Some(pre) = &self.pre_fwht {
            pre.encode_with_shared_encoder(state, parameters, encoder);
        }
        self.linear.encode_with_shared_encoder(state, parameters, encoder);
        if let Some(post) = &self.post_fwht {
            post.encode_with_shared_encoder(state, parameters, encoder);
        }
    }
}
