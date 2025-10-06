use std::{cell::RefCell, rc::Rc};

use crate::{
    DeviceContext,
    backends::metal::{MTLContext, MetalArray, ModelShape},
};

type ArrayCell = RefCell<MetalArray>;

#[derive(Debug, Clone)]
pub struct MoeActivationTrace {
    pub suffix_length: usize,
    pub mixture_size: usize,
    pub k: usize,
    pub model_dim: usize,

    pub topk_ids: Vec<i32>,
    pub topk_probs: Vec<f32>,
    pub counts: Vec<u32>,
    pub offsets: Vec<u32>,
    pub sumk: Vec<u32>,
    pub bucketed_ids: Vec<u32>,
    pub bucketed_probs: Vec<f32>,
    pub tok2row: Vec<i32>,
    pub y_partial: Vec<f32>,
}

impl MoeActivationTrace {
    pub fn new(
        suffix_length: usize,
        mixture_size: usize,
        k: usize,
        model_dim: usize,
    ) -> Self {
        let routed_tokens = suffix_length * k;
        let y_partial_len = routed_tokens * model_dim;

        Self {
            suffix_length,
            mixture_size,
            k,
            model_dim,
            topk_ids: vec![0; routed_tokens],
            topk_probs: vec![0.0; routed_tokens],
            counts: vec![0; mixture_size],
            offsets: vec![0; mixture_size + 1],
            sumk: vec![0; 1],
            bucketed_ids: vec![0; routed_tokens],
            bucketed_probs: vec![0.0; routed_tokens],
            tok2row: vec![0; routed_tokens],
            y_partial: vec![0.0; y_partial_len],
        }
    }

    pub fn matches(
        &self,
        suffix_length: usize,
        mixture_size: usize,
        k: usize,
        model_dim: usize,
    ) -> bool {
        self.suffix_length == suffix_length
            && self.mixture_size == mixture_size
            && self.k == k
            && self.model_dim == model_dim
    }
}

pub struct DecoderLayerActivationTrace {
    pub inputs: ArrayCell,
    pub pre_attention_norm: ArrayCell,
    pub attention: ArrayCell,
    pub post_attention_norm: ArrayCell,
    pub mlp_inputs: ArrayCell,
    pub pre_mlp_norm: ArrayCell,
    pub mlp: ArrayCell,
    pub post_mlp_norm: ArrayCell,
    pub outputs: ArrayCell,
    pub moe: Option<MoeActivationTrace>,
}

impl DecoderLayerActivationTrace {
    pub fn new(
        context: &MTLContext,
        model_shape: &ModelShape,
        suffix_length: usize,
    ) -> Self {
        unsafe {
            Self {
                inputs: RefCell::new(context.array_uninitialized(
                    &model_shape.main_shape(suffix_length),
                    model_shape.activation_data_type(),
                )),
                pre_attention_norm: RefCell::new(context.array_uninitialized(
                    &model_shape.main_shape(suffix_length),
                    model_shape.activation_data_type(),
                )),
                attention: RefCell::new(context.array_uninitialized(
                    &model_shape.main_shape(suffix_length),
                    model_shape.activation_data_type(),
                )),
                post_attention_norm: RefCell::new(context.array_uninitialized(
                    &model_shape.main_shape(suffix_length),
                    model_shape.activation_data_type(),
                )),
                mlp_inputs: RefCell::new(context.array_uninitialized(
                    &model_shape.main_shape(suffix_length),
                    model_shape.activation_data_type(),
                )),
                pre_mlp_norm: RefCell::new(context.array_uninitialized(
                    &model_shape.main_shape(suffix_length),
                    model_shape.activation_data_type(),
                )),
                mlp: RefCell::new(context.array_uninitialized(
                    &model_shape.main_shape(suffix_length),
                    model_shape.activation_data_type(),
                )),
                post_mlp_norm: RefCell::new(context.array_uninitialized(
                    &model_shape.main_shape(suffix_length),
                    model_shape.activation_data_type(),
                )),
                outputs: RefCell::new(context.array_uninitialized(
                    &model_shape.main_shape(suffix_length),
                    model_shape.activation_data_type(),
                )),
                moe: None,
            }
        }
    }

    pub fn ensure_moe_trace(
        &mut self,
        suffix_length: usize,
        mixture_size: usize,
        k: usize,
        model_dim: usize,
    ) -> &mut MoeActivationTrace {
        let needs_init = self
            .moe
            .as_ref()
            .map(|trace| {
                !trace.matches(suffix_length, mixture_size, k, model_dim)
            })
            .unwrap_or(true);

        if needs_init {
            self.moe = Some(MoeActivationTrace::new(
                suffix_length,
                mixture_size,
                k,
                model_dim,
            ));
        }

        self.moe.as_mut().unwrap()
    }
}

pub struct DecoderActivationTrace {
    pub layer_results: Vec<Rc<RefCell<DecoderLayerActivationTrace>>>,
    pub output_norm: ArrayCell,
    pub logits: ArrayCell,
}

impl DecoderActivationTrace {
    pub fn new(
        context: &MTLContext,
        model_shape: &ModelShape,
        suffix_length: usize,
    ) -> Self {
        let layer_results = (0..model_shape.num_layers)
            .map(|_| {
                Rc::new(RefCell::new(DecoderLayerActivationTrace::new(
                    context,
                    model_shape,
                    suffix_length,
                )))
            })
            .collect();
        unsafe {
            Self {
                layer_results,
                output_norm: RefCell::new(context.array_uninitialized(
                    &model_shape.main_shape(suffix_length),
                    model_shape.activation_data_type(),
                )),
                logits: RefCell::new(context.array_uninitialized(
                    &model_shape.logits_shape(suffix_length),
                    model_shape.activation_data_type(),
                )),
            }
        }
    }
}
