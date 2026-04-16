use crate::{
    DataType,
    array::{Array, ArrayContextExt},
    backends::common::Backend,
    forward_pass::model_shape::ModelShape,
};

pub struct ScratchBuffers<B: Backend> {
    pub token_ids: Array<B>,
    pub logits: Array<B>,
}

impl<B: Backend> ScratchBuffers<B> {
    pub fn new(
        context: &B::Context,
        model_shape: &ModelShape,
        max_suffix_len: usize,
    ) -> Self {
        // Helper closure for allocation
        let alloc = |shape: &[usize], dtype: DataType, label: &str| -> Array<B> {
            context.create_array_uninitialized(shape, dtype, &format!("scratch_buffers_{label}"))
        };

        let act_ty = model_shape.activation_data_type();

        Self {
            token_ids: alloc(&[max_suffix_len], DataType::U64, "token_ids"),
            logits: alloc(&model_shape.logits_shape(max_suffix_len), act_ty, "logits"),
        }
    }
}
