use std::collections::HashMap;

use metal::Buffer as MTLBuffer;

use super::{super::MTLContext, model_shape::ModelShape};
use crate::{DataType, array::array_size_in_bytes};

/// Holds one shared Metal buffer for every temporary tensor that can appear during a forward pass.
/// Each buffer is large enough for the worst-case shape: `max_suffix_len` / `max_prefix_len`.
///
/// Per-iteration code merely *wraps* the buffer into a `MetalArray` with the runtime shape.
#[derive(Debug)]
pub struct ForwardPassBuffers {
    // 1-D
    pub token_ids: MTLBuffer,
    pub token_positions: MTLBuffer,
    pub sampling_output: MTLBuffer,

    // 2-D
    pub attention_window_size_to_bias: HashMap<Option<usize>, MTLBuffer>,
    pub logits: MTLBuffer,
    pub main: MTLBuffer,
    pub shortcut: MTLBuffer,
    pub qkv: MTLBuffer,
    pub attention_output: MTLBuffer,
    pub mlp_fused_up: MTLBuffer,
    pub mlp_hidden: MTLBuffer,

    // 3-D
    pub rotated_queries: MTLBuffer,
    pub rotated_keys: MTLBuffer,

    // 2-pass attention intermediate buffers
    pub attention_partials: MTLBuffer, // [num_heads * max_suffix_len * total_blocks_count * head_dim]
    pub attention_sums: MTLBuffer, // [num_heads * max_suffix_len * total_blocks_count]
    pub attention_maxs: MTLBuffer, // [num_heads * max_suffix_len * total_blocks_count]
}

impl ForwardPassBuffers {
    // TODO: use device arrays instead of MTLBuffers
    /// Allocate the buffers with `StorageModeShared` so that they are CPU-accessible as well.
    pub fn new(
        context: &MTLContext,
        model_shape: &ModelShape,
        max_prefix_len: usize,
        max_suffix_len: usize,
    ) -> Self {
        // Helper closure for allocation
        let alloc = |shape: &[usize], dtype: DataType| -> MTLBuffer {
            let bytes = array_size_in_bytes(shape, dtype);
            context.device.new_buffer(
                bytes as u64,
                metal::MTLResourceOptions::StorageModeShared,
            )
        };

        let act_ty = model_shape.activation_data_type();

        let partials_shape =
            model_shape.attention_partials_shape(max_suffix_len);
        let sums_maxs_shape = model_shape.attention_sums_shape(max_suffix_len);

        Self {
            // 1-D
            token_ids: alloc(&[max_suffix_len], DataType::U64),
            token_positions: alloc(&[max_suffix_len], DataType::I32),
            sampling_output: alloc(&[max_suffix_len], DataType::U32),

            // 2-D
            attention_window_size_to_bias: (0..model_shape.num_layers)
                .map(|layer| {
                    (
                        model_shape.sliding_window_length_per_layer
                            [layer as usize],
                        alloc(
                            &[max_suffix_len, max_suffix_len + max_prefix_len],
                            act_ty,
                        ),
                    )
                })
                .collect::<HashMap<Option<usize>, MTLBuffer>>(),
            logits: alloc(&model_shape.logits_shape(max_suffix_len), act_ty),
            main: alloc(&model_shape.main_shape(max_suffix_len), act_ty),
            shortcut: alloc(&model_shape.main_shape(max_suffix_len), act_ty),
            qkv: alloc(&model_shape.qkv_shape(max_suffix_len), act_ty),
            attention_output: alloc(
                &model_shape.attention_output_shape(max_suffix_len),
                act_ty,
            ),
            mlp_fused_up: alloc(
                &model_shape.mlp_fused_up_shape(max_suffix_len),
                act_ty,
            ),
            mlp_hidden: alloc(
                &model_shape.mlp_hidden_shape(max_suffix_len),
                act_ty,
            ),

            // 3-D
            rotated_queries: alloc(
                &model_shape.rotated_queries_shape(max_suffix_len),
                act_ty,
            ),
            rotated_keys: alloc(
                &model_shape.rotated_keys_shape(max_suffix_len),
                act_ty,
            ),

            attention_partials: alloc(&partials_shape, act_ty),
            attention_sums: alloc(&sums_maxs_shape, act_ty),
            attention_maxs: alloc(&sums_maxs_shape, act_ty),
        }
    }
}
