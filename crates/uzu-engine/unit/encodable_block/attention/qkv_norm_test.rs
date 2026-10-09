use uzu_engine_macros::uzu_test;

use super::QKVNorm;
use crate::{
    backends::common::{Backend, CommandBufferEncoding, CommandBufferExecutable, CommandBufferPending, Context},
    config::normalization::{NormalizationConfig, UpcastMode},
    data_type::DataType,
    tests::{
        assert::assert_eq_float,
        helpers::{buffer_to_vec, create_buffer_with_data, create_context, for_each_backend},
    },
};

fn run_key_value_row_stride_test<B: Backend>() {
    const BATCH_SIZE: u32 = 2;
    const NUM_KV_HEADS: u32 = 2;
    const HEAD_DIM: u32 = 4;
    const ROW_WIDTH: u32 = 2 * NUM_KV_HEADS * HEAD_DIM;
    const EPSILON: f32 = 1e-6;

    let context = create_context::<B>();
    let config = NormalizationConfig {
        epsilon: EPSILON,
        scale_offset: None,
        upcast_mode: UpcastMode::OnlyNormalization,
        subtract_mean: false,
        has_scale: false,
        has_biases: false,
    };
    let key = QKVNorm::<B>::build_head(&context, DataType::F32, config, None, HEAD_DIM)
        .expect("failed to construct key norm");
    let norm = QKVNorm {
        query: None,
        key: Some(key),
        value: None,
        num_q_heads: 0,
        num_kv_heads: NUM_KV_HEADS,
        projection_row_stride: ROW_WIDTH,
        head_dim: HEAD_DIM,
    };

    let row_width = ROW_WIDTH as usize;
    let head_dim = HEAD_DIM as usize;
    let input = (0..BATCH_SIZE as usize * row_width).map(|index| 1.0 + index as f32 * 0.125).collect::<Vec<_>>();
    let mut expected = input.clone();
    for batch in 0..BATCH_SIZE as usize {
        for head in 0..NUM_KV_HEADS as usize {
            let start = batch * row_width + head * head_dim;
            let mean_square =
                input[start..start + head_dim].iter().map(|value| value * value).sum::<f32>() / HEAD_DIM as f32;
            let inverse_rms = (mean_square + EPSILON).sqrt().recip();
            for index in start..start + head_dim {
                expected[index] *= inverse_rms;
            }
        }
    }

    let mut key_value = create_buffer_with_data::<B, f32>(&context, &input);
    let mut command_buffer = context.create_command_buffer(None, None).expect("failed to create command buffer");
    norm.encode_key_value(&mut key_value, BATCH_SIZE, &mut command_buffer).expect("failed to encode key/value norm");
    command_buffer.end_encoding().submit().wait_until_completed().expect("failed to execute key/value norm");

    let output = buffer_to_vec::<B, f32>(&key_value);
    assert_eq_float(&expected, &output, 1e-5, "key/value norm row stride mismatch");
}

#[uzu_test]
fn test_key_value_norm_uses_element_row_stride() {
    for_each_backend!(|B| run_key_value_row_stride_test::<B>());
}
