use std::{path::PathBuf, time::Duration};

use proc_macros::uzu_test;

use super::{
    blocks::{
        activation, attention_single_pass, attention_update_kv_cache, full_precision_embedding, gated_act_mul,
        hadamard_transform, layer_norm, logit_soft_cap, matmul, qkv_norm, repetition_penalty, rms_norm, rope,
        sigmoid_gate, softmax, tensor_add_bias, tensor_add_scale, tensor_copy, unified_sampling,
    },
    model_shapes,
};
use crate::{backends::metal::Metal, data_type::DataType, tests::helpers::create_context};

const MEASUREMENT_WINDOW: Duration = Duration::from_secs(2);

#[uzu_test]
#[ignore = "GPU profiling; run via `cargo profile`"]
fn profile_blocks() {
    let context = create_context::<Metal>();
    let output_directory = output_directory();

    matmul::profile_parameters(&context, &matmul_parameters(), &output_directory, MEASUREMENT_WINDOW);

    rms_norm::profile_and_write(&context, &output_directory, MEASUREMENT_WINDOW);
    layer_norm::profile_and_write(&context, &output_directory, MEASUREMENT_WINDOW);
    qkv_norm::profile_and_write(&context, &output_directory, MEASUREMENT_WINDOW);
    activation::profile_and_write(&context, &output_directory, MEASUREMENT_WINDOW);
    gated_act_mul::profile_and_write(&context, &output_directory, MEASUREMENT_WINDOW);
    sigmoid_gate::profile_and_write(&context, &output_directory, MEASUREMENT_WINDOW);
    softmax::profile_and_write(&context, &output_directory, MEASUREMENT_WINDOW);
    logit_soft_cap::profile_and_write(&context, &output_directory, MEASUREMENT_WINDOW);
    hadamard_transform::profile_and_write(&context, &output_directory, MEASUREMENT_WINDOW);
    tensor_add_bias::profile_and_write(&context, &output_directory, MEASUREMENT_WINDOW);
    tensor_add_scale::profile_and_write(&context, &output_directory, MEASUREMENT_WINDOW);
    tensor_copy::profile_and_write(&context, &output_directory, MEASUREMENT_WINDOW);
    rope::profile_and_write(&context, &output_directory, MEASUREMENT_WINDOW);
    attention_single_pass::profile_and_write(&context, &output_directory, MEASUREMENT_WINDOW);
    attention_update_kv_cache::profile_and_write(&context, &output_directory, MEASUREMENT_WINDOW);
    full_precision_embedding::profile_and_write(&context, &output_directory, MEASUREMENT_WINDOW);
    unified_sampling::profile_and_write(&context, &output_directory, MEASUREMENT_WINDOW);
    repetition_penalty::profile_and_write(&context, &output_directory, MEASUREMENT_WINDOW);
}

fn matmul_parameters() -> Vec<matmul::Parameters> {
    let (head_input_dimension, head_output_dimension) = model_shapes::language_model_head_shape();
    let mut parameters = Vec::new();
    for data_type in [DataType::F32, DataType::BF16] {
        for tokens in model_shapes::REAL_TOKEN_COUNTS {
            for (input_dimension, output_dimension) in model_shapes::matmul_projection_shapes() {
                parameters.push(matmul::Parameters {
                    tokens,
                    input_dimension,
                    output_dimension,
                    data_type,
                });
            }
        }
        parameters.push(matmul::Parameters {
            tokens: model_shapes::DECODE_TOKEN_COUNT,
            input_dimension: head_input_dimension,
            output_dimension: head_output_dimension,
            data_type,
        });
    }
    parameters
}

fn output_directory() -> PathBuf {
    std::env::var_os("UZU_PROFILE_OUTPUT_DIRECTORY").map(PathBuf::from).unwrap_or_else(std::env::temp_dir)
}
