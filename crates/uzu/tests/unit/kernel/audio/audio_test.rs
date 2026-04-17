#![cfg(metal_backend)]

use uzu::{
    ArrayContextExt, DataType, allocation_as_slice,
    backends::{
        common::{
            Backend, Context, Encoder, Kernels,
            gpu_types::ActivationType,
            kernel::{
                ActivationKernel, AudioAddKernel, AudioCausalConv1dKernel, AudioCausalConvTranspose1dCausalPadKernel,
                AudioCausalConvTranspose1dKernel, AudioConv1dKernel, AudioFsqDecodeKernel, AudioFsqEncodeKernel,
                AudioHalfSnakeKernel, AudioNormNcsKernel,
            },
        },
        metal::Metal,
    },
};

use super::super::common::audio::{
    fsq_reference::{fsq_decode_reference, fsq_encode_reference},
    ops_reference::{
        CausalConv1dSpec, CausalConvTranspose1dSpec, Conv1dSpec, HalfSnakeSpec, PadMode,
        causal_conv_transpose1d_causal_pad_reference, causal_conv_transpose1d_reference, causal_conv1d_reference,
        conv1d_reference, half_snake_reference,
    },
};
use crate::uzu_test;

macro_rules! borrow_array_buffer {
    ($name:ident = $array:expr) => {
        let $name = $array.allocation();
    };
}

macro_rules! borrow_array_buffer_mut {
    ($name:ident = $array:expr) => {
        let $name = &mut $array;
    };
}

fn create_test_context() -> std::rc::Rc<<Metal as Backend>::Context> {
    <Metal as Backend>::Context::new().expect("MetalContext")
}

fn run_command_buffer(
    context: &std::rc::Rc<<Metal as Backend>::Context>,
    encode: impl FnOnce(&mut Encoder<'_, Metal>),
) {
    let mut encoder = Encoder::<Metal>::new(context.as_ref()).expect("command buffer");
    encode(&mut encoder);
    encoder.end_encoding().submit().wait_until_completed().expect("command buffer completed");
}

#[uzu_test]
fn audio_conv1d_replicate_matches_reference_f32() {
    let context = create_test_context();
    let kernel = <<Metal as Backend>::Kernels as Kernels>::AudioConv1dKernel::new(&context, DataType::F32)
        .expect("audio runtime");

    let batch_size = 2usize;
    let cin = 3usize;
    let cout = 4usize;
    let seq_len_in = 9usize;

    let stride = 3usize;
    let kernel_size = 2 * stride;
    let dilation = 1usize;
    let padding = 2usize;

    let seq_len_out = ((seq_len_in + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1;
    let lengths_out: [i32; 2] = [2, seq_len_out as i32];

    let input_len = batch_size * cin * seq_len_in;
    let weight_len = cout * cin * kernel_size;
    let bias_len = cout;
    let output_len = batch_size * cout * seq_len_out;

    let input_values: Vec<f32> = (0..input_len).map(|i| (i as f32 * 0.01).sin() * 0.5).collect();
    let weight_values: Vec<f32> = (0..weight_len).map(|i| ((i as f32) * 0.07 - 1.3).cos() * 0.2).collect();
    let bias_values: Vec<f32> = (0..bias_len).map(|i| i as f32 * 0.01 - 0.02).collect();

    let mut input = context.create_array_zeros(&[batch_size, cin, seq_len_in], DataType::F32, "audio_conv1d_input");
    input.as_slice_mut::<f32>().copy_from_slice(&input_values);

    let mut weight = context.create_array_zeros(&[cout, cin, kernel_size], DataType::F32, "audio_conv1d_weight");
    weight.as_slice_mut::<f32>().copy_from_slice(&weight_values);

    let mut bias = context.create_array_zeros(&[cout], DataType::F32, "audio_conv1d_bias");
    bias.as_slice_mut::<f32>().copy_from_slice(&bias_values);

    let mut lengths = context.create_array_zeros(&[batch_size], DataType::I32, "audio_conv1d_lengths");
    lengths.as_slice_mut::<i32>().copy_from_slice(&lengths_out);

    let mut output =
        context.create_array_zeros(&[batch_size, cout, seq_len_out], DataType::F32, "audio_conv1d_output").into_allocation();
    let expected = conv1d_reference(Conv1dSpec {
        input: &input_values,
        weight: &weight_values,
        bias: &bias_values,
        lengths: &lengths_out,
        batch_size,
        cin,
        cout,
        seq_len_in,
        seq_len_out,
        kernel_size,
        stride,
        dilation,
        padding,
        pad_mode: PadMode::Replicate,
    })
    .expect("reference conv1d");

    run_command_buffer(&context, |command_buffer| {
        borrow_array_buffer!(input_buffer = input);
        borrow_array_buffer!(weight_buffer = weight);
        borrow_array_buffer!(bias_buffer = bias);
        borrow_array_buffer_mut!(output_buffer = output);
        borrow_array_buffer!(lengths_buffer = lengths);
        kernel.encode(
            input_buffer,
            weight_buffer,
            bias_buffer,
            output_buffer,
            lengths_buffer,
            cin as i32,
            cout as i32,
            seq_len_in as i32,
            seq_len_out as i32,
            kernel_size as i32,
            stride as i32,
            dilation as i32,
            padding as i32,
            1_i32,
            batch_size as i32,
            command_buffer,
        );
    });

    let got = allocation_as_slice::<f32, Metal>(&output);
    for index in 0..output_len {
        let delta = (expected[index] - got[index]).abs();
        assert!(
            delta <= 1e-4,
            "Mismatch at index={index}: expected {}, got {}, delta={delta}",
            expected[index],
            got[index]
        );
    }
}

#[uzu_test]
fn audio_causal_conv1d_matches_reference_f32() {
    let context = create_test_context();
    let kernel = <<Metal as Backend>::Kernels as Kernels>::AudioCausalConv1dKernel::new(&context, DataType::F32)
        .expect("audio runtime");

    let batch_size = 2usize;
    let cin = 3usize;
    let cout = 4usize;
    let seq_len = 8usize;
    let kernel_size = 5usize;
    let dilation = 2usize;

    let lengths: [i32; 2] = [5, 8];

    let input_len = batch_size * cin * seq_len;
    let weight_len = cout * cin * kernel_size;
    let bias_len = cout;
    let output_len = batch_size * cout * seq_len;

    let input_values: Vec<f32> = (0..input_len).map(|i| (i as f32 * 0.01).sin()).collect();
    let weight_values: Vec<f32> = (0..weight_len).map(|i| ((i as f32) * 0.02 - 0.5).cos() * 0.1).collect();
    let bias_values: Vec<f32> = (0..bias_len).map(|i| i as f32 * 0.001 - 0.002).collect();

    let mut input = context.create_array_zeros(&[batch_size, cin, seq_len], DataType::F32, "audio_causal_conv1d_input");
    input.as_slice_mut::<f32>().copy_from_slice(&input_values);

    let mut weight = context.create_array_zeros(&[cout, cin, kernel_size], DataType::F32, "audio_causal_conv1d_weight");
    weight.as_slice_mut::<f32>().copy_from_slice(&weight_values);

    let mut bias = context.create_array_zeros(&[cout], DataType::F32, "audio_causal_conv1d_bias");
    bias.as_slice_mut::<f32>().copy_from_slice(&bias_values);

    let mut lengths_array = context.create_array_zeros(&[batch_size], DataType::I32, "audio_causal_conv1d_lengths");
    lengths_array.as_slice_mut::<i32>().copy_from_slice(&lengths);

    let mut output = context
        .create_array_zeros(&[batch_size, cout, seq_len], DataType::F32, "audio_causal_conv1d_output")
        .into_allocation();
    let expected = causal_conv1d_reference(CausalConv1dSpec {
        input: &input_values,
        weight: &weight_values,
        bias: &bias_values,
        lengths: &lengths,
        batch_size,
        cin,
        cout,
        seq_len,
        kernel_size,
        dilation,
    })
    .expect("reference causal conv1d");

    run_command_buffer(&context, |command_buffer| {
        borrow_array_buffer!(input_buffer = input);
        borrow_array_buffer!(weight_buffer = weight);
        borrow_array_buffer!(bias_buffer = bias);
        borrow_array_buffer_mut!(output_buffer = output);
        borrow_array_buffer!(lengths_buffer = lengths_array);
        kernel.encode(
            input_buffer,
            weight_buffer,
            bias_buffer,
            output_buffer,
            lengths_buffer,
            cin as i32,
            cout as i32,
            seq_len as i32,
            kernel_size as i32,
            dilation as i32,
            0,
            batch_size as i32,
            command_buffer,
        );
    });

    let got = allocation_as_slice::<f32, Metal>(&output);
    for index in 0..output_len {
        let delta = (expected[index] - got[index]).abs();
        assert!(
            delta <= 1e-4,
            "Mismatch at index={index}: expected {}, got {}, delta={delta}",
            expected[index],
            got[index]
        );
    }
}

#[uzu_test]
fn audio_causal_conv_transpose1d_matches_reference_f32() {
    let context = create_test_context();
    let kernel =
        <<Metal as Backend>::Kernels as Kernels>::AudioCausalConvTranspose1dKernel::new(&context, DataType::F32)
            .expect("audio runtime");

    let batch_size = 2usize;
    let cin = 4usize;
    let groups = 2usize;
    let cout = 2usize;
    let stride = 3usize;
    let kernel_size = 2 * stride;

    let seq_len_in = 5usize;
    let seq_len_out = seq_len_in * stride;

    let lengths_out: [i32; 2] = [9, 15];

    let input_len = batch_size * cin * seq_len_in;
    let weight_len = cout * (cin / groups) * kernel_size;
    let output_len = batch_size * cout * seq_len_out;

    let input_values: Vec<f32> = (0..input_len).map(|i| (i as f32 * 0.013).sin() * 0.5).collect();
    let weight_values: Vec<f32> = (0..weight_len).map(|i| ((i as f32) * 0.07).cos() * 0.2).collect();
    let bias_values: Vec<f32> = vec![0.01, -0.02];

    let mut input =
        context.create_array_zeros(&[batch_size, cin, seq_len_in], DataType::F32, "audio_causal_conv_transpose_input");
    input.as_slice_mut::<f32>().copy_from_slice(&input_values);

    let mut weight = context.create_array_zeros(
        &[cout, cin / groups, kernel_size],
        DataType::F32,
        "audio_causal_conv_transpose_weight",
    );
    weight.as_slice_mut::<f32>().copy_from_slice(&weight_values);

    let mut bias = context.create_array_zeros(&[cout], DataType::F32, "audio_causal_conv_transpose_bias");
    bias.as_slice_mut::<f32>().copy_from_slice(&bias_values);

    let mut lengths = context.create_array_zeros(&[batch_size], DataType::I32, "audio_causal_conv_transpose_lengths");
    lengths.as_slice_mut::<i32>().copy_from_slice(&lengths_out);

    let mut output = context
        .create_array_zeros(
        &[batch_size, cout, seq_len_out],
        DataType::F32,
        "audio_causal_conv_transpose_output",
    )
        .into_allocation();
    let expected = causal_conv_transpose1d_reference(CausalConvTranspose1dSpec {
        input: &input_values,
        weight: &weight_values,
        bias: &bias_values,
        lengths: &lengths_out,
        batch_size,
        cin,
        cout,
        seq_len_in,
        seq_len_out,
        stride,
        groups,
    })
    .expect("reference causal conv transpose1d");

    run_command_buffer(&context, |command_buffer| {
        borrow_array_buffer!(input_buffer = input);
        borrow_array_buffer!(weight_buffer = weight);
        borrow_array_buffer!(bias_buffer = bias);
        borrow_array_buffer_mut!(output_buffer = output);
        borrow_array_buffer!(lengths_buffer = lengths);
        kernel.encode(
            input_buffer,
            weight_buffer,
            bias_buffer,
            output_buffer,
            lengths_buffer,
            cin as i32,
            cout as i32,
            seq_len_in as i32,
            seq_len_out as i32,
            stride as i32,
            groups as i32,
            batch_size as i32,
            command_buffer,
        );
    });

    let got = allocation_as_slice::<f32, Metal>(&output);
    for index in 0..output_len {
        let delta = (expected[index] - got[index]).abs();
        assert!(
            delta <= 1e-4,
            "Mismatch at index={index}: expected {}, got {}, delta={delta}",
            expected[index],
            got[index]
        );
    }
}

#[uzu_test]
fn audio_causal_conv_transpose1d_causal_pad_matches_reference_f32() {
    let context = create_test_context();
    let kernel = <<Metal as Backend>::Kernels as Kernels>::AudioCausalConvTranspose1dCausalPadKernel::new(
        &context,
        DataType::F32,
    )
    .expect("audio runtime");

    let batch_size = 1usize;
    let cin = 2usize;
    let groups = 1usize;
    let cout = 2usize;
    let stride = 2usize;
    let kernel_size = 3usize;

    let seq_len_in = 5usize;
    let seq_len_out = 10usize;
    let lengths_out: [i32; 1] = [10];

    let input_len = batch_size * cin * seq_len_in;
    let weight_len = cout * (cin / groups) * kernel_size;
    let output_len = batch_size * cout * seq_len_out;

    let input_values_ncs: Vec<f32> = (0..input_len).map(|i| (i as f32 * 0.017).sin() * 0.5).collect();
    let weight_values: Vec<f32> = (0..weight_len).map(|i| ((i as f32) * 0.09).cos() * 0.2).collect();
    let bias_values: Vec<f32> = vec![0.01, -0.02];
    let expected = causal_conv_transpose1d_causal_pad_reference(CausalConvTranspose1dSpec {
        input: &input_values_ncs,
        weight: &weight_values,
        bias: &bias_values,
        lengths: &lengths_out,
        batch_size,
        cin,
        cout,
        seq_len_in,
        seq_len_out,
        stride,
        groups,
    })
    .expect("reference causal conv transpose lalamo");

    let mut input_values_nsc = vec![0.0_f32; input_len];
    for b in 0..batch_size {
        for c in 0..cin {
            for t in 0..seq_len_in {
                let src_index = (b * cin + c) * seq_len_in + t;
                let dst_index = (b * seq_len_in + t) * cin + c;
                input_values_nsc[dst_index] = input_values_ncs[src_index];
            }
        }
    }

    for (layout_name, layout_code, input_shape, input_values) in [
        ("NCS", 0_i32, [batch_size, cin, seq_len_in], input_values_ncs.as_slice()),
        ("NSC", 1_i32, [batch_size, seq_len_in, cin], input_values_nsc.as_slice()),
    ] {
        let mut input =
            context.create_array_zeros(&input_shape, DataType::F32, "audio_causal_conv_transpose_lalamo_input");
        input.as_slice_mut::<f32>().copy_from_slice(input_values);

        let mut weight = context.create_array_zeros(
            &[cout, cin / groups, kernel_size],
            DataType::F32,
            "audio_causal_conv_transpose_lalamo_weight",
        );
        weight.as_slice_mut::<f32>().copy_from_slice(&weight_values);

        let mut bias = context.create_array_zeros(&[cout], DataType::F32, "audio_causal_conv_transpose_lalamo_bias");
        bias.as_slice_mut::<f32>().copy_from_slice(&bias_values);

        let mut lengths =
            context.create_array_zeros(&[batch_size], DataType::I32, "audio_causal_conv_transpose_lalamo_lengths");
        lengths.as_slice_mut::<i32>().copy_from_slice(&lengths_out);

        let mut output = context
            .create_array_zeros(
            &[batch_size, cout, seq_len_out],
            DataType::F32,
            "audio_causal_conv_transpose_lalamo_output",
        )
            .into_allocation();

        run_command_buffer(&context, |command_buffer| {
            borrow_array_buffer!(input_buffer = input);
            borrow_array_buffer!(weight_buffer = weight);
            borrow_array_buffer!(bias_buffer = bias);
            borrow_array_buffer_mut!(output_buffer = output);
            borrow_array_buffer!(lengths_buffer = lengths);
            kernel.encode(
                input_buffer,
                weight_buffer,
                bias_buffer,
                output_buffer,
                lengths_buffer,
                cin as i32,
                cout as i32,
                seq_len_in as i32,
                seq_len_out as i32,
                kernel_size as i32,
                stride as i32,
                groups as i32,
                layout_code,
                batch_size as i32,
                command_buffer,
            );
        });

        let got = allocation_as_slice::<f32, Metal>(&output);
        for index in 0..output_len {
            let delta = (expected[index] - got[index]).abs();
            assert!(
                delta <= 1e-4,
                "layout={layout_name}, index={index}: expected {}, got {}, delta={delta}",
                expected[index],
                got[index]
            );
        }
    }
}

#[uzu_test]
fn audio_leaky_relu_matches_reference_f32() {
    let context = create_test_context();
    let kernel = <<Metal as Backend>::Kernels as Kernels>::AudioHalfSnakeKernel::new(&context, DataType::F32)
        .expect("audio runtime");

    let n = 1024usize;
    let slope = 0.0f32;
    let eps = 1e-6f32;
    let input_values: Vec<f32> = (0..n).map(|i| i as f32 * 0.01 - 5.12).collect();

    let mut input = context.create_array_zeros(&[1, 1, n], DataType::F32, "audio_leaky_relu_input");
    input.as_slice_mut::<f32>().copy_from_slice(&input_values);

    let alpha = context.create_array_zeros(&[1], DataType::F32, "audio_leaky_relu_alpha");
    let mut output = context.create_array_zeros(&[1, 1, n], DataType::F32, "audio_leaky_relu_output").into_allocation();

    run_command_buffer(&context, |command_buffer| {
        borrow_array_buffer!(input_buffer = input);
        borrow_array_buffer!(alpha_buffer = alpha);
        borrow_array_buffer_mut!(output_buffer = output);
        kernel.encode(
            input_buffer,
            alpha_buffer,
            output_buffer,
            1,
            n as i32,
            0,
            slope,
            eps,
            1,
            command_buffer,
        );
    });

    let got = allocation_as_slice::<f32, Metal>(&output);
    for index in 0..n {
        let expected = if input_values[index] >= 0.0 {
            input_values[index]
        } else {
            slope * input_values[index]
        };
        let delta = (expected - got[index]).abs();
        assert!(delta <= 1e-6, "Mismatch at index={index}: expected {expected}, got {}, delta={delta}", got[index]);
    }
}

#[uzu_test]
fn audio_tanh_matches_reference_f32() {
    let context = create_test_context();
    let kernel = <<Metal as Backend>::Kernels as Kernels>::ActivationKernel::new(&context, DataType::F32, false)
        .expect("audio runtime");

    let n = 1024usize;
    let input_values: Vec<f32> = (0..n).map(|i| i as f32 * 0.01 - 5.12).collect();

    let mut input = context.create_array_zeros(&[n], DataType::F32, "audio_tanh_input");
    input.as_slice_mut::<f32>().copy_from_slice(&input_values);

    let mut output = context.create_array_zeros(&[n], DataType::F32, "audio_tanh_output").into_allocation();

    run_command_buffer(&context, |command_buffer| {
        borrow_array_buffer!(input_buffer = input);
        borrow_array_buffer_mut!(output_buffer = output);
        kernel.encode(Some(input_buffer), output_buffer, n as u32, ActivationType::TANH, command_buffer);
    });

    let got = allocation_as_slice::<f32, Metal>(&output);
    for index in 0..n {
        let expected = input_values[index].tanh();
        let delta = (expected - got[index]).abs();
        assert!(delta <= 1e-6, "Mismatch at index={index}: expected {expected}, got {}, delta={delta}", got[index]);
    }
}

#[uzu_test]
fn audio_add_matches_reference_f32() {
    let context = create_test_context();
    let add_kernel = <<Metal as Backend>::Kernels as Kernels>::AudioAddKernel::new(&context, DataType::F32)
        .expect("audio add kernel");

    let n = 2048usize;
    let a_values: Vec<f32> = (0..n).map(|i| i as f32 * 0.001 - 1.0).collect();
    let b_values: Vec<f32> = (0..n).map(|i| (i as f32 * 0.002).sin()).collect();

    let mut a = context.create_array_zeros(&[n], DataType::F32, "audio_add_a");
    a.as_slice_mut::<f32>().copy_from_slice(&a_values);

    let mut b = context.create_array_zeros(&[n], DataType::F32, "audio_add_b");
    b.as_slice_mut::<f32>().copy_from_slice(&b_values);

    let mut sum = context.create_array_zeros(&[n], DataType::F32, "audio_add_sum").into_allocation();

    run_command_buffer(&context, |command_buffer| {
        borrow_array_buffer!(a_buffer = a);
        borrow_array_buffer!(b_buffer = b);
        borrow_array_buffer_mut!(sum_buffer = sum);
        add_kernel.encode(a_buffer, b_buffer, sum_buffer, n as i32, command_buffer);
    });

    let sum_got = allocation_as_slice::<f32, Metal>(&sum);

    for index in 0..n {
        let expected_sum = a_values[index] + b_values[index];
        let sum_delta = (expected_sum - sum_got[index]).abs();

        assert!(
            sum_delta <= 1e-6,
            "Add mismatch at index={index}: expected {expected_sum}, got {}, delta={sum_delta}",
            sum_got[index]
        );
    }
}

#[uzu_test]
fn audio_half_snake_matches_reference_f32() {
    let context = create_test_context();
    let kernel = <<Metal as Backend>::Kernels as Kernels>::AudioHalfSnakeKernel::new(&context, DataType::F32)
        .expect("audio runtime");

    let batch_size = 1usize;
    let channels = 6usize;
    let snake_channels = channels / 2;
    let seq_len = 64usize;
    let n = batch_size * channels * seq_len;

    let negative_slope = 0.01f32;
    let eps = 1e-9f32;

    let input_values: Vec<f32> = (0..n).map(|i| (i as f32 * 0.01 - 3.2).sin()).collect();
    let alpha_values: Vec<f32> = (0..snake_channels).map(|i| 0.5 + i as f32 * 0.1).collect();

    let mut input =
        context.create_array_zeros(&[batch_size, channels, seq_len], DataType::F32, "audio_half_snake_input");
    input.as_slice_mut::<f32>().copy_from_slice(&input_values);

    let mut alpha = context.create_array_zeros(&[snake_channels], DataType::F32, "audio_half_snake_alpha");
    alpha.as_slice_mut::<f32>().copy_from_slice(&alpha_values);

    let mut output = context
        .create_array_zeros(&[batch_size, channels, seq_len], DataType::F32, "audio_half_snake_output")
        .into_allocation();

    run_command_buffer(&context, |command_buffer| {
        borrow_array_buffer!(input_buffer = input);
        borrow_array_buffer!(alpha_buffer = alpha);
        borrow_array_buffer_mut!(output_buffer = output);
        kernel.encode(
            input_buffer,
            alpha_buffer,
            output_buffer,
            channels as i32,
            seq_len as i32,
            snake_channels as i32,
            negative_slope,
            eps,
            batch_size as i32,
            command_buffer,
        );
    });

    let got = allocation_as_slice::<f32, Metal>(&output);
    let expected = half_snake_reference(HalfSnakeSpec {
        input: &input_values,
        alpha: &alpha_values,
        batch_size,
        channels,
        seq_len,
        snake_channels,
        negative_slope,
        eps,
    })
    .expect("reference half snake");
    for index in 0..n {
        let delta = (expected[index] - got[index]).abs();
        assert!(
            delta <= 1e-5,
            "Mismatch at index={index}: expected {}, got {}, delta={delta}",
            expected[index],
            got[index]
        );
    }
}

#[uzu_test]
fn audio_norm_ncs_matches_reference_f32() {
    let context = create_test_context();
    let kernel = <<Metal as Backend>::Kernels as Kernels>::AudioNormNcsKernel::new(&context, DataType::F32)
        .expect("audio runtime");

    let batch_size = 2usize;
    let channels = 7usize;
    let seq_len = 11usize;
    let lengths_values: [i32; 2] = [8, 11];

    let input_len = batch_size * channels * seq_len;
    let input_values: Vec<f32> = (0..input_len).map(|i| (i as f32 * 0.013 - 3.0).sin() * 0.7).collect();
    let scales_values: Vec<f32> = (0..channels).map(|i| 0.7 + i as f32 * 0.05).collect();
    let bias_values: Vec<f32> = (0..channels).map(|i| -0.2 + i as f32 * 0.03).collect();
    let epsilon = 1e-5f32;

    let reference = |subtract_mean: bool| -> Vec<f32> {
        let mut out = vec![0.0_f32; input_values.len()];
        for batch in 0..batch_size {
            let active_len = lengths_values[batch].max(0) as usize;
            for t in 0..active_len {
                let mut mean = 0.0_f32;
                if subtract_mean {
                    for c in 0..channels {
                        let idx = (batch * channels + c) * seq_len + t;
                        mean += input_values[idx];
                    }
                    mean /= channels as f32;
                }
                let mut variance_sum = 0.0_f32;
                for c in 0..channels {
                    let idx = (batch * channels + c) * seq_len + t;
                    let centered = if subtract_mean {
                        input_values[idx] - mean
                    } else {
                        input_values[idx]
                    };
                    variance_sum += centered * centered;
                }
                let inv_std = 1.0_f32 / (variance_sum / channels as f32 + epsilon).sqrt();
                for c in 0..channels {
                    let idx = (batch * channels + c) * seq_len + t;
                    let centered = if subtract_mean {
                        input_values[idx] - mean
                    } else {
                        input_values[idx]
                    };
                    out[idx] = centered * inv_std * scales_values[c] + bias_values[c];
                }
            }
        }
        out
    };

    for subtract_mean in [false, true] {
        let mut input = context.create_array_zeros(&[batch_size, channels, seq_len], DataType::F32, "audio_norm_input");
        input.as_slice_mut::<f32>().copy_from_slice(&input_values);

        let mut scales = context.create_array_zeros(&[channels], DataType::F32, "audio_norm_scales");
        scales.as_slice_mut::<f32>().copy_from_slice(&scales_values);

        let mut bias = context.create_array_zeros(&[channels], DataType::F32, "audio_norm_bias");
        bias.as_slice_mut::<f32>().copy_from_slice(&bias_values);

        let mut lengths = context.create_array_zeros(&[batch_size], DataType::I32, "audio_norm_lengths");
        lengths.as_slice_mut::<i32>().copy_from_slice(&lengths_values);

        let mut output = context
            .create_array_zeros(&[batch_size, channels, seq_len], DataType::F32, "audio_norm_output")
            .into_allocation();

        run_command_buffer(&context, |command_buffer| {
            borrow_array_buffer!(input_buffer = input);
            borrow_array_buffer!(scales_buffer = scales);
            borrow_array_buffer!(bias_buffer = bias);
            borrow_array_buffer_mut!(output_buffer = output);
            borrow_array_buffer!(lengths_buffer = lengths);
            kernel.encode(
                input_buffer,
                scales_buffer,
                bias_buffer,
                output_buffer,
                lengths_buffer,
                channels as i32,
                seq_len as i32,
                epsilon,
                if subtract_mean {
                    1
                } else {
                    0
                },
                batch_size as i32,
                command_buffer,
            );
        });

        let got = allocation_as_slice::<f32, Metal>(&output);
        let expected = reference(subtract_mean);
        for index in 0..input_len {
            let delta = (expected[index] - got[index]).abs();
            assert!(
                delta <= 1e-4,
                "subtract_mean={subtract_mean}, index={index}: expected {}, got {}, delta={delta}",
                expected[index],
                got[index]
            );
        }
    }
}

#[uzu_test]
fn audio_fsq_decode_matches_reference() {
    let context = create_test_context();
    let kernel = <<Metal as Backend>::Kernels as Kernels>::AudioFsqDecodeKernel::new(&context, DataType::F32)
        .expect("audio runtime");

    let batch_size = 1usize;
    let num_groups = 2usize;
    let seq_len = 3usize;
    let codebook_dim = 2usize;
    let num_levels: Vec<i32> = vec![8, 5];
    let dim_base_index: Vec<i32> = vec![1, 8];

    let mut tokens = context.create_array_zeros(&[batch_size, num_groups, seq_len], DataType::I32, "audio_fsq_tokens");
    tokens.as_slice_mut::<i32>().copy_from_slice(&[
        0, 7, 11, // g0 tokens at t0..t2
        3, 5, 9, // g1 tokens at t0..t2
    ]);

    let mut lengths = context.create_array_zeros(&[batch_size], DataType::I32, "audio_fsq_lengths");
    lengths.as_slice_mut::<i32>().copy_from_slice(&[2]);

    let mut output = context
        .create_array_zeros(
        &[batch_size, num_groups * codebook_dim, seq_len],
        DataType::F32,
        "audio_fsq_output",
    )
        .into_allocation();

    run_command_buffer(&context, |command_buffer| {
        borrow_array_buffer!(tokens_buffer = tokens);
        borrow_array_buffer_mut!(output_buffer = output);
        borrow_array_buffer!(lengths_buffer = lengths);
        kernel.encode(
            tokens_buffer,
            output_buffer,
            lengths_buffer,
            num_groups as i32,
            seq_len as i32,
            codebook_dim as i32,
            &num_levels,
            &dim_base_index,
            batch_size as i32,
            command_buffer,
        );
    });

    let expected =
        fsq_decode_reference(&[0, 7, 11, 3, 5, 9], &[2], batch_size, num_groups, seq_len, codebook_dim, &num_levels)
            .expect("reference decode");
    let actual = allocation_as_slice::<f32, Metal>(&output);

    for (index, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
        let delta = (a - e).abs();
        assert!(delta <= 1e-5, "index={index}, actual={a}, expected={e}, delta={delta}");
    }
}

#[uzu_test]
fn audio_fsq_encode_matches_reference() {
    let context = create_test_context();
    let kernel = <<Metal as Backend>::Kernels as Kernels>::AudioFsqEncodeKernel::new(&context, DataType::F32)
        .expect("audio runtime");

    let batch_size = 1usize;
    let num_groups = 2usize;
    let codebook_dim = 2usize;
    let seq_len = 4usize;
    let eps = 1e-3_f32;
    let num_levels: Vec<i32> = vec![8, 6];
    let dim_base_index: Vec<i32> = vec![1, 8];

    let input_values = vec![
        -0.9, -0.1, 0.2, 0.7, // g0 d0
        0.8, 0.3, -0.4, -0.7, // g0 d1
        -0.2, 0.0, 0.4, 0.9, // g1 d0
        0.5, -0.5, 0.1, -0.1, // g1 d1
    ];
    let lengths_values = vec![3_i32];

    let mut input = context.create_array_zeros(
        &[batch_size, num_groups * codebook_dim, seq_len],
        DataType::F32,
        "audio_fsq_encode_input",
    );
    input.as_slice_mut::<f32>().copy_from_slice(&input_values);

    let mut lengths = context.create_array_zeros(&[batch_size], DataType::I32, "audio_fsq_encode_lengths");
    lengths.as_slice_mut::<i32>().copy_from_slice(&lengths_values);

    let mut tokens = context
        .create_array_zeros(&[batch_size, num_groups, seq_len], DataType::I32, "audio_fsq_encode_tokens")
        .into_allocation();

    run_command_buffer(&context, |command_buffer| {
        borrow_array_buffer!(input_buffer = input);
        borrow_array_buffer_mut!(tokens_buffer = tokens);
        borrow_array_buffer!(lengths_buffer = lengths);
        kernel.encode(
            input_buffer,
            tokens_buffer,
            lengths_buffer,
            num_groups as i32,
            seq_len as i32,
            codebook_dim as i32,
            &num_levels,
            &dim_base_index,
            eps,
            batch_size as i32,
            command_buffer,
        );
    });

    let expected = fsq_encode_reference(
        &input_values,
        &lengths_values,
        batch_size,
        num_groups,
        seq_len,
        codebook_dim,
        &num_levels,
        &dim_base_index,
        eps,
    )
    .expect("reference encode");
    let actual = allocation_as_slice::<i32, Metal>(&tokens);

    assert_eq!(actual, expected);
}
