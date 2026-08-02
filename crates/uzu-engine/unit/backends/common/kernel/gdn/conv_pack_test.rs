use std::fmt::{Debug, Display};

use futures::executor::block_on;
use half::bf16;
use num_traits::Float;
use uzu_engine_macros::uzu_test;

use crate::{
    backends::{
        common::{
            Backend, CommandBufferEncoding, CommandBufferExecutable, CommandBufferPending, Context, Kernels,
            kernel::DeltaNetConvPackKernel,
        },
        cpu::Cpu,
    },
    data_type::DataTypeRepr,
    tests::{
        assert::assert_eq_float,
        helpers::{buffer_to_vec, create_buffer, create_buffer_with_data, for_each_non_cpu_backend},
    },
};

const KERNEL_SIZE: u32 = 4;
const TAPS: u32 = KERNEL_SIZE - 1;

struct Input<T: DataTypeRepr + Float> {
    state_in: Box<[f32]>,
    x: Box<[T]>,
    suffix_len: u32,
    tile_tokens: u32,
    row_stride: u32,
    conv_dim: u32,
}

fn get_input<T: DataTypeRepr + Float>(
    suffix_len: u32,
    tile_tokens: u32,
    row_stride: u32,
    conv_dim: u32,
) -> Input<T> {
    let state_in: Vec<f32> = (0..conv_dim * TAPS).map(|i| 0.1 * (i as f32 + 1.0)).collect();
    let x: Vec<T> = (0..suffix_len * row_stride).map(|i| T::from(-0.05 * (i as f64 + 1.0)).unwrap()).collect();

    Input {
        state_in: state_in.into_boxed_slice(),
        x: x.into_boxed_slice(),
        suffix_len,
        tile_tokens,
        row_stride,
        conv_dim,
    }
}

fn get_output<B: Backend, T: DataTypeRepr + Float>(input: &Input<T>) -> Vec<f32> {
    let context = block_on(B::Context::new()).expect("Failed to create Context");
    let kernel =
        <<B as Backend>::Kernels as Kernels>::DeltaNetConvPackKernel::new(&context, T::data_type(), KERNEL_SIZE)
            .expect("Failed to create DeltaNetConvPackKernel");

    let num_tiles = input.suffix_len.div_ceil(input.tile_tokens);
    let state = create_buffer_with_data::<B, f32>(&context, &input.state_in);
    let x = create_buffer_with_data::<B, T>(&context, &input.x);
    let mut packed = create_buffer::<B, f32>(&context, (num_tiles * TAPS * input.conv_dim) as usize);

    let mut command_buffer = context.create_command_buffer(None, None).expect("Failed to create command buffer");
    kernel.encode(
        &state,
        &x,
        &mut packed,
        input.suffix_len,
        input.tile_tokens,
        input.row_stride,
        input.conv_dim,
        &mut command_buffer,
    );
    block_on(command_buffer.end_encoding().submit().wait_until_completed()).expect("Failed to wait command buffer");

    buffer_to_vec(&*context, &packed)
}

fn test_internal<T: DataTypeRepr + Float + Debug + Display>(
    suffix_len: u32,
    tile_tokens: u32,
    row_stride: u32,
    conv_dim: u32,
    label: &str,
) {
    let input = get_input::<T>(suffix_len, tile_tokens, row_stride, conv_dim);
    let expected = get_output::<Cpu, T>(&input);

    for_each_non_cpu_backend!(|B| {
        let output = get_output::<B, T>(&input);
        let msg =
            format!("DeltaNetConvPack {} {} (type={})", std::any::type_name::<B>(), label, std::any::type_name::<T>());
        assert_eq_float::<f32>(&expected, &output, 1e-6, &msg);
    });
}

fn test_basic<T: DataTypeRepr + Float + Debug + Display>() {
    // several tiles, partial last tile
    test_internal::<T>(20, 8, 16, 8, "basic");
}

fn test_single_tile<T: DataTypeRepr + Float + Debug + Display>() {
    // a single tile takes only the carried state
    test_internal::<T>(5, 16, 8, 8, "single_tile");
}

fn test_exact_tiles<T: DataTypeRepr + Float + Debug + Display>() {
    test_internal::<T>(64, 16, 24, 8, "exact_tiles");
}

fn test_channels_lt_row_stride<T: DataTypeRepr + Float + Debug + Display>() {
    // conv_dim < row_stride (trailing projection channels are not packed)
    test_internal::<T>(30, 4, 40, 32, "channels_lt_row_stride");
}

// f32
#[uzu_test]
fn test_basic_f32() {
    test_basic::<f32>();
}

#[uzu_test]
fn test_single_tile_f32() {
    test_single_tile::<f32>();
}

#[uzu_test]
fn test_exact_tiles_f32() {
    test_exact_tiles::<f32>();
}

#[uzu_test]
fn test_channels_lt_row_stride_f32() {
    test_channels_lt_row_stride::<f32>();
}

// bf16
#[uzu_test]
fn test_basic_bf16() {
    test_basic::<bf16>();
}

#[uzu_test]
fn test_single_tile_bf16() {
    test_single_tile::<bf16>();
}

#[uzu_test]
fn test_exact_tiles_bf16() {
    test_exact_tiles::<bf16>();
}

#[uzu_test]
fn test_channels_lt_row_stride_bf16() {
    test_channels_lt_row_stride::<bf16>();
}
