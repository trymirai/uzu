#![cfg(backend = "metal")]

use futures::executor::block_on;
use half::bf16;
use num_traits::Float;
use uzu_engine_macros::uzu_test;

use crate::{
    backends::{
        common::{
            Backend, CommandBufferEncoding, CommandBufferExecutable, CommandBufferPending, Context, Kernels,
            kernel::TreeVerifyKernel,
        },
        cpu::Cpu,
        metal::Metal,
    },
    data_type::DataTypeRepr,
    tests::{
        assert::assert_eq_float,
        helpers::{buffer_to_vec, create_buffer_with_data},
    },
};

const K_HEADS: usize = 2;
const V_HEADS: usize = 6;
const DIM: usize = 128;
const CONV_DIM: usize = 5;
const PROJ_DIM: usize = CONV_DIM + V_HEADS * DIM + 3;

fn run<B: Backend, T: DataTypeRepr + Float>(n: usize) -> Vec<T> {
    let context = block_on(B::Context::new()).expect("context");
    let kernel = <B::Kernels as Kernels>::TreeVerifyKernel::new(&context, T::data_type(), (n > 16) as u32 + 1).unwrap();
    let wave = |len: usize, step: f32, amplitude: f32, bias: f32| -> Vec<f32> {
        (0..len).map(|i| (i as f32 * step).sin() * amplitude + bias).collect()
    };
    let typed = |data: Vec<f32>| {
        create_buffer_with_data::<B, T>(&context, &data.iter().map(|&x| T::from(x).unwrap()).collect::<Vec<_>>())
    };
    let float = |data: Vec<f32>| create_buffer_with_data::<B, f32>(&context, &data);
    // Caterpillar: odd nodes are leaves, even nodes the spine, so every 16-node block branches.
    let trie: Vec<u32> = (0..n as u32).flat_map(|node| [node, [n as u32 - 1, node][node as usize % 2], 0]).collect();
    let mut o = create_buffer_with_data::<B, T>(&context, &vec![T::zero(); n * V_HEADS * DIM]);

    let mut command_buffer = context.create_command_buffer(None, None).unwrap();
    kernel.encode(
        &typed(wave(n * K_HEADS * DIM, 0.017, 0.2, 0.01)),
        &typed(wave(n * K_HEADS * DIM, 0.019, 0.18, -0.02)),
        &typed(wave(n * V_HEADS * DIM, 0.023, 0.5, 0.03)),
        &create_buffer_with_data::<B, u32>(&context, &trie),
        &float(wave(n * V_HEADS, 1.7, 0.2, -0.25)),
        &float(wave(n * V_HEADS, 0.013, 0.2, 0.45)),
        &float(wave(V_HEADS * DIM * DIM, 0.007, 0.05, -0.01)),
        &typed(wave(n * PROJ_DIM, 0.031, 1.5, -0.1)),
        &float(wave(DIM, 0.05, 0.2, 1.0)),
        &mut o,
        1.0,
        1e-6,
        1,
        n as u32,
        K_HEADS as u32,
        V_HEADS as u32,
        DIM as u32,
        DIM as u32,
        CONV_DIM as u32,
        PROJ_DIM as u32,
        &mut command_buffer,
    );
    block_on(command_buffer.end_encoding().submit().wait_until_completed()).unwrap();
    buffer_to_vec(&o)
}

#[uzu_test]
fn test_tree_verify() {
    for n in [1, 16, 17, 32] {
        assert_eq_float(&run::<Cpu, f32>(n), &run::<Metal, f32>(n), 5e-4, "f32");
        assert_eq_float(&run::<Cpu, bf16>(n), &run::<Metal, bf16>(n), 1.6e-2, "bf16");
    }
}
