//! P2l (b): Weaver gathered readout GEMV, [G,N] (production) vs [N,G] (OutputGroup) scale/zero-point layout.
//! `AT_P2L_OUT=/abs/result.json cargo test --release -p uzu-engine p2l_gather -- --ignored --nocapture`
use std::time::{Duration, Instant};

use futures::executor::block_on;
use half::bf16;
use rand::{RngExt, SeedableRng, rngs::SmallRng};
use serde_json::json;
use uzu_engine_macros::uzu_test;

use crate::{
    backends::{
        common::{
            Backend, CommandBufferCompleted, CommandBufferEncoding, CommandBufferExecutable, CommandBufferPending,
            Context, Kernels,
            gpu_types::quantization_method::QuantizationMethod,
            kernel::matmul::{MatmulKernel, QuantParamsLayout},
        },
        metal::{Metal, context::MetalContext},
    },
    data_type::DataTypeRepr,
    tests::{
        helpers::{buffer_to_vec, create_buffer_with_data},
        matmul::{QuantBuffers, QuantInput, quant_arguments},
    },
};

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    values[values.len() / 2]
}

#[uzu_test]
#[ignore]
fn p2l_gather() {
    let context = block_on(MetalContext::new()).expect("Metal context");
    let rows: u32 = std::env::var("P2L_ROWS").ok().and_then(|v| v.parse().ok()).unwrap_or(8);
    let candidates: u32 = std::env::var("P2L_CAND").ok().and_then(|v| v.parse().ok()).unwrap_or(512);
    let group: u32 = std::env::var("P2L_GROUP").ok().and_then(|v| v.parse().ok()).unwrap_or(32);
    let rounds: usize = std::env::var("P2L_ROUNDS").ok().and_then(|v| v.parse().ok()).unwrap_or(9);
    let (k, vocab) = (4096u32, 248320u32);
    let dispatches = 32usize;
    let mut arms: Vec<(String, QuantParamsLayout)> = vec![
        ("GroupOutput".into(), QuantParamsLayout::GroupOutput),
        ("OutputGroup".into(), QuantParamsLayout::OutputGroup),
    ];
    arms.push(("GroupOutput_ctl".into(), QuantParamsLayout::GroupOutput));
    let make = |layout: QuantParamsLayout| {
        let input = QuantInput::<bf16>::new(rows, k, vocab, group, 4, QuantizationMethod::ScaleZeroPoint, 7);
        match layout {
            QuantParamsLayout::GroupOutput => input.with_group_output(),
            _ => input,
        }
    };
    let mut rng = SmallRng::seed_from_u64(11);
    let id_sets: Vec<Vec<u32>> =
        (0..dispatches).map(|_| (0..rows * candidates).map(|_| rng.random_range(0..vocab)).collect()).collect();
    let id_buffers: Vec<_> = id_sets.iter().map(|ids| create_buffer_with_data::<Metal, u32>(&context, ids)).collect();
    let mut matmuls: Vec<_> = arms
        .iter()
        .map(|_| {
            <<Metal as Backend>::Kernels as Kernels>::MatmulKernel::new(
                &context,
                bf16::data_type(),
                bf16::data_type(),
                bf16::data_type(),
            )
            .unwrap()
        })
        .collect();
    let inputs: Vec<_> = arms.iter().map(|(_, layout)| make(*layout)).collect();
    let mut buffers: Vec<_> =
        inputs.iter().map(|input| QuantBuffers::<Metal, bf16>::allocate(&context, input)).collect();

    // Output equality between the layouts (same logical weights, same arithmetic).
    let mut outputs = Vec::new();
    for (index, _) in arms.iter().enumerate().take(2) {
        let mut command_buffer = context.create_command_buffer(None, None).unwrap();
        let mut arguments = quant_arguments(&mut buffers[index], &inputs[index]);
        arguments.gather_indices = Some(&id_buffers[0]);
        arguments.n = candidates;
        matmuls[index].encode(arguments, &mut command_buffer).unwrap();
        block_on(command_buffer.end_encoding().submit().wait_until_completed()).unwrap();
        outputs
            .push(buffer_to_vec::<Metal, bf16>(&context, &buffers[index].y)[..(rows * candidates) as usize].to_vec());
    }
    let identical = outputs[0].iter().zip(&outputs[1]).all(|(a, b)| a.to_bits() == b.to_bits());
    let max_abs = outputs[0].iter().zip(&outputs[1]).map(|(a, b)| (a.to_f32() - b.to_f32()).abs()).fold(0f32, f32::max);

    let mut run = |index: usize| -> f64 {
        let mut command_buffer = context.create_command_buffer(None, None).unwrap();
        for id in &id_buffers {
            let mut arguments = quant_arguments(&mut buffers[index], &inputs[index]);
            arguments.gather_indices = Some(id);
            arguments.n = candidates;
            matmuls[index].encode(arguments, &mut command_buffer).unwrap();
        }
        let completed = block_on(command_buffer.end_encoding().submit().wait_until_completed()).unwrap();
        completed.gpu_execution_time().unwrap().as_secs_f64() * 1e6 / dispatches as f64
    };
    let warm = Instant::now();
    while warm.elapsed() < Duration::from_millis(800) {
        for index in 0..arms.len() {
            run(index);
        }
    }
    let mut samples: Vec<Vec<f64>> = vec![Vec::new(); arms.len()];
    for round in 0..rounds {
        let order: Vec<usize> = if round % 2 == 0 {
            (0..arms.len()).collect()
        } else {
            (0..arms.len()).rev().collect()
        };
        for index in order {
            samples[index].push(run(index));
        }
    }
    let result: Vec<_> = arms
        .iter()
        .enumerate()
        .map(|(index, (name, _))| {
            let mut values = samples[index].clone();
            json!({"arm": name, "median_us": median(&mut values), "all_us": samples[index]})
        })
        .collect();
    let text = serde_json::to_string_pretty(&json!({
        "device": context.device_name, "rows": rows, "candidates": candidates, "group": group, "k": k, "vocab": vocab,
        "identical_outputs": identical, "max_abs_diff": max_abs, "arms": result,
    }))
    .unwrap();
    match std::env::var("AT_P2L_OUT") {
        Ok(path) => std::fs::write(path, text).unwrap(),
        Err(_) => println!("{text}"),
    }
}

/// CPU cost of the load-time relayout (`pack_small_batch_units`) for one 9B-sized linear.
#[uzu_test]
#[ignore]
fn p2l_pack_cost() {
    use crate::backends::{
        common::kernel::matmul::{QuantParams, QuantParamsLayout},
        metal::kernel::matmul::gemm::small_batch_units::{Offsets, pack_small_batch_units},
    };
    let (n, k, group) = (24576usize, 4096usize, 32usize);
    let groups = k / group;
    let params = QuantParams::new(QuantParamsLayout::GroupOutput, n as u32, groups as u32);
    let mut rng = SmallRng::seed_from_u64(3);
    let codes: Vec<u32> = (0..n * k / 8).map(|_| rng.random()).collect();
    let scales: Vec<bf16> = (0..groups * n).map(|_| bf16::from_f32(0.01)).collect();
    let zero_points: Vec<u8> = (0..groups * n / 2).map(|_| rng.random()).collect();
    let start = Instant::now();
    let units = pack_small_batch_units(
        n,
        k,
        group,
        &codes,
        &scales,
        params.scale_strides(),
        Offsets::ZeroPoints(
            &zero_points,
            params.zero_point_strides(crate::backends::common::gpu_types::quantization::QuantizationMode::U4),
        ),
    );
    let ms = start.elapsed().as_secs_f64() * 1e3;
    println!(
        "p2l_pack_cost: gate_up {n}x{k} g{group}: {ms:.1} ms, {:.1} MB in, {:.1} MB units, {} threads",
        (codes.len() * 4 + scales.len() * 2 + zero_points.len()) as f64 / 1e6,
        units.len() as f64 * 4.0 / 1e6,
        std::thread::available_parallelism().map_or(1, |t| t.get())
    );
}
