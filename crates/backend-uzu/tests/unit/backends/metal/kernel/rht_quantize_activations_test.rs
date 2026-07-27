#![cfg(backend = "metal")]

use proc_macros::uzu_test;
use rand::{RngExt, SeedableRng, rngs::SmallRng};

use super::RHTQuantizeActivationsMetalKernel;
use crate::{
    backends::{
        common::{
            Backend, Context, Encoder,
            kernel::{Kernels, RHTQuantizeActivationsKernel},
        },
        cpu::Cpu,
        metal::{Metal, MetalContext},
    },
    data_type::DataType,
    tests::helpers::{alloc_allocation, alloc_allocation_with_data, allocation_to_vec},
};

#[uzu_test]
fn rht_quantize_matches_cpu() {
    let rows = 5usize;
    let columns = 96usize;
    let group_size = 32u32;
    let groups = columns.div_ceil(group_size as usize);

    let mut rng = SmallRng::seed_from_u64(0x5EED_0001);
    let input_data: Vec<f32> = (0..rows * columns).map(|_| rng.random_range(-1.0f32..1.0f32)).collect();
    let factors_data: Vec<i32> = (0..columns)
        .map(|i| {
            if i % 3 == 0 {
                -1
            } else {
                1
            }
        })
        .collect();

    let metal = MetalContext::new().expect("metal");
    let cpu = <Cpu as Backend>::Context::new().expect("cpu");
    let mut metal_values = alloc_allocation::<Metal, i8>(&metal, rows * columns);
    let mut metal_scales = alloc_allocation::<Metal, f32>(&metal, rows * groups);
    let mut cpu_values = alloc_allocation::<Cpu, i8>(&cpu, rows * columns);
    let mut cpu_scales = alloc_allocation::<Cpu, f32>(&cpu, rows * groups);
    let mut metal_group_sums = alloc_allocation::<Metal, i32>(&metal, rows * groups);
    let mut cpu_group_sums = alloc_allocation::<Cpu, i32>(&cpu, rows * groups);

    let metal_input = alloc_allocation_with_data::<Metal, f32>(&metal, &input_data);
    let metal_factors = alloc_allocation_with_data::<Metal, i32>(&metal, &factors_data);
    let cpu_input = alloc_allocation_with_data::<Cpu, f32>(&cpu, &input_data);
    let cpu_factors = alloc_allocation_with_data::<Cpu, i32>(&cpu, &factors_data);

    let metal_kernel = RHTQuantizeActivationsMetalKernel::new(&metal, DataType::F32, true).expect("metal prepare");
    let cpu_kernel =
        <<Cpu as Backend>::Kernels as Kernels>::RHTQuantizeActivationsKernel::new(&cpu, DataType::F32, true)
            .expect("cpu prepare");

    let mut metal_enc = Encoder::<Metal>::new(&metal).expect("metal encoder");
    metal_kernel.encode(
        &metal_input,
        &mut metal_values,
        &mut metal_scales,
        Some(&mut metal_group_sums),
        &metal_factors,
        rows as u32,
        columns as u32,
        group_size,
        &mut metal_enc,
    );
    metal_enc.end_encoding().submit().wait_until_completed().unwrap();

    let mut cpu_enc = Encoder::<Cpu>::new(&cpu).expect("cpu encoder");
    cpu_kernel.encode(
        &cpu_input,
        &mut cpu_values,
        &mut cpu_scales,
        Some(&mut cpu_group_sums),
        &cpu_factors,
        rows as u32,
        columns as u32,
        group_size,
        &mut cpu_enc,
    );
    cpu_enc.end_encoding().submit().wait_until_completed().unwrap();

    let (mv, ms) = (allocation_to_vec::<Metal, i8>(&metal_values), allocation_to_vec::<Metal, f32>(&metal_scales));
    let (cv, cs) = (allocation_to_vec::<Cpu, i8>(&cpu_values), allocation_to_vec::<Cpu, f32>(&cpu_scales));
    for (i, (a, e)) in ms.iter().zip(&cs).enumerate() {
        let rel = (a - e).abs() / e.abs().max(1e-6);
        assert!(rel < 1e-3, "scale {i}: {a} != {e}");
    }
    assert!(mv.iter().zip(&cv).all(|(a, e)| (*a as i32 - *e as i32).abs() <= 1));

    let mrs = allocation_to_vec::<Metal, i32>(&metal_group_sums);
    let crs = allocation_to_vec::<Cpu, i32>(&cpu_group_sums);
    for (codes, sums, label) in [(&mv, &mrs, "metal"), (&cv, &crs, "cpu")] {
        for row in 0..rows {
            for group in 0..groups {
                let start = row * columns + group * group_size as usize;
                let expected: i32 = codes[start..start + group_size as usize].iter().map(|code| *code as i32).sum();
                assert_eq!(sums[row * groups + group], expected, "{label} group_sum r{row} g{group}");
            }
        }
    }
    assert!(mrs.iter().zip(&crs).all(|(a, e)| (a - e).abs() <= group_size as i32));
}
