use std::fmt::Write;

use proc_macros::uzu_test;
use xxhash_rust::xxh3::xxh3_64;

use super::*;
use crate::{
    backends::{
        common::{
            gpu_types::gemm::{GemmBPrologueKind, GemmDTransform},
            kernel::matmul::MatmulShape,
        },
        metal::{
            device_profile::{DeviceIdentity, DeviceProfile, DeviceSize, GpuFamily},
            kernel::matmul::{MatmulDispatch, MatmulMetalKernel, gemm::GemmProblem, gemv::GemvSpecialization},
        },
    },
    data_type::DataType,
};

const FROZEN_PLAN_FINGERPRINT: u64 = 5_839_212_743_558_880_136;

const DEVICES: [(&str, DeviceIdentity); 7] = [
    ("m1", DeviceIdentity::M1),
    ("m2", DeviceIdentity::M2),
    ("m2-pro", DeviceIdentity::M2Pro),
    ("m3-max", DeviceIdentity::M3Max),
    ("m4", DeviceIdentity::M4),
    ("m4-pro", DeviceIdentity::M4Pro),
    ("m5-max", DeviceIdentity::M5Max),
];
const FORMATS: [(&str, u32, u32, GemmBPrologueKind); 4] = [
    ("W4/ZP G32", 4, 32, GemmBPrologueKind::ScaleZeroPointDequant),
    ("W4/ZP G64", 4, 64, GemmBPrologueKind::ScaleZeroPointDequant),
    ("W8/Symmetric G32", 8, 32, GemmBPrologueKind::ScaleSymmetricDequant),
    ("W8/Symmetric G64", 8, 64, GemmBPrologueKind::ScaleSymmetricDequant),
];
const SHAPES: [(&str, u32, u32); 7] = [
    ("down", 5120, 17408),
    ("gate", 6144, 5120),
    ("gate-up", 34816, 5120),
    ("projection-in", 16480, 5120),
    ("projection-out", 5120, 6144),
    ("qkv", 8192, 5120),
    ("readout", 248320, 5120),
];

fn problem(
    m: u32,
    n: u32,
    k: u32,
    bits: u32,
    group: u32,
    prologue: GemmBPrologueKind,
) -> MatmulShape {
    MatmulShape {
        m,
        n,
        k,
        b_transpose: true,
        b_leading_dimension: None,
        b_prologue: prologue,
        b_bits: Some(bits),
        b_group_size: Some(group),
        signed_codes: false,
        a_full_precision: true,
        gathered: false,
        d_transform: GemmDTransform::empty(),
    }
}

#[uzu_test]
fn table_is_complete_and_fingerprint_is_stable() {
    let mut canonical = Vec::new();
    let mut matched_rows = vec![false; ROWS.len()];
    let mut tuned = 0;
    let mut main_gemv = 0;
    let mut main_gemm = 0;
    for &(device_name, identity) in &DEVICES {
        for &(format_name, bits, group, prologue) in &FORMATS {
            for m in 2..=7 {
                for &(shape_name, n, k) in &SHAPES {
                    let mask = shape(n, k);
                    let matches: Vec<_> = ROWS
                        .iter()
                        .enumerate()
                        .filter(|row| {
                            row.1.identity == identity
                                && row.1.bits == bits
                                && row.1.group == group
                                && row.1.m == m
                                && row.1.shapes & mask != 0
                        })
                        .collect();
                    assert_eq!(matches.len(), 1, "route coverage for {device_name} {format_name} M={m} {shape_name}");
                    let (row_index, row) = matches[0];
                    matched_rows[row_index] = true;
                    let selected = row.route;
                    match selected {
                        QmvRoute::Tuned(_) => tuned += 1,
                        QmvRoute::MainGemv(_) => main_gemv += 1,
                        QmvRoute::MainGemm(_) => main_gemm += 1,
                    }
                    canonical.push(format!("{device_name}|{format_name}|{shape_name}|{m}|{n}|{k}|{selected:?}"));
                    let profile = device_profile(identity);
                    let problem = problem(m, n, k, bits, group, prologue);
                    assert_eq!(route(profile, &problem, true), Some(selected));
                    let runtime = MatmulMetalKernel::choose_dispatch(
                        &problem,
                        profile,
                        profile.supports_mxu(),
                        DataType::BF16,
                        DataType::BF16,
                        DataType::BF16,
                    );
                    if let QmvRoute::Tuned(tile) | QmvRoute::MainGemv(tile) = selected {
                        let specialization = GemvSpecialization::select_tile(
                            &problem,
                            DataType::BF16,
                            DataType::BF16,
                            DataType::BF16,
                            tile,
                        )
                        .expect("stored GEMV tile must be legal");
                        assert_eq!(specialization.tile(), tile);
                        assert!(matches!(runtime, MatmulDispatch::Gemv(actual) if actual == specialization));
                    } else if let QmvRoute::MainGemm(plan) = selected {
                        let problem =
                            GemmProblem::new(problem, DataType::BF16, DataType::BF16, profile.supports_mxu(), profile);
                        assert!(problem.plan_is_legal(plan), "stored GEMM plan must be legal");
                        assert!(matches!(runtime, MatmulDispatch::Gemm(actual) if actual == plan));
                    }
                }
            }
        }
    }
    assert!(matched_rows.into_iter().all(|matched| matched), "route table contains an orphaned row");
    canonical.sort();
    assert_eq!(canonical.len(), 1176);
    let mut expanded = String::new();
    for line in canonical {
        writeln!(&mut expanded, "{line}").unwrap();
    }
    assert_eq!(tuned, 864);
    assert_eq!(main_gemv, 221);
    assert_eq!(main_gemm, 91);
    assert_eq!(xxh3_64(expanded.trim_end().as_bytes()), FROZEN_PLAN_FINGERPRINT);
}

fn device_profile(identity: DeviceIdentity) -> DeviceProfile {
    let family = match identity {
        DeviceIdentity::M1 => GpuFamily::Legacy,
        DeviceIdentity::M2 | DeviceIdentity::M2Pro => GpuFamily::Apple8,
        DeviceIdentity::M3Max | DeviceIdentity::M4 | DeviceIdentity::M4Pro => GpuFamily::Apple9,
        DeviceIdentity::M5Max => GpuFamily::M5Plus,
        _ => GpuFamily::Legacy,
    };
    let large = matches!(identity, DeviceIdentity::M3Max | DeviceIdentity::M5Max);
    DeviceProfile::new(
        identity,
        family,
        if large {
            DeviceSize::Large
        } else {
            DeviceSize::Small
        },
        matches!(identity, DeviceIdentity::M5Max),
    )
}

#[uzu_test]
fn exact_lookup_rejects_non_matrix_inputs() {
    let profile = device_profile(DeviceIdentity::M1);
    let p = problem(2, 5120, 17408, 4, 32, GemmBPrologueKind::ScaleZeroPointDequant);
    assert!(route(profile, &p, true).is_some());
    for mutate in [
        |p: &mut MatmulShape| p.m = 1,
        |p: &mut MatmulShape| p.n = 1,
        |p: &mut MatmulShape| p.b_bits = Some(8),
        |p: &mut MatmulShape| p.gathered = true,
        |p: &mut MatmulShape| p.d_transform = GemmDTransform::RHT,
        |p: &mut MatmulShape| p.signed_codes = true,
    ] {
        let mut rejected = p;
        mutate(&mut rejected);
        assert!(route(profile, &rejected, true).is_none());
    }
    assert!(route(profile, &p, false).is_none());

    let m5_without_mxu = DeviceProfile::new(DeviceIdentity::M5Max, GpuFamily::M5Plus, DeviceSize::Large, false);
    let p = problem(7, 6144, 5120, 8, 64, GemmBPrologueKind::ScaleSymmetricDequant);
    assert!(matches!(route(device_profile(DeviceIdentity::M5Max), &p, true), Some(QmvRoute::MainGemm(_))));
    assert_eq!(route(m5_without_mxu, &p, true), None);
}

#[uzu_test]
fn normal_routing_handles_inputs_outside_the_frozen_matrix() {
    let profile = device_profile(DeviceIdentity::M1);
    for (m, n, k) in [(1, 5120, 17408), (2, 4096, 5120)] {
        let problem = problem(m, n, k, 4, 64, GemmBPrologueKind::ScaleZeroPointDequant);
        assert_eq!(route(profile, &problem, true), None);
        let specialization =
            GemvSpecialization::select_shape(&problem, DataType::BF16, DataType::BF16, DataType::BF16, profile)
                .expect("normal M1 policy should select GEMV for this anchor");
        assert!(matches!(
            MatmulMetalKernel::choose_dispatch(
                &problem,
                profile,
                false,
                DataType::BF16,
                DataType::BF16,
                DataType::BF16,
            ),
            MatmulDispatch::Gemv(actual) if actual == specialization
        ));
    }
}
