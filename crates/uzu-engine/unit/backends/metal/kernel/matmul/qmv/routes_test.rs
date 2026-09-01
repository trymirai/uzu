use std::fmt::Write;

use metal::MTLGPUFamily;
use uzu_engine_macros::uzu_test;
use xxhash_rust::xxh3::xxh3_64;

use super::*;
use crate::{
    backends::{
        common::{
            gpu_types::gemm::{GemmBPrologueKind, GemmDTransform},
            kernel::matmul::MatmulShape,
        },
        metal::kernel::matmul::{MatmulDispatch, MatmulMetalKernel, gemv::GemvSpecialization},
    },
    data_type::DataType,
};

const FROZEN_PLAN_FINGERPRINT: u64 = 5_839_212_743_558_880_136;
const APPLE7: Option<MTLGPUFamily> = Some(MTLGPUFamily::Apple7);
const APPLE8: Option<MTLGPUFamily> = Some(MTLGPUFamily::Apple8);
const APPLE9: Option<MTLGPUFamily> = Some(MTLGPUFamily::Apple9);
const APPLE10: Option<MTLGPUFamily> = Some(MTLGPUFamily::Apple10);

const DEVICES: [(&str, &str, u32, Option<MTLGPUFamily>, bool); 7] = [
    ("m1", "Apple M1", 8, APPLE7, false),
    ("m2", "Apple M2", 10, APPLE8, false),
    ("m2-pro", "Apple M2 Pro", 19, APPLE8, false),
    ("m3-max", "Apple M3 Max", 40, APPLE9, false),
    ("m4", "Apple M4", 10, APPLE9, false),
    ("m4-pro", "Apple M4 Pro", 20, APPLE9, false),
    ("m5-max", "Apple M5 Max", 40, APPLE10, true),
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
    for &(device_label, device_name, gpu_core_count, apple_gpu_family, supports_mxu) in &DEVICES {
        for &(format_name, bits, group, prologue) in &FORMATS {
            for m in 2..=7 {
                for &(shape_name, n, k) in &SHAPES {
                    let mask = shape(n, k);
                    let matches: Vec<_> = ROWS
                        .iter()
                        .enumerate()
                        .filter(|row| {
                            row.1.device_name == device_name
                                && row.1.apple_gpu_family == apple_gpu_family
                                && row.1.bits == bits
                                && row.1.group == group
                                && row.1.m == m
                                && row.1.shapes & mask != 0
                        })
                        .collect();
                    assert_eq!(matches.len(), 1, "route coverage for {device_label} {format_name} M={m} {shape_name}");
                    let (row_index, row) = matches[0];
                    matched_rows[row_index] = true;
                    let selected = row.route;
                    match selected {
                        QmvRoute::Tuned(_) => tuned += 1,
                        QmvRoute::MainGemv(_) => main_gemv += 1,
                        QmvRoute::MainGemm(_) => main_gemm += 1,
                    }
                    canonical.push(format!("{device_label}|{format_name}|{shape_name}|{m}|{n}|{k}|{selected:?}"));
                    let problem = problem(m, n, k, bits, group, prologue);
                    assert_eq!(route(device_name, apple_gpu_family, supports_mxu, &problem, true), Some(selected));
                    let runtime = MatmulMetalKernel::choose_dispatch(
                        &problem,
                        device_name,
                        gpu_core_count,
                        apple_gpu_family,
                        supports_mxu,
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
                        assert!(matches!(runtime, MatmulDispatch::Gemv(actual) if actual == specialization));
                    } else if let QmvRoute::MainGemm(plan) = selected {
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
        writeln!(&mut expanded, "{line}").expect("writing to a String must succeed");
    }
    assert_eq!(tuned, 864);
    assert_eq!(main_gemv, 221);
    assert_eq!(main_gemm, 91);
    assert_eq!(xxh3_64(expanded.trim_end().as_bytes()), FROZEN_PLAN_FINGERPRINT);
}

#[uzu_test]
fn exact_lookup_rejects_non_matrix_inputs() {
    let p = problem(2, 5120, 17408, 4, 64, GemmBPrologueKind::ScaleZeroPointDequant);
    assert!(route("Apple M1", APPLE7, false, &p, true).is_some());
    for mutate in [
        |p: &mut MatmulShape| p.m = 1,
        |p: &mut MatmulShape| p.n = 1,
        |p: &mut MatmulShape| p.b_bits = Some(8),
        |p: &mut MatmulShape| p.gathered = true,
    ] {
        let mut rejected = p;
        mutate(&mut rejected);
        assert!(route("Apple M1", APPLE7, false, &rejected, true).is_none());
    }
    assert!(route("Apple M1", APPLE7, false, &p, false).is_none());

    let mut rht = p;
    rht.d_transform = GemmDTransform::RHT;
    rht.signed_codes = true;
    let selected = route("Apple M1", APPLE7, false, &rht, true).expect("RHT must preserve the exact route");
    let QmvRoute::Tuned(tile) = selected else {
        panic!("test anchor must use a tuned tile");
    };
    assert!(GemvSpecialization::select_tile(&rht, DataType::BF16, DataType::BF16, DataType::BF16, tile).is_some());
    rht.n -= 1;
    assert!(GemvSpecialization::select_tile(&rht, DataType::BF16, DataType::BF16, DataType::BF16, tile).is_none());

    let p = problem(7, 6144, 5120, 8, 64, GemmBPrologueKind::ScaleSymmetricDequant);
    assert!(matches!(route("Apple M5 Max", APPLE10, true, &p, true), Some(QmvRoute::MainGemm(_))));
    assert_eq!(route("Apple M5 Max", APPLE10, false, &p, true), None);
}

#[uzu_test]
fn normal_routing_handles_inputs_outside_the_frozen_matrix() {
    for (m, n, k) in [(1, 5120, 17408), (2, 4096, 5120)] {
        let problem = problem(m, n, k, 4, 64, GemmBPrologueKind::ScaleZeroPointDequant);
        assert_eq!(route("Apple M1", APPLE7, false, &problem, true), None);
        let specialization =
            GemvSpecialization::select_shape(&problem, DataType::BF16, DataType::BF16, DataType::BF16, 8, APPLE7)
                .expect("normal M1 policy should select GEMV for this anchor");
        assert!(matches!(
            MatmulMetalKernel::choose_dispatch(
                &problem,
                "Apple M1",
                8,
                APPLE7,
                false,
                DataType::BF16,
                DataType::BF16,
                DataType::BF16,
            ),
            MatmulDispatch::Gemv(actual) if actual == specialization
        ));
    }
}

#[uzu_test]
fn family_lookup_requires_one_unanimous_route() {
    let m1_route = problem(4, 8192, 5120, 4, 64, GemmBPrologueKind::ScaleZeroPointDequant);
    let measured_m1 = route("Apple M1", APPLE7, false, &m1_route, true);
    for device_name in ["Apple M1 Pro", "Apple M1 Max", "Apple M1 Ultra"] {
        assert_eq!(route(device_name, APPLE7, false, &m1_route, true), measured_m1);
    }

    let unanimous = problem(6, 5120, 17408, 4, 64, GemmBPrologueKind::ScaleZeroPointDequant);
    let measured_m2 = route("Apple M2", APPLE8, false, &unanimous, true);
    for device_name in ["Apple M2 Max", "Apple M2 Ultra"] {
        assert_eq!(route(device_name, APPLE8, false, &unanimous, true), measured_m2);
    }
    let measured_m3_max = route("Apple M3 Max", APPLE9, false, &unanimous, true);
    for device_name in ["Apple M3", "Apple M3 Pro", "Apple M4 Max"] {
        assert_eq!(route(device_name, APPLE9, false, &unanimous, true), measured_m3_max);
    }
    let measured_m5_max = route("Apple M5 Max", APPLE10, true, &unanimous, true);
    for device_name in ["Apple M5", "Apple M5 Pro"] {
        assert_eq!(route(device_name, APPLE10, true, &unanimous, true), measured_m5_max);
    }

    let disagreement = problem(3, 5120, 6144, 4, 64, GemmBPrologueKind::ScaleZeroPointDequant);
    assert_eq!(route("Apple M2 Max", APPLE8, false, &disagreement, true), None);
    assert!(route("Apple M2", APPLE8, false, &disagreement, true).is_some());
    assert_eq!(route("External GPU", None, false, &m1_route, true), None);
}
