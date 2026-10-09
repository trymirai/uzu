use metal::MTLGPUFamily;
use uzu_engine_macros::uzu_test;
use xxhash_rust::xxh3::xxh3_64;

use super::*;
use crate::{
    backends::{
        common::{
            gpu_types::gemm::{GemmBPrologueKind, GemmDTransform},
            kernel::matmul::{MatmulShape, QuantParamsLayout},
        },
        metal::kernel::matmul::{MatmulDispatch, MatmulMetalKernel, gemv::GemvSpecialization},
    },
    data_type::DataType,
};

const FROZEN_DISPATCH_FINGERPRINT: u64 = 13_760_538_547_558_457_563;

const DEVICES: [(&str, &str, u32, MTLGPUFamily, bool); 17] = [
    ("m1", "Apple M1", 8, MTLGPUFamily::Apple7, false),
    ("m1-pro", "Apple M1 Pro", 16, MTLGPUFamily::Apple7, false),
    ("m1-max", "Apple M1 Max", 32, MTLGPUFamily::Apple7, false),
    ("m1-ultra", "Apple M1 Ultra", 64, MTLGPUFamily::Apple7, false),
    ("m2", "Apple M2", 10, MTLGPUFamily::Apple8, false),
    ("m2-pro", "Apple M2 Pro", 19, MTLGPUFamily::Apple8, false),
    ("m2-max", "Apple M2 Max", 38, MTLGPUFamily::Apple8, false),
    ("m2-ultra", "Apple M2 Ultra", 76, MTLGPUFamily::Apple8, false),
    ("m3", "Apple M3", 10, MTLGPUFamily::Apple9, false),
    ("m3-pro", "Apple M3 Pro", 18, MTLGPUFamily::Apple9, false),
    ("m3-max", "Apple M3 Max", 40, MTLGPUFamily::Apple9, false),
    ("m4", "Apple M4", 10, MTLGPUFamily::Apple9, false),
    ("m4-pro", "Apple M4 Pro", 20, MTLGPUFamily::Apple9, false),
    ("m4-max", "Apple M4 Max", 40, MTLGPUFamily::Apple9, false),
    ("m5", "Apple M5", 10, MTLGPUFamily::Apple10, true),
    ("m5-pro", "Apple M5 Pro", 20, MTLGPUFamily::Apple10, true),
    ("m5-max", "Apple M5 Max", 40, MTLGPUFamily::Apple10, true),
];
const FORMATS: [(&str, u32, u32, GemmBPrologueKind); 2] = [
    ("W4/ZP G64", 4, 64, GemmBPrologueKind::ScaleZeroPointDequant),
    ("W8/Symmetric G64", 8, 64, GemmBPrologueKind::ScaleSymmetricDequant),
];
const SHAPES: [(&str, u32, u32); 6] = [
    ("down", 5120, 17408),
    ("gate", 6144, 5120),
    ("gate-up", 34816, 5120),
    ("projection-in", 16480, 5120),
    ("projection-out", 5120, 6144),
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
        b_is_trellis: false,
        b_bits: Some(bits),
        b_group_size: Some(group),
        signed_codes: false,
        a_full_precision: true,
        gathered: false,
        params_layout: Some(QuantParamsLayout::OutputGroup),
        d_transform: GemmDTransform::empty(),
    }
}

#[uzu_test]
fn dispatch_fingerprint_is_stable_and_rows_are_live() {
    let mut canonical = Vec::new();
    let mut matched_rows = vec![false; ROWS.len()];
    for &(device_label, device_name, gpu_core_count, apple_gpu_family, supports_mxu) in &DEVICES {
        for &(format_name, bits, group, prologue) in &FORMATS {
            for m in 2..=7 {
                for &(shape_name, n, k) in SHAPES.iter().filter(|shape| bits == 4 || shape.0 != "gate") {
                    let problem = problem(m, n, k, bits, group, prologue);
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
                    canonical.push(format!("{device_label}|{format_name}|{shape_name}|{m}|{n}|{k}|{runtime:?}"));
                    let mask = shape(n, k);
                    let rows: Vec<_> = ROWS
                        .iter()
                        .enumerate()
                        .filter(|(_, row)| {
                            row.device_name == device_name
                                && row.bits == bits
                                && row.group == group
                                && row.m == m
                                && row.shapes & mask != 0
                        })
                        .collect();
                    assert!(
                        rows.len() <= 1,
                        "duplicate route rows for {device_label} {format_name} M={m} {shape_name}"
                    );
                    let Some(&(row_index, row)) = rows.first() else {
                        continue;
                    };
                    matched_rows[row_index] = true;
                    assert_eq!(route(device_name, apple_gpu_family, &problem, true), Some(row.tile));
                    let specialization = GemvSpecialization::select_tile(
                        &problem,
                        DataType::BF16,
                        DataType::BF16,
                        DataType::BF16,
                        row.tile,
                    )
                    .expect("stored GEMV tile must be legal");
                    assert!(matches!(runtime, MatmulDispatch::Gemv(actual) if actual == specialization));
                }
            }
        }
    }
    assert!(matched_rows.into_iter().all(|matched| matched), "route table contains an orphaned row");
    canonical.sort();
    assert_eq!(canonical.len(), 1122);
    assert_eq!(xxh3_64(canonical.join("\n").as_bytes()), FROZEN_DISPATCH_FINGERPRINT);
}

#[uzu_test]
fn exact_lookup_rejects_non_matrix_inputs() {
    let p = problem(2, 5120, 17408, 4, 64, GemmBPrologueKind::ScaleZeroPointDequant);
    assert!(route("Apple M1", MTLGPUFamily::Apple7, &p, true).is_some());
    for mutate in [
        |p: &mut MatmulShape| p.m = 1,
        |p: &mut MatmulShape| p.n = 1,
        |p: &mut MatmulShape| p.b_bits = Some(8),
        |p: &mut MatmulShape| p.gathered = true,
    ] {
        let mut rejected = p;
        mutate(&mut rejected);
        assert!(route("Apple M1", MTLGPUFamily::Apple7, &rejected, true).is_none());
    }
    assert!(route("Apple M1", MTLGPUFamily::Apple7, &p, false).is_none());

    let mut rht = p;
    rht.d_transform = GemmDTransform::RHT;
    rht.signed_codes = true;
    let tile = route("Apple M1", MTLGPUFamily::Apple7, &rht, true).expect("RHT must preserve the exact route");
    let deferred = GemvSpecialization::select_tile(&rht, DataType::BF16, DataType::BF16, DataType::BF16, tile).unwrap();
    assert_eq!(deferred.output_row_tile(), 16);
    assert!(!deferred.fuses_rht());
    rht.n -= 1;
    assert!(GemvSpecialization::select_tile(&rht, DataType::BF16, DataType::BF16, DataType::BF16, tile).is_none());
}

#[uzu_test]
fn normal_routing_handles_inputs_outside_the_frozen_matrix() {
    for (m, n, k) in [(1, 5120, 17408), (2, 4096, 5120)] {
        let problem = problem(m, n, k, 4, 64, GemmBPrologueKind::ScaleZeroPointDequant);
        assert_eq!(route("Apple M1", MTLGPUFamily::Apple7, &problem, true), None);
        let specialization = GemvSpecialization::select_shape(
            &problem,
            DataType::BF16,
            DataType::BF16,
            DataType::BF16,
            8,
            MTLGPUFamily::Apple7,
        )
        .expect("normal M1 policy should select GEMV for this anchor");
        assert!(matches!(
            MatmulMetalKernel::choose_dispatch(
                &problem,
                "Apple M1",
                8,
                MTLGPUFamily::Apple7,
                false,
                DataType::BF16,
                DataType::BF16,
                DataType::BF16,
            ),
            MatmulDispatch::Gemv(actual) if actual == specialization
        ));
    }

    let mut fp = problem(1, 1024, 512, 4, 32, GemmBPrologueKind::ScaleZeroPointDequant);
    fp.b_prologue = GemmBPrologueKind::FullPrecision;
    fp.b_bits = None;
    fp.b_group_size = None;
    let mut with_rht = fp;
    with_rht.d_transform = GemmDTransform::RHT;
    for (cores, family) in [(8, MTLGPUFamily::Apple7), (40, MTLGPUFamily::Apple10)] {
        let plain =
            GemvSpecialization::select_shape(&fp, DataType::BF16, DataType::BF16, DataType::BF16, cores, family)
                .expect("generic M=1 FP GEMV tile");
        let rht =
            GemvSpecialization::select_shape(&with_rht, DataType::BF16, DataType::BF16, DataType::BF16, cores, family)
                .expect("generic M=1 FP RHT GEMV tile");
        assert!(plain.output_row_tile() < 32);
        assert_eq!(rht, plain);
    }
}

#[uzu_test]
fn family_lookup_requires_one_unanimous_route() {
    let m1_route = problem(4, 34816, 5120, 4, 64, GemmBPrologueKind::ScaleZeroPointDequant);
    let unanimous = problem(6, 5120, 17408, 4, 64, GemmBPrologueKind::ScaleZeroPointDequant);
    let families: [(&MatmulShape, MTLGPUFamily, &str, &[&str]); 4] = [
        (&m1_route, MTLGPUFamily::Apple7, "Apple M1", &["Apple M1 Pro", "Apple M1 Max", "Apple M1 Ultra"]),
        (&unanimous, MTLGPUFamily::Apple8, "Apple M2", &["Apple M2 Max", "Apple M2 Ultra"]),
        (&unanimous, MTLGPUFamily::Apple9, "Apple M3 Max", &["Apple M3", "Apple M3 Pro", "Apple M4 Max"]),
        (&unanimous, MTLGPUFamily::Apple10, "Apple M5 Max", &["Apple M5", "Apple M5 Pro"]),
    ];
    for (shape, family, measured_device, aliases) in families {
        let measured = route(measured_device, family, shape, true);
        for device in aliases {
            assert_eq!(route(device, family, shape, true), measured);
        }
    }

    let disagreement = problem(3, 5120, 6144, 4, 64, GemmBPrologueKind::ScaleZeroPointDequant);
    assert_eq!(route("Apple M2 Max", MTLGPUFamily::Apple8, &disagreement, true), None);
    assert!(route("Apple M2", MTLGPUFamily::Apple8, &disagreement, true).is_some());
    assert_eq!(route("Apple A11 GPU", MTLGPUFamily::Apple4, &m1_route, true), None);
}
