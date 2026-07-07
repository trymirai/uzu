#![cfg(metal_backend)]

use half::bf16;
use proc_macros::uzu_test;

use super::tree_update_solve_test::{CASES, run_case};

// MXU needs an even number of column fragments or MPP row pairing; BV16 (a single
// 16x16 fragment) is not supported.
const MXU_BVS: &[u32] = &[32];
use crate::{
    backends::{
        common::Context,
        cpu::Cpu,
        metal::{Metal, MetalContext},
    },
    data_type::DataType,
    tests::assert::assert_eq_float,
};

#[derive(Clone, Copy)]
struct KernelPath {
    name: &'static str,
    use_mxu: bool,
}

fn bf16_kernel_paths(context: &MetalContext) -> Vec<KernelPath> {
    let mut paths = vec![KernelPath {
        name: "Simdgroup",
        use_mxu: false,
    }];
    if context.supports_mxu() {
        paths.push(KernelPath {
            name: "MXU",
            use_mxu: true,
        });
    }
    paths
}

#[uzu_test]
fn test_tree_update_solve_bf16_dispatch_paths() {
    let context = MetalContext::new().expect("context");
    let paths = bf16_kernel_paths(&context);
    for &bv in MXU_BVS {
        for &use_h0 in &[true, false] {
            for &case in CASES {
                let expected = run_case::<Cpu, bf16>(case, bv, false, use_h0, DataType::BF16, bf16::from_f32);
                for path in &paths {
                    let output =
                        run_case::<Metal, bf16>(case, bv, path.use_mxu, use_h0, DataType::BF16, bf16::from_f32);
                    assert_eq_float(
                        &expected,
                        &output,
                        1e-2,
                        &format!("{} {} BV{bv} h0={use_h0}", case.name, path.name),
                    );
                }
            }
        }
    }
}
