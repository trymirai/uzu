use std::ops::RangeInclusive;

use ash::vk;
use uzu_engine_macros::uzu_test;

use super::*;
use crate::backends::vulkan::VkPhysicalDeviceSubgroupProperties;

const DATA_TYPES: [DataType; 3] = [DataType::BF16; 3];
const ROWS: [u32; 6] = [1, 2, 8, 16, 64, 1024];

/// The facts the planner reads from a device's report: `shared` bytes of workgroup memory, the workgroup count and size
/// limits, the default subgroup size, the sizes compute pipelines may require and the most required subgroups a
/// workgroup, the unit count, and the largest buffer and allocation. Leaked, so the fixtures outlive the borrowing facts.
fn facts(
    capabilities: Capability,
    shared: u32,
    (count, size): ([u32; 3], [u32; 3]),
    (subgroup, requirable, workgroup_subgroups): (u32, RangeInclusive<u32>, u32),
    compute_units: Option<u32>,
    [max_buffer_size, max_memory_allocation_size]: [u64; 2],
) -> VkDeviceFacts<'static> {
    let limits = vk::PhysicalDeviceLimits {
        max_compute_shared_memory_size: shared,
        max_compute_work_group_count: count,
        max_compute_work_group_size: size,
        max_compute_work_group_invocations: 1024,
        ..Default::default()
    };
    let subgroups = VkPhysicalDeviceSubgroupProperties {
        size: subgroup,
        max_compute_workgroup_subgroups: workgroup_subgroups,
        supported_operations: vk::SubgroupFeatureFlags::empty(),
        requirable_sizes: Some(requirable),
    };
    VkDeviceFacts {
        capabilities,
        limits: Box::leak(Box::new(limits)),
        subgroups: Box::leak(Box::new(subgroups)),
        compute_units,
        max_buffer_size,
        max_memory_allocation_size,
    }
}

/// The Radeon 890M on AMD's Windows driver, from vulkan.gpuinfo.org report 50864: 32 KB of shared memory, 16 units,
/// subgroups of 64 by default and 32 to 64 requirable, signed dot products, 2 GiB buffers and no BF16 cooperative
/// matrices. The report omits the cooperative-matrix shapes; RDNA 3.5's 16 x 16 x 16 FP16 and INT8 ones are assumed.
fn radeon_890m() -> VkDeviceFacts<'static> {
    let capabilities = Capability::FLOAT_CONTROLS
        | Capability::SUBGROUP_SIZE_32
        | Capability::MEMORY_BUDGET
        | Capability::BF16_TYPE
        | Capability::CALIBRATED_TIMESTAMPS
        | Capability::FP16_TENSOR_CORES
        | Capability::INT8_TENSOR_CORES_K16;
    let workgroups = ([u32::MAX, 65535, 65535], [1024; 3]);
    facts(capabilities, 32768, workgroups, (64, 32..=64, u32::MAX), Some(16), [1 << 31; 2])
}

/// The RTX 4090 on NVIDIA's Windows driver 617.42.
fn rtx_4090() -> VkDeviceFacts<'static> {
    let capabilities = Capability::FLOAT_CONTROLS
        | Capability::SUBGROUP_SIZE_32
        | Capability::MEMORY_BUDGET
        | Capability::BF16_TYPE
        | Capability::CALIBRATED_TIMESTAMPS
        | Capability::BF16_TENSOR_CORES
        | Capability::FP16_TENSOR_CORES
        | Capability::INT8_TENSOR_CORES_K32
        | Capability::INT8_ELEMENT_FOLD;
    let workgroups = ([i32::MAX as u32, 65535, 65535], [1024, 1024, 64]);
    facts(capabilities, 49152, workgroups, (32, 32..=32, 32), Some(128), [1 << 40, u64::MAX])
}

/// The Pixel 10 Pro XL's PowerVR DXT-48-1536 on driver 25.3.
fn pixel_10() -> VkDeviceFacts<'static> {
    let capabilities = Capability::FLOAT_CONTROLS
        | Capability::SUBGROUP_SIZE_32
        | Capability::CALIBRATED_TIMESTAMPS
        | Capability::FP16_TENSOR_CORES
        | Capability::INT8_TENSOR_CORES_K16
        | Capability::UNSIGNED_DOT_PRODUCT;
    let workgroups = ([65536; 3], [1024, 1024, 64]);
    facts(capabilities, 32768, workgroups, (128, 32..=128, 32), None, [u32::MAX.into(), 1 << 33])
}

/// Qwen3.5 9B-M's linears, n by k: the MLP's fused up and gate and its down projection with an output transform, and
/// the tied readout, whose metadata is output-major. B holds U4 codes with zero points in groups of 32.
fn linears() -> [MatmulShape; 3] {
    let linear = |n, k, params_layout, d_transform| MatmulShape {
        m: 1,
        n,
        k,
        b_transpose: true,
        b_leading_dimension: None,
        b_prologue: GemmBPrologueKind::ScaleZeroPointDequant,
        b_is_trellis: false,
        b_bits: Some(4),
        b_group_size: Some(32),
        signed_codes: false,
        a_full_precision: true,
        gathered: false,
        params_layout: Some(params_layout),
        d_transform,
    };
    [
        linear(24576, 4096, QuantParamsLayout::GroupOutput, GemmDTransform::RHT),
        linear(4096, 12288, QuantParamsLayout::GroupOutput, GemmDTransform::RHT),
        linear(248320, 4096, QuantParamsLayout::OutputGroup, GemmDTransform::empty()),
    ]
}

/// The baseline for `rows` rows of `linear` as the encoder routes them: the loaded weights' layout, then the rows'
/// activation format.
fn routed(
    device: VkDeviceFacts<'_>,
    linear: MatmulShape,
    rows: u32,
) -> (MatmulShape, GemmBLayout, VkMatmulPlan) {
    let capabilities = device.capabilities;
    let a8 = MatmulShape {
        a_full_precision: false,
        ..linear
    };
    let b_layout = match VkMatmulPlan::tiles_weights(&a8, DATA_TYPES, capabilities) {
        true => GemmBLayout::Tiled,
        false => GemmBLayout::RowMajor,
    };
    let bf16 = MatmulShape {
        m: rows,
        ..linear
    };
    let shape = MatmulShape {
        a_full_precision: VkMatmulPlan::activation_format(&bf16, b_layout, DATA_TYPES, capabilities)
            == ActivationFormat::Bf16,
        ..bf16
    };
    let eligible = capabilities.float_tensor_cores(DataType::BF16);
    (shape, b_layout, VkMatmulPlan::baseline(&shape, b_layout, DATA_TYPES, device, eligible))
}

#[uzu_test]
fn routed_baselines_pass_their_checks() {
    for device in [radeon_890m(), rtx_4090(), pixel_10()] {
        for (linear, rows) in itertools::iproduct!(linears(), ROWS) {
            let (shape, b_layout, plan) = routed(device, linear, rows);
            let checked = plan.check(&shape, b_layout, DATA_TYPES, device);
            assert_eq!(checked, Ok(()), "{plan:?} for {rows} x {} x {}", shape.n, shape.k);
        }
    }
}

#[uzu_test]
fn decode_and_prefill_take_each_device_s_paths() {
    let [up, ..] = linears();
    let plan = |device, rows| routed(device, up, rows).2;
    // Signed-dot devices decode full-precision A; unsigned-dot ones take A8 throughout, 8 rows on 128-lane subgroups.
    assert!(matches!(plan(radeon_890m(), 1), VkMatmulPlan::Gemv(_)));
    assert!(matches!(plan(rtx_4090(), 1), VkMatmulPlan::Gemv(_)));
    assert!(matches!(
        plan(pixel_10(), 1),
        VkMatmulPlan::GemvInt8 {
            rows: 8,
            ..
        }
    ));
    // The 64-row block's staging exceeds 32 KB.
    assert_eq!(
        plan(radeon_890m(), 1024),
        VkMatmulPlan::GemmInt8 {
            matrices: Capability::INT8_TENSOR_CORES_K16,
            row_block: 32,
            simdgroups: 4,
            stage_depth: 64,
            splits: 1,
        }
    );
    assert!(matches!(plan(rtx_4090(), 1024), VkMatmulPlan::GemmInt8 { matrices, row_block: 64, .. }
        if matrices.contains(Capability::INT8_ELEMENT_FOLD)));
    assert!(matches!(
        plan(pixel_10(), 1024),
        VkMatmulPlan::GemvInt8 {
            rows: 8,
            ..
        }
    ));
}
