#![cfg(metal_backend)]

use std::fmt::Display;

use backend_uzu::{ArrayElement, backends::{common::Context, metal::MetalContext}};
use half::{bf16, f16};
use metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLCompileOptions, MTLComputeCommandEncoder,
    MTLComputePipelineState, MTLDevice, MTLDeviceExt, MTLResourceOptions, MTLSize, MLTLanguageVersion, MTLLibrary,
    MTLLibraryExt,
};
use num_traits::Float;
use objc2::runtime::ProtocolObject;

use crate::{common::assert::assert_eq_float, uzu_test};

const MATRIX_DIMENSION: usize = 32;
const ELEMENT_COUNT: usize = MATRIX_DIMENSION * MATRIX_DIMENSION;

const METAL_SOURCE: &str = r#"
#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace metal;
using namespace mpp::tensor_ops;

kernel void minimal_mpp_device(
    device half* left_matrix [[buffer(0)]],
    device half* right_matrix [[buffer(1)]],
    device half* output_matrix [[buffer(2)]]
) {
    constexpr auto descriptor = matmul2d_descriptor(32, 32);
    matmul2d<descriptor, execution_simdgroup> op;

    auto left_tensor = tensor(left_matrix, dextents<int, 2>{32, 32}, array<int, 2>{1, 32});
    auto right_tensor = tensor(right_matrix, dextents<int, 2>{32, 32}, array<int, 2>{1, 32});
    auto output_tensor = tensor(output_matrix, dextents<int, 2>{32, 32}, array<int, 2>{1, 32});
    auto accumulator = op.get_destination_cooperative_tensor<decltype(left_tensor), decltype(right_tensor), half>();

    #pragma unroll
    for (uint16_t index = 0; index < accumulator.get_capacity(); ++index) {
        if (accumulator.is_valid_element(index)) {
            accumulator[index] = half(0);
        }
    }

    op.run(left_tensor, right_tensor, accumulator);
    accumulator.store(output_tensor);
}

kernel void minimal_mpp_threadgroup(
    device half* left_matrix [[buffer(0)]],
    device half* right_matrix [[buffer(1)]],
    device half* output_matrix [[buffer(2)]],
    threadgroup half* left_shared [[threadgroup(0)]],
    threadgroup half* right_shared [[threadgroup(1)]],
    uint lane_index [[thread_index_in_threadgroup]]
) {
    if (lane_index < 32) {
        const uint row = lane_index;
        for (uint col = 0; col < 32; ++col) {
            left_shared[row * 32 + col] = left_matrix[row * 32 + col];
            right_shared[row * 32 + col] = right_matrix[row * 32 + col];
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    constexpr auto descriptor = matmul2d_descriptor(32, 32);
    matmul2d<descriptor, execution_simdgroup> op;

    auto left_tensor = tensor(left_shared, dextents<int, 2>{32, 32}, array<int, 2>{1, 32});
    auto right_tensor = tensor(right_shared, dextents<int, 2>{32, 32}, array<int, 2>{1, 32});
    auto output_tensor = tensor(output_matrix, dextents<int, 2>{32, 32}, array<int, 2>{1, 32});
    auto accumulator = op.get_destination_cooperative_tensor<decltype(left_tensor), decltype(right_tensor), half>();

    #pragma unroll
    for (uint16_t index = 0; index < accumulator.get_capacity(); ++index) {
        if (accumulator.is_valid_element(index)) {
            accumulator[index] = half(0);
        }
    }

    op.run(left_tensor, right_tensor, accumulator);
    accumulator.store(output_tensor);
}
"#;

fn create_input_matrices() -> (Vec<f16>, Vec<f16>) {
    let left = (0..ELEMENT_COUNT)
        .map(|index| f16::from_f32(((index % 13) as f32) * 0.125 - 0.75))
        .collect();
    let right = (0..ELEMENT_COUNT)
        .map(|index| f16::from_f32(((index % 17) as f32) * 0.1 - 0.5))
        .collect();
    (left, right)
}

fn compute_reference_output(
    left: &[f16],
    right: &[f16],
) -> Vec<f16> {
    (0..MATRIX_DIMENSION)
        .flat_map(|row| {
            (0..MATRIX_DIMENSION).map(move |col| {
                let accumulator = (0..MATRIX_DIMENSION)
                    .map(|inner| {
                        let left_value = left[row * MATRIX_DIMENSION + inner].to_f32();
                        let right_value = right[inner * MATRIX_DIMENSION + col].to_f32();
                        left_value * right_value
                    })
                    .sum::<f32>();
                f16::from_f32(accumulator)
            })
        })
        .collect()
}

fn compile_library(device: &ProtocolObject<dyn MTLDevice>) -> RetainedLibrary {
    let compile_options = MTLCompileOptions::new();
    compile_options.set_language_version(MLTLanguageVersion::Version4_0);
    device
        .new_library_with_source(METAL_SOURCE, Some(&compile_options))
        .unwrap_or_else(|error| panic!("failed to compile Metal source: {}", error.localizedDescription()))
}

type RetainedLibrary = objc2::rc::Retained<ProtocolObject<dyn MTLLibrary>>;
type RetainedPipeline = objc2::rc::Retained<ProtocolObject<dyn MTLComputePipelineState>>;

fn compile_pipeline(
    library: &ProtocolObject<dyn MTLLibrary>,
    function_name: &str,
) -> RetainedPipeline {
    let function = library
        .new_function_with_name(function_name)
        .unwrap_or_else(|| panic!("missing Metal function `{function_name}`"));
    library
        .device()
        .new_compute_pipeline_state_with_function(&function)
        .unwrap_or_else(|error| panic!("failed to create compute pipeline: {}", error.localizedDescription()))
}

fn run_kernel(
    function_name: &str,
    use_threadgroup_operands: bool,
) -> Vec<f16> {
    let context = MetalContext::new().expect("Metal context");
    let device = &context.device;
    let command_queue = &context.command_queue;

    let library = compile_library(device);
    let pipeline = compile_pipeline(&library, function_name);
    let (left, right) = create_input_matrices();

    let left_buffer = device
        .new_buffer_with_data(bytemuck::cast_slice(&left), MTLResourceOptions::STORAGE_MODE_SHARED)
        .expect("left buffer");
    let right_buffer = device
        .new_buffer_with_data(bytemuck::cast_slice(&right), MTLResourceOptions::STORAGE_MODE_SHARED)
        .expect("right buffer");
    let output_buffer = device
        .new_buffer(ELEMENT_COUNT * std::mem::size_of::<f16>(), MTLResourceOptions::STORAGE_MODE_SHARED)
        .expect("output buffer");

    let command_buffer = command_queue.command_buffer().expect("command buffer");
    let compute_encoder = command_buffer.compute_command_encoder().expect("compute encoder");
    compute_encoder.set_compute_pipeline_state(&pipeline);
    compute_encoder.set_buffer(Some(&left_buffer), 0, 0);
    compute_encoder.set_buffer(Some(&right_buffer), 0, 1);
    compute_encoder.set_buffer(Some(&output_buffer), 0, 2);

    if use_threadgroup_operands {
        compute_encoder.set_threadgroup_memory_length(ELEMENT_COUNT * std::mem::size_of::<f16>(), 0);
        compute_encoder.set_threadgroup_memory_length(ELEMENT_COUNT * std::mem::size_of::<f16>(), 1);
    }

    compute_encoder.dispatch_threadgroups(MTLSize::new(1, 1, 1), MTLSize::new(32, 1, 1));
    compute_encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();

    if let Some(error) = command_buffer.error() {
        panic!("command buffer failed: {}", error.localizedDescription());
    }

    unsafe {
        std::slice::from_raw_parts(output_buffer.contents().as_ptr() as *const f16, ELEMENT_COUNT).to_vec()
    }
}

#[uzu_test]
fn test_mpp_minimal_device_half_to_half() {
    let (left, right) = create_input_matrices();
    let expected = compute_reference_output(&left, &right);
    let actual = run_kernel("minimal_mpp_device", false);
    assert_eq_float(&expected, &actual, 0.05, "minimal MPP device GEMM");
}

#[uzu_test]
fn test_mpp_minimal_threadgroup_half_to_half() {
    let (left, right) = create_input_matrices();
    let expected = compute_reference_output(&left, &right);
    let actual = run_kernel("minimal_mpp_threadgroup", true);
    assert_eq_float(&expected, &actual, 0.05, "minimal MPP threadgroup GEMM");
}

trait ProbeScalar: Copy + Display + Float {
    const METAL_NAME: &'static str;
}

impl ProbeScalar for f16 {
    const METAL_NAME: &'static str = "half";
}

impl ProbeScalar for bf16 {
    const METAL_NAME: &'static str = "bfloat";
}

impl ProbeScalar for f32 {
    const METAL_NAME: &'static str = "float";
}

#[derive(Clone, Copy)]
enum ProbeScope {
    Simdgroup,
    Simdgroups4,
}

impl ProbeScope {
    fn metal_name(self) -> &'static str {
        match self {
            Self::Simdgroup => "execution_simdgroup",
            Self::Simdgroups4 => "execution_simdgroups<4>",
        }
    }

    fn thread_count(self) -> usize {
        match self {
            Self::Simdgroup => 32,
            Self::Simdgroups4 => 128,
        }
    }
}

#[derive(Clone, Copy)]
enum ProbeStorage {
    Device,
    Threadgroup,
}

#[derive(Clone, Copy)]
enum ProbeSliceMode {
    DirectTensor,
    DynamicOriginSlice,
    StaticOriginSlice,
}

#[derive(Clone, Copy)]
struct ProbeKLoop {
    tile_k: usize,
}

#[derive(Clone, Copy)]
struct ProbeRunConfig {
    problem_m: usize,
    problem_n: usize,
    problem_k: usize,
    descriptor_m: usize,
    descriptor_n: usize,
    transpose_right: bool,
    storage: ProbeStorage,
    scope: ProbeScope,
    slice_mode: ProbeSliceMode,
    k_loop: Option<ProbeKLoop>,
}

fn create_probe_input_matrices<T: ProbeScalar>(
    m: usize,
    k: usize,
    n: usize,
    transpose_right: bool,
) -> (Vec<T>, Vec<T>) {
    let left = (0..m * k)
        .map(|index| T::from(((index % 13) as f32) * 0.125 - 0.75).unwrap())
        .collect();
    let right = if transpose_right {
        (0..n * k)
            .map(|index| T::from(((index % 17) as f32) * 0.1 - 0.5).unwrap())
            .collect()
    } else {
        (0..k * n)
            .map(|index| T::from(((index % 17) as f32) * 0.1 - 0.5).unwrap())
            .collect()
    };
    (left, right)
}

fn compute_probe_reference_output<TInput: ProbeScalar, TOutput: ProbeScalar>(
    left: &[TInput],
    right: &[TInput],
    config: ProbeRunConfig,
) -> Vec<TOutput> {
    (0..config.problem_m)
        .flat_map(|row| {
            (0..config.problem_n).map(move |col| {
                let accumulator = (0..config.problem_k)
                    .map(|inner| {
                        let left_value = left[row * config.problem_k + inner].to_f32().unwrap();
                        let right_index = if config.transpose_right {
                            col * config.problem_k + inner
                        } else {
                            inner * config.problem_n + col
                        };
                        let right_value = right[right_index].to_f32().unwrap();
                        left_value * right_value
                    })
                    .sum::<f32>();
                TOutput::from(accumulator).unwrap()
            })
        })
        .collect()
}

fn tensor_extents_literal(config: ProbeRunConfig) -> (String, String, String, usize, usize) {
    let left_extents = format!("dextents<int, 2>{{{}, {}}}", config.problem_k, config.problem_m);
    let output_extents = format!("dextents<int, 2>{{{}, {}}}", config.problem_n, config.problem_m);

    if config.transpose_right {
        (
            left_extents,
            format!("dextents<int, 2>{{{}, {}}}", config.problem_k, config.problem_n),
            output_extents,
            config.problem_k,
            config.problem_n,
        )
    } else {
        (
            left_extents,
            format!("dextents<int, 2>{{{}, {}}}", config.problem_n, config.problem_k),
            output_extents,
            config.problem_n,
            config.problem_n,
        )
    }
}

fn storage_access_expressions(
    config: ProbeRunConfig,
) -> (String, String, String, String, String) {
    let (left_base_ptr, right_base_ptr) = match config.storage {
        ProbeStorage::Device => ("left_matrix", "right_matrix"),
        ProbeStorage::Threadgroup => ("left_shared", "right_shared"),
    };

    let (left_extents, right_extents, output_extents, right_stride, output_stride) =
        tensor_extents_literal(config);
    let left_stride = config.problem_k;

    let base_tensor_setup = format!(
        r#"
    auto left_tensor_base = tensor({left_base_ptr}, {left_extents}, array<int, 2>{{1, {left_stride}}});
    auto right_tensor_base = tensor({right_base_ptr}, {right_extents}, array<int, 2>{{1, {right_stride}}});
    auto output_tensor_base = tensor(output_matrix, {output_extents}, array<int, 2>{{1, {output_stride}}});
"#,
    );

    let (left_expr, right_expr, output_expr) = match config.slice_mode {
        ProbeSliceMode::DirectTensor => (
            "left_tensor_base".to_owned(),
            "right_tensor_base".to_owned(),
            "output_tensor_base".to_owned(),
        ),
        ProbeSliceMode::DynamicOriginSlice => (
            "left_tensor_base.slice(0, 0)".to_owned(),
            "right_tensor_base.slice(0, 0)".to_owned(),
            "output_tensor_base.slice(0, 0)".to_owned(),
        ),
        ProbeSliceMode::StaticOriginSlice => (
            format!(
                "left_tensor_base.template slice<{}, {}>(0, 0)",
                config.problem_k, config.problem_m
            ),
            if config.transpose_right {
                format!(
                    "right_tensor_base.template slice<{}, {}>(0, 0)",
                    config.problem_k, config.problem_n
                )
            } else {
                format!(
                    "right_tensor_base.template slice<{}, {}>(0, 0)",
                    config.problem_n, config.problem_k
                )
            },
            format!(
                "output_tensor_base.template slice<{}, {}>(0, 0)",
                config.problem_n, config.problem_m
            ),
        ),
    };

    (base_tensor_setup, left_expr, right_expr, output_expr, output_stride.to_string())
}

fn build_probe_source<TInput: ProbeScalar, TOutput: ProbeScalar>(
    config: ProbeRunConfig,
    use_raw_destination: bool,
) -> String {
    let thread_index_argument = "uint thread_index [[thread_index_in_threadgroup]]";
    let threadgroup_parameters = if matches!(config.storage, ProbeStorage::Threadgroup) {
        format!(
            ",\n    threadgroup {input_type}* left_shared [[threadgroup(0)]],\n    threadgroup {input_type}* right_shared [[threadgroup(1)]],\n    {thread_index_argument}",
            input_type = TInput::METAL_NAME,
        )
    } else {
        format!(",\n    {thread_index_argument}")
    };

    let preload_code = if matches!(config.storage, ProbeStorage::Threadgroup) {
        let left_elements = config.problem_m * config.problem_k;
        let right_elements = if config.transpose_right {
            config.problem_n * config.problem_k
        } else {
            config.problem_k * config.problem_n
        };
        format!(
            r#"
    for (uint index = thread_index; index < {left_elements}; index += {threads_per_threadgroup}) {{
        left_shared[index] = left_matrix[index];
    }}
    for (uint index = thread_index; index < {right_elements}; index += {threads_per_threadgroup}) {{
        right_shared[index] = right_matrix[index];
    }}

    threadgroup_barrier(mem_flags::mem_threadgroup);
"#,
            threads_per_threadgroup = config.scope.thread_count(),
        )
    } else {
        String::new()
    };

    let (tensor_setup, left_expr, right_expr, output_expr, _) = storage_access_expressions(config);
    let descriptor = match config.k_loop {
        Some(k_loop) => format!(
            "matmul2d_descriptor({}, {}, {}, false, {}, false, mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate)",
            config.descriptor_m, config.descriptor_n, k_loop.tile_k, if config.transpose_right { "true" } else { "false" }
        ),
        None => format!(
            "matmul2d_descriptor({}, {}, static_cast<int>(dynamic_extent), false, {}, false)",
            config.descriptor_m, config.descriptor_n, if config.transpose_right { "true" } else { "false" }
        ),
    };

    let body = if use_raw_destination {
        format!(
            r#"
    auto left_tensor = {left_expr};
    auto right_tensor = {right_expr};
    auto output_tensor = {output_expr};
    op.run(left_tensor, right_tensor, output_tensor);
"#,
        )
    } else if let Some(k_loop) = config.k_loop {
        let right_slice = if config.transpose_right {
            format!(
                "right_tensor_base.template slice<{tile_k}, {descriptor_n}>(inner_k, 0)",
                tile_k = k_loop.tile_k,
                descriptor_n = config.descriptor_n,
            )
        } else {
            format!(
                "right_tensor_base.template slice<{descriptor_n}, {tile_k}>(0, inner_k)",
                tile_k = k_loop.tile_k,
                descriptor_n = config.descriptor_n,
            )
        };
        let right_tail = if config.transpose_right {
            format!(
                "right_tensor_base.template slice<dynamic_extent, {descriptor_n}>(inner_k, 0)",
                descriptor_n = config.descriptor_n,
            )
        } else {
            format!(
                "right_tensor_base.template slice<{descriptor_n}, dynamic_extent>(0, inner_k)",
                descriptor_n = config.descriptor_n,
            )
        };

        format!(
            r#"
    auto output_tensor = output_tensor_base;
    auto accumulator = op.get_destination_cooperative_tensor<decltype(left_tensor_base), decltype(right_tensor_base), {output_type}>();

    #pragma unroll
    for (uint16_t index = 0; index < accumulator.get_capacity(); ++index) {{
        if (accumulator.is_valid_element(index)) {{
            accumulator[index] = {output_type}(0);
        }}
    }}

    int inner_k = 0;
    for (; inner_k + {tile_k} <= {problem_k}; inner_k += {tile_k}) {{
        auto left_tensor = left_tensor_base.template slice<{tile_k}, {descriptor_m}>(inner_k, 0);
        auto right_tensor = {right_slice};
        op.run(left_tensor, right_tensor, accumulator);
    }}

    if (inner_k < {problem_k}) {{
        auto left_tensor = left_tensor_base.template slice<dynamic_extent, {descriptor_m}>(inner_k, 0);
        auto right_tensor = {right_tail};
        op.run(left_tensor, right_tensor, accumulator);
    }}

    accumulator.store(output_tensor);
"#,
            output_type = TOutput::METAL_NAME,
            tile_k = k_loop.tile_k,
            problem_k = config.problem_k,
            descriptor_m = config.descriptor_m,
            right_slice = right_slice,
            right_tail = right_tail,
        )
    } else {
        format!(
            r#"
    auto left_tensor = {left_expr};
    auto right_tensor = {right_expr};
    auto output_tensor = {output_expr};
    auto accumulator = op.get_destination_cooperative_tensor<decltype(left_tensor), decltype(right_tensor), {output_type}>();

    #pragma unroll
    for (uint16_t index = 0; index < accumulator.get_capacity(); ++index) {{
        if (accumulator.is_valid_element(index)) {{
            accumulator[index] = {output_type}(0);
        }}
    }}

    op.run(left_tensor, right_tensor, accumulator);
    accumulator.store(output_tensor);
"#,
            output_type = TOutput::METAL_NAME,
        )
    };

    format!(
        r#"
#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace metal;
using namespace mpp::tensor_ops;

kernel void probe_kernel(
    device {input_type}* left_matrix [[buffer(0)]],
    device {input_type}* right_matrix [[buffer(1)]],
    device {output_type}* output_matrix [[buffer(2)]]{threadgroup_parameters}
) {{
{preload_code}
    constexpr auto descriptor = {descriptor};
    matmul2d<descriptor, {scope}> op;
{tensor_setup}
{body}
}}
"#,
        input_type = TInput::METAL_NAME,
        output_type = TOutput::METAL_NAME,
        threadgroup_parameters = threadgroup_parameters,
        preload_code = preload_code,
        descriptor = descriptor,
        scope = config.scope.metal_name(),
        tensor_setup = tensor_setup,
        body = body,
    )
}

fn try_compile_library(
    device: &ProtocolObject<dyn MTLDevice>,
    source: &str,
) -> Result<RetainedLibrary, String> {
    let compile_options = MTLCompileOptions::new();
    compile_options.set_language_version(MLTLanguageVersion::Version4_0);
    device
        .new_library_with_source(source, Some(&compile_options))
        .map_err(|error| error.localizedDescription().to_string())
}

fn slice_as_bytes<T>(
    slice: &[T],
) -> &[u8] {
    unsafe { std::slice::from_raw_parts(slice.as_ptr() as *const u8, std::mem::size_of_val(slice)) }
}

fn run_probe_kernel<TInput: ProbeScalar, TOutput: ProbeScalar>(
    config: ProbeRunConfig,
    use_raw_destination: bool,
) -> Vec<TOutput> {
    let context = MetalContext::new().expect("Metal context");
    let device = &context.device;
    let command_queue = &context.command_queue;

    let source = build_probe_source::<TInput, TOutput>(config, use_raw_destination);
    let library = try_compile_library(device, &source)
        .unwrap_or_else(|error| panic!("failed to compile probe Metal source: {error}"));
    let pipeline = compile_pipeline(&library, "probe_kernel");
    let (left, right) = create_probe_input_matrices::<TInput>(
        config.problem_m,
        config.problem_k,
        config.problem_n,
        config.transpose_right,
    );

    let left_buffer = device
        .new_buffer_with_data(slice_as_bytes(&left), MTLResourceOptions::STORAGE_MODE_SHARED)
        .expect("left buffer");
    let right_buffer = device
        .new_buffer_with_data(slice_as_bytes(&right), MTLResourceOptions::STORAGE_MODE_SHARED)
        .expect("right buffer");
    let output_buffer = device
        .new_buffer(
            config.problem_m * config.problem_n * std::mem::size_of::<TOutput>(),
            MTLResourceOptions::STORAGE_MODE_SHARED,
        )
        .expect("output buffer");
    unsafe {
        std::ptr::write_bytes(
            output_buffer.contents().as_ptr(),
            0,
            config.problem_m * config.problem_n * std::mem::size_of::<TOutput>(),
        );
    }

    let command_buffer = command_queue.command_buffer().expect("command buffer");
    let compute_encoder = command_buffer.compute_command_encoder().expect("compute encoder");
    compute_encoder.set_compute_pipeline_state(&pipeline);
    compute_encoder.set_buffer(Some(&left_buffer), 0, 0);
    compute_encoder.set_buffer(Some(&right_buffer), 0, 1);
    compute_encoder.set_buffer(Some(&output_buffer), 0, 2);

    if matches!(config.storage, ProbeStorage::Threadgroup) {
        compute_encoder.set_threadgroup_memory_length(
            config.problem_m * config.problem_k * std::mem::size_of::<TInput>(),
            0,
        );
        compute_encoder.set_threadgroup_memory_length(
            if config.transpose_right {
                config.problem_n * config.problem_k * std::mem::size_of::<TInput>()
            } else {
                config.problem_k * config.problem_n * std::mem::size_of::<TInput>()
            },
            1,
        );
    }

    compute_encoder.dispatch_threadgroups(
        MTLSize::new(1, 1, 1),
        MTLSize::new(config.scope.thread_count(), 1, 1),
    );
    compute_encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();

    if let Some(error) = command_buffer.error() {
        panic!("probe command buffer failed: {}", error.localizedDescription());
    }

    unsafe {
        std::slice::from_raw_parts(
            output_buffer.contents().as_ptr() as *const TOutput,
            config.problem_m * config.problem_n,
        )
        .to_vec()
    }
}

fn assert_probe_matches_reference<TInput: ProbeScalar, TOutput: ProbeScalar + ArrayElement>(
    config: ProbeRunConfig,
    tolerance: f32,
    message: &str,
) {
    let (left, right) = create_probe_input_matrices::<TInput>(
        config.problem_m,
        config.problem_k,
        config.problem_n,
        config.transpose_right,
    );
    let expected = compute_probe_reference_output::<TInput, TOutput>(&left, &right, config);
    let actual = run_probe_kernel::<TInput, TOutput>(config, false);
    assert_eq_float(&expected, &actual, tolerance, message);
}

fn assert_raw_destination_matches_reference<TInput: ProbeScalar, TOutput: ProbeScalar + ArrayElement>(
    config: ProbeRunConfig,
    tolerance: f32,
    message: &str,
) {
    let (left, right) = create_probe_input_matrices::<TInput>(
        config.problem_m,
        config.problem_k,
        config.problem_n,
        config.transpose_right,
    );
    let expected = compute_probe_reference_output::<TInput, TOutput>(&left, &right, config);
    let actual = run_probe_kernel::<TInput, TOutput>(config, true);
    assert_eq_float(&expected, &actual, tolerance, message);
}

#[uzu_test]
fn test_mpp_probe_device_bfloat_to_bfloat() {
    assert_probe_matches_reference::<bf16, bf16>(
        ProbeRunConfig {
            problem_m: 32,
            problem_n: 32,
            problem_k: 32,
            descriptor_m: 32,
            descriptor_n: 32,
            transpose_right: true,
            storage: ProbeStorage::Device,
            scope: ProbeScope::Simdgroup,
            slice_mode: ProbeSliceMode::DirectTensor,
            k_loop: None,
        },
        0.2,
        "device bfloat->bfloat probe",
    );
}

#[uzu_test]
fn test_mpp_probe_threadgroup_half_to_half() {
    assert_probe_matches_reference::<f16, f16>(
        ProbeRunConfig {
            problem_m: 32,
            problem_n: 32,
            problem_k: 32,
            descriptor_m: 32,
            descriptor_n: 32,
            transpose_right: true,
            storage: ProbeStorage::Threadgroup,
            scope: ProbeScope::Simdgroup,
            slice_mode: ProbeSliceMode::DirectTensor,
            k_loop: None,
        },
        0.05,
        "threadgroup half->half probe",
    );
}

#[uzu_test]
fn test_mpp_probe_device_float_to_float() {
    assert_probe_matches_reference::<f32, f32>(
        ProbeRunConfig {
            problem_m: 32,
            problem_n: 32,
            problem_k: 32,
            descriptor_m: 32,
            descriptor_n: 32,
            transpose_right: true,
            storage: ProbeStorage::Device,
            scope: ProbeScope::Simdgroup,
            slice_mode: ProbeSliceMode::DirectTensor,
            k_loop: None,
        },
        0.01,
        "device float->float probe",
    );
}

#[uzu_test]
fn test_mpp_probe_device_half_to_float() {
    assert_probe_matches_reference::<f16, f32>(
        ProbeRunConfig {
            problem_m: 32,
            problem_n: 32,
            problem_k: 32,
            descriptor_m: 32,
            descriptor_n: 32,
            transpose_right: true,
            storage: ProbeStorage::Device,
            scope: ProbeScope::Simdgroup,
            slice_mode: ProbeSliceMode::DirectTensor,
            k_loop: None,
        },
        0.05,
        "device half->float probe",
    );
}

#[uzu_test]
fn test_mpp_probe_device_bfloat_to_float() {
    assert_probe_matches_reference::<bf16, f32>(
        ProbeRunConfig {
            problem_m: 32,
            problem_n: 32,
            problem_k: 32,
            descriptor_m: 32,
            descriptor_n: 32,
            transpose_right: true,
            storage: ProbeStorage::Device,
            scope: ProbeScope::Simdgroup,
            slice_mode: ProbeSliceMode::DirectTensor,
            k_loop: None,
        },
        0.1,
        "device bfloat->float probe",
    );
}

#[uzu_test]
fn test_mpp_probe_device_ragged_half_to_half() {
    assert_probe_matches_reference::<f16, f16>(
        ProbeRunConfig {
            problem_m: 29,
            problem_n: 31,
            problem_k: 27,
            descriptor_m: 32,
            descriptor_n: 32,
            transpose_right: true,
            storage: ProbeStorage::Device,
            scope: ProbeScope::Simdgroup,
            slice_mode: ProbeSliceMode::DirectTensor,
            k_loop: None,
        },
        0.05,
        "device ragged half->half probe",
    );
}

#[uzu_test]
fn test_mpp_probe_static_origin_slice_half_to_half() {
    assert_probe_matches_reference::<f16, f16>(
        ProbeRunConfig {
            problem_m: 32,
            problem_n: 32,
            problem_k: 32,
            descriptor_m: 32,
            descriptor_n: 32,
            transpose_right: true,
            storage: ProbeStorage::Device,
            scope: ProbeScope::Simdgroup,
            slice_mode: ProbeSliceMode::StaticOriginSlice,
            k_loop: None,
        },
        0.05,
        "static origin slice half->half probe",
    );
}

#[uzu_test]
fn test_mpp_probe_dynamic_origin_slice_half_to_half() {
    assert_probe_matches_reference::<f16, f16>(
        ProbeRunConfig {
            problem_m: 32,
            problem_n: 32,
            problem_k: 32,
            descriptor_m: 32,
            descriptor_n: 32,
            transpose_right: true,
            storage: ProbeStorage::Device,
            scope: ProbeScope::Simdgroup,
            slice_mode: ProbeSliceMode::DynamicOriginSlice,
            k_loop: None,
        },
        0.05,
        "dynamic origin slice half->half probe",
    );
}

#[uzu_test]
fn test_mpp_probe_kloop_half_to_half() {
    assert_probe_matches_reference::<f16, f16>(
        ProbeRunConfig {
            problem_m: 32,
            problem_n: 32,
            problem_k: 64,
            descriptor_m: 32,
            descriptor_n: 32,
            transpose_right: true,
            storage: ProbeStorage::Device,
            scope: ProbeScope::Simdgroup,
            slice_mode: ProbeSliceMode::DirectTensor,
            k_loop: Some(ProbeKLoop { tile_k: 32 }),
        },
        0.05,
        "K-loop half->half probe",
    );
}

#[uzu_test]
fn test_mpp_probe_kloop_tail_half_to_half() {
    assert_probe_matches_reference::<f16, f16>(
        ProbeRunConfig {
            problem_m: 32,
            problem_n: 32,
            problem_k: 65,
            descriptor_m: 32,
            descriptor_n: 32,
            transpose_right: true,
            storage: ProbeStorage::Device,
            scope: ProbeScope::Simdgroup,
            slice_mode: ProbeSliceMode::DirectTensor,
            k_loop: Some(ProbeKLoop { tile_k: 32 }),
        },
        0.05,
        "K-loop tail half->half probe",
    );
}

#[uzu_test]
fn test_mpp_probe_execution_simdgroups4_half_to_half() {
    assert_probe_matches_reference::<f16, f16>(
        ProbeRunConfig {
            problem_m: 64,
            problem_n: 64,
            problem_k: 32,
            descriptor_m: 64,
            descriptor_n: 64,
            transpose_right: true,
            storage: ProbeStorage::Device,
            scope: ProbeScope::Simdgroups4,
            slice_mode: ProbeSliceMode::DirectTensor,
            k_loop: None,
        },
        0.05,
        "execution_simdgroups<4> half->half probe",
    );
}

#[uzu_test]
fn test_mpp_probe_raw_tensor_destination_half_to_half() {
    assert_raw_destination_matches_reference::<f16, f16>(
        ProbeRunConfig {
            problem_m: 32,
            problem_n: 32,
            problem_k: 32,
            descriptor_m: 32,
            descriptor_n: 32,
            transpose_right: true,
            storage: ProbeStorage::Device,
            scope: ProbeScope::Simdgroup,
            slice_mode: ProbeSliceMode::DirectTensor,
            k_loop: None,
        },
        0.05,
        "raw tensor destination half->half probe",
    );
}
