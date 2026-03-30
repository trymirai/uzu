#![cfg(metal_backend)]

use metal::{MTLBuffer, MTLDeviceExt, MTLResourceOptions};
use objc2::{rc::Retained, runtime::ProtocolObject};
use uzu::{
    DataType,
    backends::{
        common::{
            Backend, Context, Encoder, Kernels,
            gpu_types::QuantizationMode,
            kernel::{
                HadamardTransformMulKernel,
                quant_matmul::{
                    QuantizedMatmulArguments, QuantizedMatmulConfiguration, QuantizedMatmulKernelEncodable,
                    QuantizedMatmulType,
                },
            },
        },
        metal::Metal,
    },
};

type MetalContext = <Metal as Backend>::Context;
type MetalBuffer = Retained<ProtocolObject<dyn MTLBuffer>>;

const WARMUP_ITERATIONS: usize = 5;
const BENCHMARK_ITERATIONS: usize = 50;
const GROUP_SIZE: usize = 32;
const PACKING_DIVISOR: usize = 2; // uint4 packs 2 values per byte

struct BenchmarkShape {
    label: &'static str,
    batch_size: usize,
    input_dimension: usize,
    output_dimension: usize,
}

const BENCHMARK_SHAPES: &[BenchmarkShape] = &[
    BenchmarkShape {
        label: "decode_out_proj",
        batch_size: 1,
        input_dimension: 2048,
        output_dimension: 2048,
    },
    BenchmarkShape {
        label: "decode_in_proj",
        batch_size: 1,
        input_dimension: 2048,
        output_dimension: 6144,
    },
    BenchmarkShape {
        label: "decode_mlp_up",
        batch_size: 1,
        input_dimension: 2048,
        output_dimension: 16384,
    },
    BenchmarkShape {
        label: "decode_mlp_down",
        batch_size: 1,
        input_dimension: 8192,
        output_dimension: 2048,
    },
    BenchmarkShape {
        label: "prefill_128_out_proj",
        batch_size: 128,
        input_dimension: 2048,
        output_dimension: 2048,
    },
    BenchmarkShape {
        label: "prefill_128_in_proj",
        batch_size: 128,
        input_dimension: 2048,
        output_dimension: 6144,
    },
];

fn fill_buffer_deterministic(
    buffer: &ProtocolObject<dyn MTLBuffer>,
    byte_count: usize,
) {
    let pointer = buffer.contents().as_ptr() as *mut u8;
    let slice = unsafe { std::slice::from_raw_parts_mut(pointer, byte_count) };
    for (index, byte) in slice.iter_mut().enumerate() {
        *byte = (index % 251) as u8;
    }
}

fn allocate_buffer(
    context: &MetalContext,
    byte_count: usize,
) -> MetalBuffer {
    context
        .device
        .new_buffer(byte_count, MTLResourceOptions::STORAGE_MODE_SHARED)
        .expect("Failed to allocate Metal buffer")
}

struct QuantizedMatmulBuffers {
    input_buffer: MetalBuffer,
    weights_buffer: MetalBuffer,
    scales_buffer: MetalBuffer,
    zero_points_buffer: MetalBuffer,
    output_buffer: MetalBuffer,
}

fn allocate_quantized_matmul_buffers(
    context: &MetalContext,
    shape: &BenchmarkShape,
) -> QuantizedMatmulBuffers {
    let activation_element_size = DataType::BF16.size_in_bytes();
    let groups_per_row = (shape.input_dimension + GROUP_SIZE - 1) / GROUP_SIZE;
    let zero_point_entries = (groups_per_row + PACKING_DIVISOR - 1) / PACKING_DIVISOR;

    let input_buffer = allocate_buffer(context, shape.batch_size * shape.input_dimension * activation_element_size);
    let weights_buffer = allocate_buffer(context, shape.output_dimension * shape.input_dimension / PACKING_DIVISOR);
    let scales_buffer = allocate_buffer(context, shape.output_dimension * groups_per_row * activation_element_size);
    let zero_points_buffer = allocate_buffer(context, shape.output_dimension * zero_point_entries);
    let output_buffer = allocate_buffer(context, shape.batch_size * shape.output_dimension * activation_element_size);

    fill_buffer_deterministic(&input_buffer, shape.batch_size * shape.input_dimension * activation_element_size);
    fill_buffer_deterministic(&weights_buffer, shape.output_dimension * shape.input_dimension / PACKING_DIVISOR);
    fill_buffer_deterministic(&scales_buffer, shape.output_dimension * groups_per_row * activation_element_size);
    fill_buffer_deterministic(&zero_points_buffer, shape.output_dimension * zero_point_entries);

    QuantizedMatmulBuffers {
        input_buffer,
        weights_buffer,
        scales_buffer,
        zero_points_buffer,
        output_buffer,
    }
}

fn run_plain_quantized_matmul(
    context: &MetalContext,
    kernel: &QuantizedMatmulKernelEncodable<Metal>,
    shape: &BenchmarkShape,
    buffers: &mut QuantizedMatmulBuffers,
) -> f64 {
    let mut encoder = Encoder::new(context).expect("Failed to create encoder");

    kernel
        .encode(
            &mut encoder,
            QuantizedMatmulArguments {
                a_buffer: &buffers.input_buffer,
                a_offset: 0,
                b_buffer: &buffers.weights_buffer,
                scales_buffer: &buffers.scales_buffer,
                zero_points_or_biases_buffer: &buffers.zero_points_buffer,
                output_buffer: &mut buffers.output_buffer,
                batch: shape.batch_size,
                input_dim: shape.input_dimension,
                output_dim: shape.output_dimension,
                quantization_type: QuantizedMatmulType::ZeroPoint,
            },
        )
        .expect("Failed to encode quantized matmul");

    let completed = encoder.end_encoding().submit().wait_until_completed().expect("Command buffer execution failed");

    completed.gpu_execution_time().map(|duration| duration.as_secs_f64() * 1e6).expect("GPU timestamps not available")
}

fn run_rht_quantized_matmul(
    context: &MetalContext,
    quantized_matmul_kernel: &QuantizedMatmulKernelEncodable<Metal>,
    input_hadamard_kernel: &<<Metal as Backend>::Kernels as Kernels>::HadamardTransformMulKernel,
    output_hadamard_kernel: &<<Metal as Backend>::Kernels as Kernels>::HadamardTransformMulKernel,
    shape: &BenchmarkShape,
    buffers: &mut QuantizedMatmulBuffers,
    input_factors_buffer: &MetalBuffer,
    output_factors_buffer: &MetalBuffer,
) -> f64 {
    let input_total_blocks = (shape.batch_size * shape.input_dimension / 32) as u32;
    let output_total_blocks = (shape.batch_size * shape.output_dimension / 32) as u32;

    let mut encoder = Encoder::new(context).expect("Failed to create encoder");

    input_hadamard_kernel.encode(
        &mut buffers.input_buffer,
        &*input_factors_buffer,
        input_total_blocks,
        shape.input_dimension as u32,
        &mut encoder,
    );

    quantized_matmul_kernel
        .encode(
            &mut encoder,
            QuantizedMatmulArguments {
                a_buffer: &buffers.input_buffer,
                a_offset: 0,
                b_buffer: &buffers.weights_buffer,
                scales_buffer: &buffers.scales_buffer,
                zero_points_or_biases_buffer: &buffers.zero_points_buffer,
                output_buffer: &mut buffers.output_buffer,
                batch: shape.batch_size,
                input_dim: shape.input_dimension,
                output_dim: shape.output_dimension,
                quantization_type: QuantizedMatmulType::ZeroPoint,
            },
        )
        .expect("Failed to encode quantized matmul");

    output_hadamard_kernel.encode(
        &mut buffers.output_buffer,
        &*output_factors_buffer,
        output_total_blocks,
        shape.output_dimension as u32,
        &mut encoder,
    );

    let completed = encoder.end_encoding().submit().wait_until_completed().expect("Command buffer execution failed");

    completed.gpu_execution_time().map(|duration| duration.as_secs_f64() * 1e6).expect("GPU timestamps not available")
}

fn benchmark_shape(
    context: &MetalContext,
    shape: &BenchmarkShape,
    use_rht: bool,
) -> f64 {
    let quantized_matmul_kernel = QuantizedMatmulKernelEncodable::<Metal>::new(
        context,
        QuantizedMatmulConfiguration {
            data_type: DataType::BF16,
            group_size: GROUP_SIZE,
            input_dim: shape.input_dimension,
            output_dim: shape.output_dimension,
            mode: QuantizationMode::UINT4,
            quantization_type: QuantizedMatmulType::ZeroPoint,
            weights_transposed: true,
        },
    )
    .expect("Failed to create quantized matmul kernel");

    let mut buffers = allocate_quantized_matmul_buffers(context, shape);

    let activation_element_size = DataType::BF16.size_in_bytes();
    let input_factors_buffer = allocate_buffer(context, shape.input_dimension * activation_element_size);
    let output_factors_buffer = allocate_buffer(context, shape.output_dimension * activation_element_size);
    fill_buffer_deterministic(&input_factors_buffer, shape.input_dimension * activation_element_size);
    fill_buffer_deterministic(&output_factors_buffer, shape.output_dimension * activation_element_size);

    let input_hadamard_kernel =
        <<Metal as Backend>::Kernels as Kernels>::HadamardTransformMulKernel::new(context, DataType::BF16)
            .expect("Failed to create input hadamard kernel");
    let output_hadamard_kernel =
        <<Metal as Backend>::Kernels as Kernels>::HadamardTransformMulKernel::new(context, DataType::BF16)
            .expect("Failed to create output hadamard kernel");

    for _ in 0..WARMUP_ITERATIONS {
        if use_rht {
            run_rht_quantized_matmul(
                context,
                &quantized_matmul_kernel,
                &input_hadamard_kernel,
                &output_hadamard_kernel,
                shape,
                &mut buffers,
                &input_factors_buffer,
                &output_factors_buffer,
            );
        } else {
            run_plain_quantized_matmul(context, &quantized_matmul_kernel, shape, &mut buffers);
        }
    }

    let total_microseconds: f64 = (0..BENCHMARK_ITERATIONS)
        .map(|_| {
            if use_rht {
                run_rht_quantized_matmul(
                    context,
                    &quantized_matmul_kernel,
                    &input_hadamard_kernel,
                    &output_hadamard_kernel,
                    shape,
                    &mut buffers,
                    &input_factors_buffer,
                    &output_factors_buffer,
                )
            } else {
                run_plain_quantized_matmul(context, &quantized_matmul_kernel, shape, &mut buffers)
            }
        })
        .sum();

    total_microseconds / BENCHMARK_ITERATIONS as f64
}

fn main() {
    let use_rht = std::env::args().any(|argument| argument == "--rht");
    let mode_label = if use_rht {
        "RHT + QuantizedMatmul"
    } else {
        "QuantizedMatmul"
    };

    let context = MetalContext::new().expect("Failed to create Metal context");

    println!("Mode: {mode_label}");
    println!("Warmup: {WARMUP_ITERATIONS}, Iterations: {BENCHMARK_ITERATIONS}");
    println!("{:-<80}", "");
    println!("{:<25} {:>6} {:>10} {:>10} {:>12}", "Shape", "Batch", "InputDim", "OutputDim", "Avg GPU (us)");
    println!("{:-<80}", "");

    for shape in BENCHMARK_SHAPES {
        let average_microseconds = benchmark_shape(&context, shape, use_rht);
        println!(
            "{:<25} {:>6} {:>10} {:>10} {:>12.1}",
            shape.label, shape.batch_size, shape.input_dimension, shape.output_dimension, average_microseconds
        );
    }

    println!("{:-<80}", "");
}
