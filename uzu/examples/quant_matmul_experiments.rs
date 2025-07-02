use std::collections::HashMap;
use std::rc::Rc;
use std::time::{SystemTime, UNIX_EPOCH, Instant};

use clap::Parser;
use half::f16;
use metal::Device;
use mpsgraph::{
    CommandBuffer, DataType as MPSDataType, ExecutableExecutionDescriptor, Graph,
    GraphQuantizationOps, Shape, ShapedType, Optimization, OptimizationProfile,
    GraphTensorShapeOps,
};
use objc2::rc::autoreleasepool;
use thiserror::Error;

use uzu::backends::metal::{
    compilation_parameters::{make_compilation_descriptor, BlockDevice},
    utils::mps_shape,
    MTLContext,
};
use uzu::{Array, DataType, DeviceContext};
use uzu::root_dir::{root_dir, RootLocation};

#[derive(Parser)]
#[command(
    name = "quant_matmul_experiments",
    about = "Synthetic experiment: dequantize weights then matmul, measure execution time"
)]
struct Args {
    /// Print the compiled executable IR / debug dump
    #[arg(long = "print-exec-dump")]
    print_exec_dump: bool,

    /// Print ANE/GPU placement analysis during compilation
    #[arg(long = "print-placement-analysis")]
    print_placement_analysis: bool,

    /// Serialize the compiled MPSGraph executable to a .mpsgraphpackage file
    #[arg(long = "dump-mpsgraphpackage")]
    dump_mpsgraphpackage: bool,
}

#[derive(Debug, Error)]
pub enum ExampleError {
    #[error(transparent)]
    Metal(#[from] uzu::backends::metal::error::MTLError),
    #[error("No Metal device available")]
    NoDevice,
    #[error("Graph dequantization failed")] 
    Dequantization,
}

fn main() -> Result<(), ExampleError> {
    autoreleasepool(|_| {
        let args = Args::parse();

        // --- Constants ---

        let suffix_length: usize = 128;
        let model_dim: usize = 2560;
        let output_dim: usize = 3072 * 2;
        let group_size: usize = 20;

        let weights_shape = [output_dim, model_dim];
        let scales_shape = [output_dim, model_dim / group_size];
        let zero_points_shape = scales_shape;
        let input_shape = [-1_i64, model_dim as i64];

        // --- Concrete shapes for actual buffer allocation at runtime ---

        let input_tensor_shape = [suffix_length, model_dim];
        let result_tensor_shape = [suffix_length, output_dim];

        // --- Device ---

        let device = Device::system_default().ok_or(ExampleError::NoDevice)?;
        let command_queue = device.new_command_queue();
        let mtl_context = Rc::new(MTLContext::new(device, command_queue)?);

        // --- Arrays ---

        let mut weights_array = unsafe {
            mtl_context.array_uninitialized(&weights_shape, DataType::U4)
        };
        weights_array.buffer_mut().fill(0x11);

        let scales_array = mtl_context.array_from_elem(&scales_shape, f16::from_f32(1.0));

        let mut zero_points_array = unsafe {
            mtl_context.array_uninitialized(&zero_points_shape, DataType::U4)
        };
        zero_points_array.buffer_mut().fill(0x00);

        let mut input_array = mtl_context.array_from_elem(&input_tensor_shape, f16::from_f32(1.0));

        let mut result_array = unsafe {
            mtl_context.array_uninitialized(&result_tensor_shape, DataType::F16)
        };

        // --- Graph ---

        let graph = Graph::new();

        let weights_const = graph.constant_with_data(
            weights_array.buffer(),
            &mps_shape(&weights_shape),
            DataType::U4.into(),
        );

        let scales_const = graph.constant_with_data(
            scales_array.buffer(),
            &mps_shape(&scales_shape),
            DataType::F16.into(),
        );

        let zero_points_const = graph.constant_with_data(
            zero_points_array.buffer(),
            &mps_shape(&zero_points_shape),
            DataType::U4.into(),
        );

        let input_placeholder = graph.placeholder(
            MPSDataType::Float16,
            &Shape::from_dimensions(&input_shape),
            Some("input"),
        );

        // --- Dequantize weights ---

        let dequantized_weights = graph
            .dequantize_with_scale_tensor_and_zero_point_tensor(
                &weights_const,
                &scales_const,
                &zero_points_const,
                MPSDataType::Float16,
                None,
            )
            .ok_or(ExampleError::Dequantization)?;

        // --- Matmul ---

        let matmul = graph.matmul(
            &input_placeholder,
            &graph.transpose(&dequantized_weights, &[1, 0], None),
            false,
            false,
            None,
        );

        // --- Compile ---

        let optimization_level = Optimization::Level1;
        let optimization_profile = OptimizationProfile::PowerEfficiency;
        let compilation_descriptor = make_compilation_descriptor(
            BlockDevice::Ane,
            optimization_level,
            optimization_profile,
            args.print_placement_analysis,
        );

        let mut feeds: HashMap<&mpsgraph::Tensor, &ShapedType> = HashMap::new();
        let input_shaped_type = ShapedType::new(
            &Shape::from_dimensions(&input_shape),
            MPSDataType::Float16,
        );
        feeds.insert(input_placeholder.as_ref(), input_shaped_type.as_ref());

        let executable = graph.compile(
            &mpsgraph::device::Device::with_device(&mtl_context.device),
            &feeds,
            &[matmul.as_ref()],
            Some(&compilation_descriptor),
        );

        // --- Print ---

        if args.print_exec_dump {
            executable.dump();
        }

        if args.dump_mpsgraphpackage {
            let timestamp = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("Time went backwards")
                .as_secs();
            let package_path = root_dir(RootLocation::Downloads).join(format!(
                "quant_matmul_experiments-{timestamp}.mpsgraphpackage"
            ));
            let serialization_desc = mpsgraph::SerializationDescriptor::new();
            executable.serialize_to_url(&package_path, &serialization_desc);
            println!("MPSGraph package saved to {:?}", package_path);
        }

        // --- Run ---

        let input_td = unsafe { input_array.to_mps_tensor_data() };
        let result_td = unsafe { result_array.to_mps_tensor_data() };

        let inputs = [input_td.as_ref()];
        let outputs = [result_td.as_ref()];

        run_once(&mtl_context, &executable, &inputs, &outputs);

        const NUM_ITERATIONS: usize = 50;
        let mut total_duration = 0.0f64; // milliseconds

        for _ in 0..NUM_ITERATIONS {
            let start = Instant::now();
            run_once(&mtl_context, &executable, &inputs, &outputs);
            total_duration += start.elapsed().as_secs_f64() * 1e3;
        }

        let avg_ms = total_duration / NUM_ITERATIONS as f64;
        println!(
            "Average execution time over {} iterations: {:.3} ms",
            NUM_ITERATIONS, avg_ms
        );

        Ok(())
    })
}

fn run_once(
    mtl_context: &Rc<MTLContext>,
    executable: &mpsgraph::Executable,
    inputs: &[&mpsgraph::tensor_data::TensorData],
    outputs: &[&mpsgraph::tensor_data::TensorData],
) {
    let command_buffer = CommandBuffer::from_command_queue(&mtl_context.command_queue);
    let root_command_buffer = command_buffer.root_command_buffer().to_owned();

    let exec_desc = ExecutableExecutionDescriptor::new();

    executable.encode_to_command_buffer(
        &command_buffer,
        inputs,
        Some(outputs),
        Some(&exec_desc),
    );

    command_buffer.commit_and_continue();
    root_command_buffer.wait_until_completed();
} 