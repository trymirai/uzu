use std::{
    collections::HashMap,
    fs::File,
    rc::Rc,
    time::{SystemTime, UNIX_EPOCH},
};

use clap::Parser;
use metal::{CaptureDescriptor, CaptureManager, MTLCaptureDestination};
use mpsgraph::{
    CommandBuffer, DequantizationArguments, Device as MPSDevice,
    ExecutableExecutionDescriptor, ExecutableSerializationDescriptor, Graph,
    Optimization, OptimizationProfile, ShapedType, Tensor, TensorData,
};
use objc2::rc::autoreleasepool;
use thiserror::Error;
use uzu::{
    Array, DataType, DeviceContext,
    backends::metal::{
        MTLContext,
        compilation_parameters::{BlockDevice, make_compilation_descriptor},
        error::MTLError,
        graph::common::{GraphConstructionError, load_constant},
    },
    parameters::{ParameterLoader, ParameterLoaderError},
    utils::{NSSearchPathDirectory, root_dir},
};

#[derive(Debug, Error)]
pub enum ExampleError {
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Metal(#[from] MTLError),
    #[error(transparent)]
    Loader(#[from] ParameterLoaderError),
    #[error("header loading error")]
    Header,
    #[error("capture error")]
    Capture,
    #[error(transparent)]
    Graph(#[from] GraphConstructionError),
    #[error("dequantization failed")]
    Dequantization,
    #[error("result buffer view unavailable")]
    ResultView,
}

#[derive(Parser)]
#[command(
    name = "dequantize_qkv_layer0",
    about = "Demonstrates QKV dequantization with optional GPU trace capture and MPSGraph package serialization"
)]
struct Args {
    /// Dump a .gputrace file capturing the GPU execution
    #[arg(long = "dump-gpu-trace")]
    dump_gpu_trace: bool,

    /// Serialize the compiled MPSGraph executable to a .mpsgraphpackage file
    #[arg(long = "dump-mpsgraphpackage")]
    dump_mpsgraphpackage: bool,

    /// Perform and print ANE/GPU placement analysis during compilation
    #[arg(long = "print-placement-analysis")]
    print_placement_analysis: bool,

    /// Print the executable's IR / debug info after compilation
    #[arg(long = "print-executable-dump")]
    print_executable_dump: bool,

    /// Dump the result array to a text file in ~/Downloads with a timestamped name
    #[arg(long = "dump-result-array")]
    dump_result_array: bool,
}

fn main() -> Result<(), ExampleError> {
    autoreleasepool(|_| {
        let args = Args::parse();
        let dump_gpu_trace = args.dump_gpu_trace;
        let dump_mpsgraphpackage = args.dump_mpsgraphpackage;
        let print_placement_analysis = args.print_placement_analysis;
        let print_executable_dump = args.print_executable_dump;
        let dump_result_array = args.dump_result_array;

        if dump_gpu_trace {
            unsafe {
                std::env::set_var("METAL_CAPTURE_ENABLED", "1");
            }
        }

        let model_dir = root_dir(NSSearchPathDirectory::Downloads)
            .join("Qwen3-4B-AWQ-Temp");
        let safetensors = File::open(model_dir.join("model.safetensors"))?;

        let device =
            metal::Device::system_default().ok_or(ExampleError::Capture)?;
        let command_queue = device.new_command_queue();
        let mtl_context = Rc::new(MTLContext::new(device, command_queue)?);

        let loader = ParameterLoader::new(&safetensors, &mtl_context)
            .map_err(|_| ExampleError::Header)?;
        let tree = loader
            .tree()
            .subtree("layers")?
            .subtree("0")?
            .subtree("attention")?
            .subtree("qkv_projection")?;

        let graph = Graph::new();

        let weights = load_constant(
            &graph,
            &tree,
            "weights",
            &[3072 * 2, 2560],
            DataType::U4,
        )
        .map_err(ExampleError::from)?;

        let scales =
            load_constant(&graph, &tree, "scales", &[6144, 20], DataType::F16)
                .map_err(ExampleError::from)?;

        let zero_points = load_constant(
            &graph,
            &tree,
            "zero_points",
            &[3072 * 2, 20],
            DataType::U4,
        )
        .map_err(ExampleError::from)?;

        let dequantized_weights = graph.dequantize(
            &weights,
            DequantizationArguments::ScaleTensorZeroPointTensorDataType {
                scale_tensor: &scales,
                zero_point_tensor: &zero_points,
                data_type: DataType::F16.into(),
            },
            None,
        );

        let mut dequantized_weights_buffer = unsafe {
            mtl_context.array_uninitialized(&[3072 * 2, 2560], DataType::F16)
        };
        let dequantized_weights_tensor_data = unsafe {
            TensorData::new_with_mtl_buffer(
                &dequantized_weights_buffer.mtl_buffer(),
                &[3072 * 2, 2560],
                DataType::F16.into(),
                None,
            )
        };

        let capture_manager = CaptureManager::shared();
        let capture_manager_descriptor = CaptureDescriptor::new();
        capture_manager_descriptor
            .set_destination(MTLCaptureDestination::GpuTraceDocument);

        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("Time went backwards")
            .as_secs();
        let trace_path = root_dir(NSSearchPathDirectory::Downloads)
            .join(format!("dequantize_qkv_layer0-{timestamp}.gputrace"));

        capture_manager_descriptor.set_output_url(trace_path);

        mtl_context.command_queue.set_label("uzu_command_queue");
        capture_manager_descriptor
            .set_capture_command_queue(&mtl_context.command_queue);

        let feeds: HashMap<&Tensor, &ShapedType> = HashMap::new();
        let optimization_level = Optimization::Level1;
        let optimization_profile = OptimizationProfile::Performance;

        let compilation_descriptor = make_compilation_descriptor(
            BlockDevice::Ane,
            optimization_level,
            optimization_profile,
            print_placement_analysis,
        );

        let executable = graph.compile(
            &MPSDevice::with_device(&mtl_context.device),
            &feeds,
            &[&dequantized_weights],
            None,
            Some(&compilation_descriptor),
        );

        if print_executable_dump {
            executable.dump();
        }

        if dump_mpsgraphpackage {
            let package_path = root_dir(NSSearchPathDirectory::Downloads).join(
                format!("dequantize_qkv_layer0-{timestamp}.mpsgraphpackage"),
            );
            let serialization_desc = ExecutableSerializationDescriptor::new();
            executable.serialize_to_graph_package(
                &package_path,
                Some(&serialization_desc),
            );
            println!("MPSGraph package saved to: {:?}", package_path);
        }

        if dump_gpu_trace {
            if let Err(err) =
                capture_manager.start_capture(&capture_manager_descriptor)
            {
                eprintln!("⚠️  Failed to start GPU capture: {err}");
            } else {
                println!("GPU capture started successfully");
            }
        }

        let command_buffer =
            CommandBuffer::from_command_queue(&mtl_context.command_queue);
        let root_command_buffer =
            command_buffer.root_command_buffer().to_owned();

        let execution_descriptor = ExecutableExecutionDescriptor::new();

        executable.encode_to_command_buffer(
            &command_buffer,
            &[],
            Some(&[&dequantized_weights_tensor_data]),
            Some(&*execution_descriptor),
        );
        command_buffer.commit_and_continue();

        root_command_buffer.wait_until_completed();

        if dump_gpu_trace {
            capture_manager.stop_capture();
        }

        let result = dequantized_weights_buffer
            .as_view::<half::f16>()
            .map_err(|_| ExampleError::ResultView)?;

        if dump_result_array {
            let result_path = root_dir(NSSearchPathDirectory::Downloads)
                .join(format!("dequantize_qkv_layer0-{timestamp}.txt"));
            std::fs::write(&result_path, format!("{:?}", result))?;
            println!("Result array dumped to {:?}", result_path);
        }

        Ok(())
    })
}
