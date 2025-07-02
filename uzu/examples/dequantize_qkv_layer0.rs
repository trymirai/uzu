use std::time::{SystemTime, UNIX_EPOCH};
use std::{collections::HashMap, fs::File, rc::Rc};

use metal::{
    CaptureDescriptor, CaptureManager, MTLCaptureDestination,
};
use mpsgraph::{
    CommandBuffer, Device as MPSDevice, ExecutableExecutionDescriptor, Graph,
    GraphQuantizationOps, Shape, ShapedType, Tensor, TensorData,
};
use mpsgraph::{Optimization, OptimizationProfile};
use ndarray::s;
use objc2::rc::autoreleasepool;
use thiserror::Error;
use uzu::root_dir::{RootLocation, root_dir};
use uzu::{
    Array, DataType, DeviceContext,
    backends::metal::{
        MTLContext,
        compilation_parameters::{BlockDevice, make_compilation_descriptor},
        error::MTLError,
        graph::common::load_constant,
    },
    parameters::{ParameterLoader, ParameterLoaderError},
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
    #[error("parameter mismatch")]
    ParamMismatch,
    #[error("capture error")]
    Capture,
}

fn main() -> Result<(), ExampleError> {
    autoreleasepool(|_| {
        let dump_gpu_trace = false;

        let model_dir =
            root_dir(RootLocation::Downloads).join("Qwen3-4B-AWQ-Temp");
        let safetensors =
            File::open(model_dir.join("model.safetensors")).unwrap();

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
        .map_err(|_| ExampleError::ParamMismatch)
        .unwrap();

        let scales =
            load_constant(&graph, &tree, "scales", &[6144, 20], DataType::F16)
                .map_err(|_| ExampleError::ParamMismatch)
                .unwrap();

        // let scales_ones = graph.constant_with_filled_scalar(
        //     1.0_f64,
        //     DataType::F16.into(),
        //     &Shape::from_dimensions(&[20, 6144]),
        // );

        let zero_points = load_constant(
            &graph,
            &tree,
            "zero_points",
            &[3072 * 2, 20],
            DataType::U4,
        )
        .map_err(|_| ExampleError::ParamMismatch)
        .unwrap();

        // let zero_points_zeroes = graph.constant_with_filled_scalar(
        //     0.0_f64,
        //     DataType::U4.into(),
        //     &Shape::from_dimensions(&[20, 3072 * 2]),
        // );

        let dequantized_weights = graph
            .dequantize_with_scale_tensor_and_zero_point_tensor(
                &weights,
                &scales,
                &zero_points,
                DataType::F16.into(),
                None,
            )
            .ok_or(ExampleError::ParamMismatch)
            .unwrap();

        let mut dequantized_weights_buffer = unsafe {
            mtl_context.array_uninitialized(&[3072 * 2, 2560], DataType::F16)
        };
        let dequantized_weights_tensor_data = unsafe {
            TensorData::from_buffer(
                &dequantized_weights_buffer.mtl_buffer(),
                &Shape::from_dimensions(&[3072 * 2, 2560]),
                DataType::F16.into(),
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
        let trace_path = root_dir(RootLocation::Downloads)
            .join(format!("dequantize_qkv_layer0-{timestamp}.gputrace"));

        capture_manager_descriptor.set_output_url(trace_path);

        mtl_context.command_queue.set_label("uzu_command_queue");
        capture_manager_descriptor
            .set_capture_command_queue(&mtl_context.command_queue);

        let output_url = capture_manager_descriptor.output_url();
        println!("output_url: {:?}", output_url);

        let feeds: HashMap<&Tensor, &ShapedType> = HashMap::new();
        let optimization_level = Optimization::Level1;
        let optimization_profile = OptimizationProfile::Performance;
        let perform_placement_analysis = false;
        let compilation_descriptor = make_compilation_descriptor(
            BlockDevice::Ane,
            optimization_level,
            optimization_profile,
            perform_placement_analysis,
        );

        let executable = graph.compile(
            &MPSDevice::with_device(&mtl_context.device),
            &feeds,
            &[&dequantized_weights],
            Some(&compilation_descriptor),
        );
        executable.dump();

        if dump_gpu_trace {
            if let Err(err) = capture_manager.start_capture(&capture_manager_descriptor) {
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

        let result = dequantized_weights_buffer.as_view::<half::f16>().unwrap();
        println!("{:?}", result.slice(s![0, 0..8]));

        // println!("result: {:?}", result);

        Ok(())
    })
}
