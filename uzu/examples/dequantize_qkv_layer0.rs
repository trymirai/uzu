use std::{collections::HashMap, fs::File, rc::Rc};

use metal::{
    CaptureDescriptor, CaptureManager, CaptureScope, MTLCaptureDestination,
};
use mpsgraph::{
    CommandBuffer, CompilationDescriptor, Device as MPSDevice,
    ExecutableExecutionDescriptor, Graph, GraphQuantizationOps, Shape,
    ShapedType, Tensor, TensorData,
};
use ndarray::s;
use objc2::rc::autoreleasepool;
use thiserror::Error;
use uzu::{
    Array, DataType, DeviceContext,
    backends::metal::{
        MTLContext, error::MTLError, graph::common::load_constant,
    },
    parameters::{ParameterLoader, ParameterLoaderError},
};
use uzu::root_dir::{RootLocation, root_dir};

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
        let model_dir = root_dir(RootLocation::Downloads).join("Qwen3-4B-AWQ");
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
            &[2560, 3072 * 2],
            DataType::U4,
        )
        .map_err(|_| ExampleError::ParamMismatch)
        .unwrap();

        let scales =
            load_constant(&graph, &tree, "scales", &[20, 6144], DataType::F16)
                .map_err(|_| ExampleError::ParamMismatch)
                .unwrap();

        let scales_ones = graph.constant_with_filled_scalar(
            1.0_f64,
            DataType::F16.into(),
            &Shape::from_dimensions(&[20, 6144]),
        );

        let zero_points = load_constant(
            &graph,
            &tree,
            "zero_points",
            &[20, 3072 * 2],
            DataType::U4,
        )
        .map_err(|_| ExampleError::ParamMismatch)
        .unwrap();

        let zero_points_zeroes = graph.constant_with_filled_scalar(
            0.0_f64,
            DataType::U4.into(),
            &Shape::from_dimensions(&[20, 3072 * 2]),
        );

        let dequantized_weights = graph
            .dequantize_with_scale_tensor_and_zero_point_tensor(
                &weights,
                &scales_ones,
                &zero_points,
                DataType::F16.into(),
                None,
            )
            .ok_or(ExampleError::ParamMismatch)
            .unwrap();

        let mut dequantized_weights_buffer = unsafe {
            mtl_context.array_uninitialized(&[2560, 3072 * 2], DataType::F16)
        };
        let dequantized_weights_tensor_data = unsafe {
            TensorData::from_buffer(
                &dequantized_weights_buffer.mtl_buffer(),
                &Shape::from_dimensions(&[2560, 3072 * 2]),
                DataType::F16.into(),
            )
        };

        let capture_manager = CaptureManager::shared();
        let capture_manager_descriptor = CaptureDescriptor::new();
        capture_manager_descriptor
            .set_destination(MTLCaptureDestination::GpuTraceDocument);
        capture_manager_descriptor
            .set_output_url("dequantize_qkv_layer0.gputrace");
        //     root_dir(RootLocation::Downloads)
        //         .join("dequantize_qkv_layer0.gputrace"),
        // );

        let output_url = capture_manager_descriptor.output_url();
        println!("output_url: {:?}", output_url);

        let feeds: HashMap<&Tensor, &ShapedType> = HashMap::new();
        let compilation_descriptor = CompilationDescriptor::new();
        let executable = graph.compile(
            &MPSDevice::with_device(&mtl_context.device),
            &feeds,
            &[&dequantized_weights],
            Some(&compilation_descriptor),
        );
        executable.dump();

        let _ = capture_manager.start_capture(&capture_manager_descriptor);

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

        capture_manager.stop_capture();

        let result = dequantized_weights_buffer.as_view::<half::f16>().unwrap();
        // println!("{:?}", result.slice(s![0, 0..8]));

        // println!("result: {:?}", result);

        Ok(())
    })
}
