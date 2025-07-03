use std::{
    collections::HashMap,
    rc::Rc,
    time::{Instant, SystemTime, UNIX_EPOCH},
};

use clap::Parser;
use half::f16;
use metal::Device;
use mpsgraph::{
    CommandBuffer, CompilationDescriptor, DataType as MPSDataType,
    Device as MPSDevice, Executable, ExecutableExecutionDescriptor, Graph,
    GraphMatrixOps, GraphQuantizationOps, GraphTensorShapeOps, Optimization,
    OptimizationProfile, Shape, ShapedType, Tensor, tensor_data::TensorData,
};
use objc2::rc::{Retained, autoreleasepool};
use thiserror::Error;
use uzu::{
    Array, DataType, DeviceContext,
    backends::metal::{
        MTLContext, MetalArray,
        compilation_parameters::{BlockDevice, make_compilation_descriptor},
        utils::mps_shape,
    },
    root_dir::{RootLocation, root_dir},
};

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

// ================== Quantized Matmul Helpers =====================

#[derive(Debug, Clone, Copy)]
enum TensorLoadType {
    Baked,
    RuntimeLoaded,
}

#[derive(Debug, Clone, Copy)]
struct QuantizedMatmulDescriptor {
    weights: TensorLoadType,
    scales: TensorLoadType,
    zeros: TensorLoadType,
}

enum TensorOption {
    Constant(Retained<Tensor>),
    Placeholder {
        placeholder: Retained<Tensor>,
        data: Retained<TensorData>,
    },
}

impl TensorOption {
    #[inline]
    fn to_graph_tensor(&self) -> Retained<Tensor> {
        match self {
            Self::Constant(t) => t.clone(),
            Self::Placeholder {
                placeholder,
                ..
            } => placeholder.clone(),
        }
    }

    /// Return a HashMap that maps the placeholder tensor to its corresponding `TensorData`.
    /// For `Constant` variants the map is empty because no runtime data needs to be supplied.
    fn to_data_map(&self) -> HashMap<Retained<Tensor>, Retained<TensorData>> {
        match self {
            TensorOption::Constant(_) => HashMap::new(),
            TensorOption::Placeholder {
                placeholder,
                data,
            } => HashMap::from_iter([(placeholder.clone(), data.clone())]),
        }
    }
}

struct QuantizedMatmulTensors {
    weights: TensorOption,
    scales: TensorOption,
    zeros: TensorOption,
}

impl QuantizedMatmulTensors {
    fn collect_feeds(&self) -> Vec<(Retained<Tensor>, Retained<ShapedType>)> {
        let mut feeds = Vec::new();
        for tensor in [&self.weights, &self.scales, &self.zeros] {
            if let TensorOption::Placeholder {
                placeholder,
                ..
            } = tensor
            {
                feeds.push((placeholder.clone(), placeholder.shaped_type()));
            }
        }
        feeds
    }
}

fn build_quantized_matmul(
    graph: &Graph,
    tensors: &QuantizedMatmulTensors,
    input_placeholder: &Tensor,
    mtl_context: &Rc<MTLContext>,
    compilation_descriptor: &CompilationDescriptor,
) -> Result<Retained<Executable>, ExampleError> {
    let w = tensors.weights.to_graph_tensor();
    let s = tensors.scales.to_graph_tensor();
    let zp = tensors.zeros.to_graph_tensor();

    let deq = graph
        .dequantize_with_scale_tensor_and_zero_point_tensor(
            &w,
            &s,
            &zp,
            MPSDataType::Float16,
            None,
        )
        .ok_or(ExampleError::Dequantization)?;

    let matmul = graph.matmul(
        input_placeholder,
        &graph.transpose(&deq, &[1, 0], None),
        None,
    );

    let input_st = input_placeholder.shaped_type();
    let mut feeds: HashMap<&Tensor, &ShapedType> =
        HashMap::from_iter([(input_placeholder, &*input_st)]);

    let placeholder_pairs = tensors.collect_feeds();
    for (ph, st) in &placeholder_pairs {
        feeds.insert(&*ph, &*st);
    }

    let device = MPSDevice::with_device(&mtl_context.device);
    let executable = graph.compile(
        &device,
        &feeds,
        &[matmul.as_ref()],
        Some(compilation_descriptor),
    );

    Ok(executable)
}

fn make_tensor_option(
    graph: &Graph,
    load_type: TensorLoadType,
    array: &mut MetalArray,
    shape: &[usize],
    dtype: DataType,
    name: &str,
) -> TensorOption {
    match load_type {
        TensorLoadType::Baked => {
            let tensor_const = graph.constant_with_data(
                array.buffer(),
                &mps_shape(shape),
                dtype.into(),
            );
            TensorOption::Constant(tensor_const)
        },
        TensorLoadType::RuntimeLoaded => {
            let td = unsafe { array.to_mps_tensor_data() };
            let placeholder =
                graph.placeholder(dtype.into(), &mps_shape(shape), Some(name));
            TensorOption::Placeholder {
                placeholder,
                data: td,
            }
        },
    }
}

// ================== End helpers =====================

// cargo run --example quant_matmul_experiments -- --print-exec-dump --print-placement-analysis --dump-mpsgraphpackage

fn main() -> Result<(), ExampleError> {
    autoreleasepool(|_| {
        let args = Args::parse();

        let qm_descriptor = QuantizedMatmulDescriptor {
            weights: TensorLoadType::Baked,
            scales: TensorLoadType::RuntimeLoaded,
            zeros: TensorLoadType::Baked,
        };

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

        let mut scales_array =
            mtl_context.array_from_elem(&scales_shape, f16::from_f32(1.0));

        let mut zero_points_array = unsafe {
            mtl_context.array_uninitialized(&zero_points_shape, DataType::U4)
        };
        zero_points_array.buffer_mut().fill(0x00);

        let mut input_array = mtl_context
            .array_from_elem(&input_tensor_shape, f16::from_f32(1.0));

        let mut result_array = unsafe {
            mtl_context.array_uninitialized(&result_tensor_shape, DataType::F16)
        };

        // --- Graph ---

        let graph = Graph::new();

        let input_placeholder = graph.placeholder(
            MPSDataType::Float16,
            &Shape::from_dimensions(&input_shape),
            Some("input"),
        );

        // --- Quantized Matmul Sub-graph ---

        let qm_tensors = QuantizedMatmulTensors {
            weights: make_tensor_option(
                graph.as_ref(),
                qm_descriptor.weights,
                &mut weights_array,
                &weights_shape,
                DataType::U4,
                "weights_ph",
            ),
            scales: make_tensor_option(
                graph.as_ref(),
                qm_descriptor.scales,
                &mut scales_array,
                &scales_shape,
                DataType::F16,
                "scales_ph",
            ),
            zeros: make_tensor_option(
                graph.as_ref(),
                qm_descriptor.zeros,
                &mut zero_points_array,
                &zero_points_shape,
                DataType::U4,
                "zeros_ph",
            ),
        };

        let compilation_descriptor = make_compilation_descriptor(
            BlockDevice::Gpu,
            Optimization::Level1,
            OptimizationProfile::Performance,
            args.print_placement_analysis,
        );

        let executable = build_quantized_matmul(
            &graph,
            &qm_tensors,
            &input_placeholder,
            &mtl_context,
            &compilation_descriptor,
        )?;

        // --- Optional printing / dumping ---

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

        let input_tensor_option = TensorOption::Placeholder {
            placeholder: input_placeholder.clone(),
            data: input_td.clone(),
        };

        let tensor_options: [&TensorOption; 4] = [
            &qm_tensors.weights,
            &qm_tensors.scales,
            &qm_tensors.zeros,
            &input_tensor_option,
        ];

        let mut td_map: HashMap<Retained<Tensor>, Retained<TensorData>> =
            HashMap::new();
        for opt in &tensor_options {
            td_map.extend(opt.to_data_map());
        }

        let inputs: Vec<&TensorData> = match executable.feed_tensors() {
            Some(feed_tensors) => feed_tensors
                .iter()
                .map(|t| {
                    td_map.get(t).expect("TensorData for feed tensor not found")
                })
                .map(|td| &**td)
                .collect(),
            None => Vec::new(),
        };

        let outputs = [&*result_td];

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
    executable: &Executable,
    inputs: &[&TensorData],
    outputs: &[&TensorData],
) {
    let command_buffer =
        CommandBuffer::from_command_queue(&mtl_context.command_queue);
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
