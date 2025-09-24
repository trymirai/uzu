// Note: Module providing quantized matmul benchmark utilities originally prototyped
// in `examples/quant_matmul_experiments.rs`. The code is copied almost verbatim so
// that external crates can depend on these helpers without having to duplicate the
// logic themselves.

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
    DequantizationArguments, Device as MPSDevice, Executable,
    ExecutableExecutionDescriptor, ExecutableSerializationDescriptor, Graph,
    Optimization, OptimizationProfile, ShapedType, Tensor, TensorData,
};
use objc2::rc::Retained;
use thiserror::Error;

use crate::{
    Array, DataType, DeviceContext,
    backends::metal::{
        MTLContext, MetalArray,
        compilation_parameters::{BlockDevice, make_compilation_descriptor},
    },
    utils::storage::{NSSearchPathDirectory, root_dir},
};

// ================== Public error type =====================

#[derive(Debug, Error)]
pub enum ExampleError {
    #[error(transparent)]
    Metal(#[from] crate::backends::metal::error::MTLError),
    #[error("No Metal device available")]
    NoDevice,
    #[error("Graph dequantization failed")]
    Dequantization,
}

// ================== Public helper types =====================

#[derive(Debug, Clone, Copy)]
pub enum TensorLoadType {
    Baked,
    RuntimeLoaded,
}

#[derive(Debug, Clone, Copy)]
pub struct QuantizedMatmulDescriptor {
    pub weights: TensorLoadType,
    pub scales: TensorLoadType,
    pub zeros: TensorLoadType,
    pub transposed_shapes: bool,
    pub device: BlockDevice,
    pub optimization_level: Optimization,
    pub optimization_profile: OptimizationProfile,
}

// ================== CLI args used by the low-level API =====================

#[derive(Parser, Debug, Clone, Copy)]
#[command(
    name = "quant_matmul_experiments (library)",
    about = "Flags controlling extra instrumentation when running quantised matmul benchmark"
)]
pub struct Args {
    /// Print the compiled executable IR / debug dump
    #[arg(long = "print-exec-dump")]
    pub print_exec_dump: bool,

    /// Print ANE/GPU placement analysis during compilation
    #[arg(long = "print-placement-analysis")]
    pub print_placement_analysis: bool,

    /// Serialize the compiled MPSGraph executable to a .mpsgraphpackage file
    #[arg(long = "dump-mpsgraphpackage")]
    pub dump_mpsgraphpackage: bool,
}

impl Default for Args {
    fn default() -> Self {
        Self {
            print_exec_dump: false,
            print_placement_analysis: false,
            dump_mpsgraphpackage: false,
        }
    }
}

// ================== Internal tensor helpers =====================

#[derive(Debug, Clone)]
enum TensorOption {
    Constant(Retained<Tensor>),
    Placeholder {
        placeholder: Retained<Tensor>,
        data: Retained<TensorData>,
    },
}

impl TensorOption {
    #[inline]
    fn tensor(&self) -> Retained<Tensor> {
        match self {
            Self::Constant(t) => t.clone(),
            Self::Placeholder {
                placeholder,
                ..
            } => placeholder.clone(),
        }
    }

    fn feeds(&self) -> HashMap<Retained<Tensor>, Retained<ShapedType>> {
        match self {
            TensorOption::Constant(_) => HashMap::new(),
            TensorOption::Placeholder {
                placeholder,
                ..
            } => HashMap::from_iter([(
                placeholder.clone(),
                placeholder.shaped_type(),
            )]),
        }
    }

    fn data_feeds(&self) -> HashMap<Retained<Tensor>, Retained<TensorData>> {
        match self {
            TensorOption::Constant(_) => HashMap::new(),
            TensorOption::Placeholder {
                placeholder,
                data,
            } => HashMap::from_iter([(placeholder.clone(), data.clone())]),
        }
    }
}

#[derive(Debug, Clone)]
struct QMTensorOptions {
    input: TensorOption,
    weights: TensorOption,
    scales: TensorOption,
    zeros: TensorOption,
}

impl QMTensorOptions {
    fn feeds(&self) -> HashMap<Retained<Tensor>, Retained<ShapedType>> {
        HashMap::from_iter(
            [&self.input, &self.weights, &self.scales, &self.zeros]
                .iter()
                .flat_map(|opt| opt.feeds()),
        )
    }

    fn data_feeds(&self) -> HashMap<Retained<Tensor>, Retained<TensorData>> {
        HashMap::from_iter(
            [&self.input, &self.weights, &self.scales, &self.zeros]
                .iter()
                .flat_map(|opt| opt.data_feeds()),
        )
    }
}

// ================== Internal graph helpers =====================

fn build_quantized_matmul(
    graph: &Graph,
    tensor_options: &QMTensorOptions,
    mtl_context: &Rc<MTLContext>,
    compilation_descriptor: &CompilationDescriptor,
    transposed_shapes: bool,
) -> Result<Retained<Executable>, ExampleError> {
    let i = tensor_options.input.tensor();
    let w = tensor_options.weights.tensor();
    let s = tensor_options.scales.tensor();
    let zp = tensor_options.zeros.tensor();

    let dequantized_weights = graph.dequantize(
        &w,
        DequantizationArguments::ScaleTensorZeroPointTensorDataType {
            scale_tensor: &s,
            zero_point_tensor: &zp,
            data_type: MPSDataType::Float16,
        },
        None,
    );

    let matmul = if transposed_shapes {
        graph.matrix_multiplication(
            &i,
            &graph.transpose(&dequantized_weights, &[1, 0], None),
            None,
        )
    } else {
        graph.matrix_multiplication(
            &graph.transpose(&dequantized_weights, &[1, 0], None),
            &graph.transpose(&i, &[1, 0], None),
            None,
        )
    };

    let device = MPSDevice::with_device(&mtl_context.device);

    let retained_feeds = tensor_options.feeds();
    let feeds = retained_feeds.iter().map(|(t, st)| (&**t, &**st)).collect();

    let executable = graph.compile(
        &device,
        &feeds,
        &[&*matmul],
        None,
        Some(&compilation_descriptor),
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
            let tensor_const =
                graph.constant_with_data(array.buffer(), shape, dtype.into());
            TensorOption::Constant(tensor_const)
        },
        TensorLoadType::RuntimeLoaded => {
            let td = unsafe { array.to_mps_tensor_data() };
            let shape_isize: Vec<isize> =
                shape.iter().map(|d| *d as isize).collect();
            let placeholder =
                graph.placeholder(Some(&shape_isize), dtype.into(), Some(name));
            TensorOption::Placeholder {
                placeholder,
                data: td,
            }
        },
    }
}

// ================== Core public API =====================

/// Execute the quantised matmul benchmark once and return the average time per
/// iteration (in milliseconds) across an internal fixed number of runs.
///
/// The behaviour is identical to the original implementation in
/// `examples/quant_matmul_experiments.rs`.
#[allow(clippy::too_many_lines)]
pub fn run_quant_matmul(
    qm_descriptor: &QuantizedMatmulDescriptor,
    args: &Args,
) -> Result<f64, ExampleError> {
    // Constants that do not change between runs
    let suffix_length: usize = 128;
    let input_dim: usize = 2560;
    let output_dim: usize = 3072 * 2;
    let group_size: usize = 20;

    // Shapes that depend on descriptor options
    let weights_shape = if qm_descriptor.transposed_shapes {
        [output_dim, input_dim]
    } else {
        [input_dim, output_dim]
    };
    let scales_shape = if qm_descriptor.transposed_shapes {
        [output_dim, input_dim / group_size]
    } else {
        [input_dim / group_size, output_dim]
    };
    let zero_points_shape = scales_shape;
    let input_shape = [-1_isize, input_dim as isize];

    // Concrete shapes for actual buffer allocation at runtime
    let input_tensor_shape = [suffix_length, input_dim];
    let result_tensor_shape = [suffix_length, output_dim];

    // --- Device & context ---
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

    let mut input_array =
        mtl_context.array_from_elem(&input_tensor_shape, f16::from_f32(1.0));

    let mut result_array = unsafe {
        mtl_context.array_uninitialized(&result_tensor_shape, DataType::F16)
    };

    // --- Graph ---
    let graph = Graph::new();

    // --- Tensor options for quantized matmul ---
    let input_tensor_data = unsafe { input_array.to_mps_tensor_data() };
    let input_placeholder = graph.placeholder(
        Some(&input_shape),
        DataType::F16.into(),
        Some("input_ph"),
    );

    let tensor_options = QMTensorOptions {
        input: TensorOption::Placeholder {
            placeholder: input_placeholder,
            data: input_tensor_data,
        },
        weights: make_tensor_option(
            &graph,
            qm_descriptor.weights,
            &mut weights_array,
            &weights_shape,
            DataType::U4,
            "weights_ph",
        ),
        scales: make_tensor_option(
            &graph,
            qm_descriptor.scales,
            &mut scales_array,
            &scales_shape,
            DataType::F16,
            "scales_ph",
        ),
        zeros: make_tensor_option(
            &graph,
            qm_descriptor.zeros,
            &mut zero_points_array,
            &zero_points_shape,
            DataType::U4,
            "zeros_ph",
        ),
    };

    let compilation_descriptor = make_compilation_descriptor(
        qm_descriptor.device,
        qm_descriptor.optimization_level,
        qm_descriptor.optimization_profile,
        args.print_placement_analysis,
    );

    let executable = build_quantized_matmul(
        &graph,
        &tensor_options,
        &mtl_context,
        &compilation_descriptor,
        qm_descriptor.transposed_shapes,
    )?;

    // Optional printing / dumping per flags
    if args.print_exec_dump {
        executable.dump();
    }

    if args.dump_mpsgraphpackage {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("Time went backwards")
            .as_secs();
        let package_path = root_dir(NSSearchPathDirectory::Downloads).join(
            format!("quant_matmul_experiments-{timestamp}.mpsgraphpackage"),
        );
        let serialization_desc = ExecutableSerializationDescriptor::new();
        executable.serialize_to_graph_package(
            &package_path,
            Some(&serialization_desc),
        );
    }

    // --- Run & measure ---
    let result_tensor_data = unsafe { result_array.to_mps_tensor_data() };
    let retained_data_feeds = tensor_options.data_feeds();
    let inputs: Vec<&TensorData> = match executable.feed_tensors() {
        Some(feed_tensors) => feed_tensors
            .iter()
            .map(|t| {
                retained_data_feeds
                    .get(t)
                    .expect("TensorData for feed tensor not found")
            })
            .map(|td| &**td)
            .collect(),
        None => Vec::new(),
    };

    let outputs = [&*result_tensor_data];

    // Warm-up
    run_once(&mtl_context, &executable, &inputs, &outputs);

    const NUM_ITERATIONS: usize = 1000;
    let mut total_duration = 0.0f64; // milliseconds

    for _ in 0..NUM_ITERATIONS {
        let start = Instant::now();
        run_once(&mtl_context, &executable, &inputs, &outputs);
        total_duration += start.elapsed().as_secs_f64() * 1e3;
    }

    Ok(total_duration / NUM_ITERATIONS as f64)
}

/// Convenience wrapper that uses default (all-false) flags.
pub fn run_quant_matmul_simple(
    qm_descriptor: &QuantizedMatmulDescriptor
) -> Result<f64, ExampleError> {
    run_quant_matmul(qm_descriptor, &Args::default())
}

// ================== Internal util =====================

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
