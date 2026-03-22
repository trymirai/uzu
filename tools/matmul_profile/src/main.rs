use clap::{Parser, ValueEnum};
use metal::{MTLBuffer, MTLDeviceExt, MTLResourceOptions};
use uzu::{
    DataType,
    backends::{
        common::{
            Backend, Context, Encoder,
            kernel::matmul::{MatmulArguments, MatmulKernel, MatmulKernels},
        },
        metal::Metal,
    },
};

type Ctx = <Metal as Backend>::Context;

#[derive(Debug, Clone, Copy, ValueEnum)]
enum KernelChoice {
    Gemm,
    GemmMpp,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum DtypeChoice {
    Bf16,
    F16,
}

#[derive(Parser)]
#[command(about = "Profile a single matmul kernel dispatch")]
struct Args {
    #[arg(long, value_enum)]
    kernel: KernelChoice,

    #[arg(long, help = "Batch size (M dimension)")]
    m: usize,

    #[arg(long, help = "Input dimension (K dimension)")]
    k: usize,

    #[arg(long, help = "Output dimension (N dimension)")]
    n: usize,

    #[arg(long, value_enum, default_value = "bf16")]
    dtype: DtypeChoice,

    #[arg(long, default_value = "50")]
    iterations: usize,

    #[arg(long, default_value = "3")]
    warmup: usize,
}

fn fill_buffer_random(
    context: &Ctx,
    byte_count: usize,
) -> <Metal as Backend>::Buffer {
    let buffer = context
        .device
        .new_buffer(byte_count, MTLResourceOptions::STORAGE_MODE_SHARED)
        .expect("Failed to allocate buffer");
    let pointer = buffer.contents().as_ptr() as *mut u8;
    let slice = unsafe { std::slice::from_raw_parts_mut(pointer, byte_count) };
    for (i, byte) in slice.iter_mut().enumerate() {
        *byte = (i % 251) as u8;
    }
    buffer
}

fn run_iteration(
    context: &Ctx,
    kernel: &mut <<Metal as Backend>::Kernels as MatmulKernels>::MatmulKernel,
    kernel_choice: KernelChoice,
    a_buffer: &<Metal as Backend>::Buffer,
    b_buffer: &<Metal as Backend>::Buffer,
    d_buffer: &mut <Metal as Backend>::Buffer,
    m: i32,
    k: i32,
    n: i32,
) -> f64 {
    let mut encoder = Encoder::<Metal>::new(context).unwrap();

    let arguments = MatmulArguments {
        a: a_buffer,
        a_offset: 0,
        b: b_buffer,
        d: d_buffer,
        bias: None,
        batch: m,
        input_dim: k,
        output_dim: n,
        leading_dimension_a: k,
        leading_dimension_b: k,
        leading_dimension_d: n,
        transpose_b: true,
    };

    match kernel_choice {
        KernelChoice::Gemm => kernel.encode_gemm(context, arguments, &mut encoder).unwrap(),
        KernelChoice::GemmMpp => kernel.encode_gemm_mpp(context, arguments, &mut encoder).unwrap(),
    }

    let completed = encoder.end_encoding().submit().wait_until_completed().unwrap();
    completed.gpu_execution_time().map(|d| d.as_secs_f64() * 1000.0).unwrap_or(0.0)
}

fn main() {
    let args = Args::parse();

    let data_type = match args.dtype {
        DtypeChoice::Bf16 => DataType::BF16,
        DtypeChoice::F16 => DataType::F16,
    };

    let elem_size = data_type.size_in_bytes();
    let a_bytes = args.m * args.k * elem_size;
    let b_bytes = args.n * args.k * elem_size;
    let d_bytes = args.m * args.n * elem_size;

    eprintln!(
        "Profiling {:?} kernel: {}x{}x{} {:?}, {} warmup + {} iterations",
        args.kernel, args.m, args.k, args.n, args.dtype, args.warmup, args.iterations
    );

    let context = Ctx::new().expect("Metal context required");
    eprintln!("Device: {}", context.device.name());

    let mut kernel = <<Metal as Backend>::Kernels as MatmulKernels>::MatmulKernel::new(&context, data_type)
        .expect("Failed to create kernel");

    let a_buffer = fill_buffer_random(&context, a_bytes);
    let b_buffer = fill_buffer_random(&context, b_bytes);
    let mut d_buffer = fill_buffer_random(&context, d_bytes);

    for i in 0..args.warmup {
        let ms = run_iteration(
            &context,
            &mut kernel,
            args.kernel,
            &a_buffer,
            &b_buffer,
            &mut d_buffer,
            args.m as i32,
            args.k as i32,
            args.n as i32,
        );
        eprintln!("  warmup {}: {:.3} ms", i, ms);
    }

    let mut total_ms = 0.0;
    for i in 0..args.iterations {
        let ms = run_iteration(
            &context,
            &mut kernel,
            args.kernel,
            &a_buffer,
            &b_buffer,
            &mut d_buffer,
            args.m as i32,
            args.k as i32,
            args.n as i32,
        );
        total_ms += ms;
        if i < 5 || i == args.iterations - 1 {
            eprintln!("  iter {}: {:.3} ms", i, ms);
        }
    }

    let avg_ms = total_ms / args.iterations as f64;
    let flops = 2.0 * args.m as f64 * args.k as f64 * args.n as f64;
    let gflops = flops / (avg_ms / 1000.0) / 1e9;

    eprintln!("\nResults:");
    eprintln!("  Average: {:.3} ms", avg_ms);
    eprintln!("  GFLOPS:  {:.0}", gflops);
}
