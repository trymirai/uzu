use clap::{Parser, ValueEnum};
use metal::{MTLBuffer, MTLDeviceExt, MTLResourceOptions};
use uzu::{
    DataType,
    backends::{
        common::{
            Backend, Context, Encoder,
            kernel::{
                ManualKernels,
                matmul::{MatmulArgumentC, MatmulArguments, MatmulKernel},
            },
        },
        metal::Metal,
    },
};

type Ctx = <Metal as Backend>::Context;

#[derive(Debug, Clone, Copy, ValueEnum)]
enum KernelChoice {
    Gemv,
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
    kernel: &mut <<Metal as Backend>::Kernels as ManualKernels>::MatmulKernel,
    kernel_choice: KernelChoice,
    a_buffer: &<Metal as Backend>::Buffer,
    b_buffer: &<Metal as Backend>::Buffer,
    d_buffer: &mut <Metal as Backend>::Buffer,
    m: u32,
    k: u32,
    n: u32,
) -> f64 {
    let mut encoder = Encoder::<Metal>::new(context).unwrap();

    let ab_scale = match kernel_choice {
        KernelChoice::Gemv | KernelChoice::GemmMpp => 1.0,
        KernelChoice::Gemm => 0.5,
    };

    let arguments = MatmulArguments {
        a: a_buffer,
        a_offset: 0,
        b: b_buffer,
        ab_scale,
        c: MatmulArgumentC::None,
        d: d_buffer,
        batch_dim: m,
        input_dim: k,
        output_dim: n,
    };

    kernel.encode(context, arguments, &mut encoder);

    let completed = encoder.end_encoding().submit().wait_until_completed().unwrap();
    completed.gpu_execution_time().map(|d| d.as_secs_f64() * 1000.0).unwrap_or(0.0)
}

fn main() {
    let args = Args::parse();
    let m = args.m as u32;
    let k = args.k as u32;
    let n = args.n as u32;

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

    unsafe {
        match args.kernel {
            // Keep gemv enabled for the requested M.
            KernelChoice::Gemv => std::env::set_var("UZU_GEMV_MAX_BATCH", args.m.to_string()),
            // Disable gemv fallback so we can target gemm/gemm_mpp behavior.
            KernelChoice::Gemm | KernelChoice::GemmMpp => std::env::set_var("UZU_GEMV_MAX_BATCH", "0"),
        }
    }

    let context = Ctx::new().expect("Metal context required");
    eprintln!("Device: {}", context.device.name());
    if matches!(args.kernel, KernelChoice::GemmMpp) && !context.device_capabilities().supports_mxu {
        eprintln!("Warning: this device has no MXU support; GemmMpp selection will fall back to Gemm");
    }

    let mut kernel = <<Metal as Backend>::Kernels as ManualKernels>::MatmulKernel::new(&context, data_type)
        .expect("Failed to create kernel");

    let a_buffer = fill_buffer_random(&context, a_bytes);
    let b_buffer = fill_buffer_random(&context, b_bytes);
    let mut d_buffer = fill_buffer_random(&context, d_bytes);

    for i in 0..args.warmup {
        let ms = run_iteration(&context, &mut kernel, args.kernel, &a_buffer, &b_buffer, &mut d_buffer, m, k, n);
        eprintln!("  warmup {}: {:.3} ms", i, ms);
    }

    let mut total_ms = 0.0;
    for i in 0..args.iterations {
        let ms = run_iteration(&context, &mut kernel, args.kernel, &a_buffer, &b_buffer, &mut d_buffer, m, k, n);
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
