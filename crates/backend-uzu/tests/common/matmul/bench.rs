use std::{
    env,
    path::PathBuf,
    sync::atomic::{AtomicBool, Ordering},
    time::{SystemTime, UNIX_EPOCH},
};

use backend_uzu::backends::common::{Backend, Context, Encoder};
use criterion::Bencher;

use crate::tests::env_vars;

static CAPTURE_TAKEN: AtomicBool = AtomicBool::new(false);

fn should_capture_benchmark(benchmark_path: &str) -> bool {
    env_vars::enabled(env_vars::UZU_CAPTURE_BENCH)
        && benchmark_path.starts_with("Metal/")
        && env::var(env_vars::UZU_CAPTURE_BENCH_FILTER).map_or(true, |filter| benchmark_path.contains(&filter))
        && !CAPTURE_TAKEN.swap(true, Ordering::AcqRel)
}

fn benchmark_capture_path() -> PathBuf {
    let timestamp = SystemTime::now().duration_since(UNIX_EPOCH).expect("system clock before Unix epoch").as_secs();
    env::var(env_vars::UZU_CAPTURE_BENCH_DIR)
        .map(PathBuf::from)
        .unwrap_or(env::current_dir().unwrap())
        .join(format!("uzu_bench-{timestamp}.gputrace"))
}

fn start_benchmark_capture<B: Backend>(
    context: &B::Context,
    benchmark_path: &str,
) -> bool {
    if !should_capture_benchmark(benchmark_path) {
        return false;
    }

    let path = benchmark_capture_path();
    context.start_capture(&path).expect("failed to start benchmark GPU capture");
    println!("GPU benchmark capture started for {benchmark_path}: {path:?}");
    true
}

pub fn iter_encode_loop<B: Backend, F>(
    context: &B::Context,
    bencher: &mut Bencher,
    mut encode: F,
) where
    F: FnMut(&mut Encoder<B>),
{
    iter_encode_loop_named(context, bencher, "unnamed_benchmark", |encoder| encode(encoder));
}

pub fn iter_encode_loop_named<B: Backend, F>(
    context: &B::Context,
    bencher: &mut Bencher,
    benchmark_path: &str,
    mut encode: F,
) where
    F: FnMut(&mut Encoder<B>),
{
    bencher.iter_custom(|n_iters| {
        let capture = start_benchmark_capture::<B>(context, benchmark_path);
        let mut encoder = Encoder::<B>::new(context).unwrap();
        for _ in 0..n_iters {
            encode(&mut encoder);
        }
        let completed = encoder.end_encoding().submit().wait_until_completed().unwrap();
        if capture {
            context.stop_capture().expect("failed to stop benchmark GPU capture");
        }
        completed.gpu_execution_time()
    });
}
