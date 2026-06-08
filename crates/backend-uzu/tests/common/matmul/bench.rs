use std::{
    env,
    path::PathBuf,
    sync::atomic::{AtomicBool, Ordering},
    time::{SystemTime, UNIX_EPOCH},
};

use backend_uzu::backends::common::{Backend, Context, Encoder};
use criterion::Bencher;

use crate::common::env_var_enabled;

static CAPTURE_TAKEN: AtomicBool = AtomicBool::new(false);

fn capture_path(benchmark_path: &str) -> Option<PathBuf> {
    if !env_var_enabled("UZU_CAPTURE_BENCH")
        || !benchmark_path.starts_with("Metal/")
        || env::var("UZU_CAPTURE_BENCH_FILTER").is_ok_and(|filter| !benchmark_path.contains(&filter))
        || CAPTURE_TAKEN.load(Ordering::Acquire)
    {
        return None;
    }
    let timestamp = SystemTime::now().duration_since(UNIX_EPOCH).expect("system clock before Unix epoch").as_secs();
    Some(
        env::var("UZU_CAPTURE_BENCH_DIR")
            .map(PathBuf::from)
            .unwrap_or(env::current_dir().unwrap())
            .join(format!("uzu_bench-{timestamp}.gputrace")),
    )
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
        let capture = capture_path(benchmark_path).and_then(|path| {
            context.start_capture(&path).expect("failed to start benchmark GPU capture");
            if CAPTURE_TAKEN.compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire).is_err() {
                context.stop_capture().expect("failed to stop duplicate benchmark GPU capture");
                return None;
            }
            println!("GPU benchmark capture started for {benchmark_path}: {path:?}");
            Some(path)
        });
        let mut encoder = Encoder::<B>::new(context).unwrap();
        for _ in 0..n_iters {
            encode(&mut encoder);
        }
        let completed = encoder.end_encoding().submit().wait_until_completed().unwrap();
        if capture.is_some() {
            context.stop_capture().expect("failed to stop benchmark GPU capture");
        }
        completed.gpu_execution_time()
    });
}
