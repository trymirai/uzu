use backend_uzu::backends::common::{Backend, Encoder};
use criterion::Bencher;

pub fn iter_encode_loop<B: Backend, F>(
    context: &B::Context,
    bencher: &mut Bencher,
    mut encode: F,
) where
    F: FnMut(&mut Encoder<B>),
{
    bencher.iter_custom(|n_iters| {
        let mut encoder = Encoder::<B>::new(context).unwrap();
        for _ in 0..n_iters {
            encode(&mut encoder);
        }
        encoder.end_encoding().submit().wait_until_completed().unwrap().gpu_execution_time()
    });
}
