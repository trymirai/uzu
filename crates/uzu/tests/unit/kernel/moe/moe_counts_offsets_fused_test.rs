use rand::{RngExt, SeedableRng, rngs::StdRng};
use uzu::{
    ArrayContextExt, ArrayElement,
    backends::{
        common::{Backend, Context, Encoder, Kernels, kernel::MoeCountsOffsetsFusedKernel},
        cpu::Cpu,
    },
};

use crate::uzu_test;

fn gen_random_topk_ids(
    t: usize,
    e: usize,
    k: usize,
    seed: u64,
) -> Vec<i32> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut topk_ids = vec![0i32; t * k];
    for ti in 0..t {
        for kk in 0..k {
            topk_ids[ti * k + kk] = rng.random_range(0..e as i32);
        }
    }
    topk_ids
}

fn get_output<B: Backend>(
    topk_ids: &[i32],
    t: usize,
    e: usize,
    k: usize,
) -> (Vec<u32>, u32, Vec<u32>) {
    let context = B::Context::new().expect("Failed to create Context");

    let kernel = <<B as Backend>::Kernels as Kernels>::MoeCountsOffsetsFusedKernel::new(&context)
        .expect("Failed to create MoeCountsOffsetsFusedKernel");

    let topk_ids_len = (t * k).max(1);
    let topk_ids_array = if topk_ids.is_empty() {
        context.create_array_uninitialized(&[topk_ids_len], i32::data_type(), "topk_ids")
    } else {
        context.create_array_from(&[topk_ids_len], topk_ids, "topk_ids")
    };
    let mut offsets = context.create_array_uninitialized(&[e + 1], u32::data_type(), "offsets").into_allocation();
    let mut sum_k = context.create_array_uninitialized(&[1], u32::data_type(), "sum_k").into_allocation();
    let num_tiles = e.div_ceil(512).max(1);
    let mut partials =
        context.create_array_uninitialized(&[num_tiles * 512], u32::data_type(), "partials").into_allocation();

    let mut encoder = Encoder::new(context.as_ref()).expect("Failed to create encoder");
    kernel.encode(
        topk_ids_array.allocation(),
        &mut offsets,
        &mut sum_k,
        &mut partials,
        t as u32,
        e as u32,
        k as u32,
        &mut encoder,
    );
    encoder.end_encoding().submit().wait_until_completed().unwrap();

    let offsets = crate::common::helpers::allocation_to_vec::<B, u32>(&offsets);
    let sum_k = crate::common::helpers::allocation_to_vec::<B, u32>(&sum_k)[0];
    let partials = crate::common::helpers::allocation_prefix_to_vec::<B, u32>(&partials, e);

    (offsets, sum_k, partials)
}

#[uzu_test]
fn test_counts_offsets_fused_parity_random() {
    let shapes = vec![(1usize, 4usize), (7, 16), (64, 64), (1024, 128)];
    let ks = vec![1usize, 2usize, 4usize];

    for &(t, e) in &shapes {
        for &k in &ks {
            if k > e {
                continue;
            }
            let topk_ids = gen_random_topk_ids(t, e, k, 1234);

            let (offsets_ref, sum_ref, counts_ref) = get_output::<Cpu>(&topk_ids, t, e, k);
            for_each_non_cpu_backend!(|B| {
                let (offsets, sum_k, partials) = get_output::<B>(&topk_ids, t, e, k);
                let backend_name = std::any::type_name::<B>();
                assert_eq!(offsets, offsets_ref, "offsets mismatch T={} E={} K={} backend={}", t, e, k, backend_name);
                assert_eq!(sum_k, sum_ref, "sum mismatch T={} E={} K={} backend={}", t, e, k, backend_name);
                assert_eq!(partials, counts_ref, "partials mismatch T={} E={} K={} backend={}", t, e, k, backend_name);
            });
        }
    }
}

#[uzu_test]
fn test_counts_offsets_fused_all_tokens_one_expert() {
    let (t, e, k) = (16usize, 8usize, 2usize);
    let topk_ids: Vec<i32> = vec![3i32; t * k];

    let mut expected_offsets = vec![0u32; e + 1];
    expected_offsets[4..].fill((t * k) as u32);

    for_each_backend!(|B| {
        let (offsets, sum_k, _) = get_output::<B>(&topk_ids, t, e, k);
        let backend_name = std::any::type_name::<B>();
        assert_eq!(offsets, expected_offsets, "offsets mismatch backend={}", backend_name);
        assert_eq!(sum_k, (t * k) as u32, "sum mismatch backend={}", backend_name);
    });
}

#[uzu_test]
fn test_counts_offsets_fused_zero_tokens() {
    let (t, e, k) = (0usize, 8usize, 2usize);
    let topk_ids: Vec<i32> = vec![];

    for_each_backend!(|B| {
        let (offsets, sum_k, _) = get_output::<B>(&topk_ids, t, e, k);
        let backend_name = std::any::type_name::<B>();
        assert!(offsets.iter().all(|&v| v == 0), "expected all-zero offsets for T=0, backend={}", backend_name,);
        assert_eq!(sum_k, 0, "sum should be 0 for T=0, backend={}", backend_name);
    });
}

#[uzu_test]
fn test_counts_offsets_fused_negative_ids_ignored() {
    let (t, e, k) = (8usize, 4usize, 2usize);
    // Mix of valid IDs and -1 (invalid/padding)
    let mut topk_ids = vec![-1i32; t * k];
    // Only set the first slot of each token, leave second as -1
    for ti in 0..t {
        topk_ids[ti * k] = (ti % e) as i32;
    }

    let (offsets_ref, sum_ref, counts_ref) = get_output::<Cpu>(&topk_ids, t, e, k);
    // sum should be t (only one valid id per token)
    assert_eq!(sum_ref, t as u32);

    for_each_non_cpu_backend!(|B| {
        let (offsets, sum_k, partials) = get_output::<B>(&topk_ids, t, e, k);
        let backend_name = std::any::type_name::<B>();
        assert_eq!(offsets, offsets_ref, "offsets mismatch backend={}", backend_name);
        assert_eq!(sum_k, sum_ref, "sum mismatch backend={}", backend_name);
        assert_eq!(partials, counts_ref, "partials mismatch backend={}", backend_name);
    });
}

#[uzu_test]
fn test_counts_offsets_fused_single_token() {
    let (t, e, k) = (1usize, 16usize, 1usize);
    let topk_ids = vec![7i32];

    let mut expected_offsets = vec![0u32; e + 1];
    // Expert 7 gets 1 token, so offsets[8..] = 1
    expected_offsets[8..].fill(1);

    for_each_backend!(|B| {
        let (offsets, sum_k, _) = get_output::<B>(&topk_ids, t, e, k);
        let backend_name = std::any::type_name::<B>();
        assert_eq!(offsets, expected_offsets, "offsets mismatch backend={}", backend_name);
        assert_eq!(sum_k, 1, "sum mismatch backend={}", backend_name);
    });
}

#[uzu_test]
fn test_counts_offsets_fused_uniform_distribution() {
    // Each expert gets exactly 1 token
    let e = 8usize;
    let (t, k) = (e, 1usize);
    let topk_ids: Vec<i32> = (0..e as i32).collect();
    let expected_offsets: Vec<u32> = (0..=e as u32).collect();

    for_each_backend!(|B| {
        let (offsets, sum_k, partials) = get_output::<B>(&topk_ids, t, e, k);
        let backend_name = std::any::type_name::<B>();
        assert_eq!(offsets, expected_offsets, "offsets mismatch backend={}", backend_name);
        assert_eq!(sum_k, e as u32, "sum mismatch backend={}", backend_name);
        assert!(partials.iter().all(|&v| v == 1), "each expert should have count=1 backend={}", backend_name,);
    });
}

#[uzu_test]
fn test_counts_offsets_fused_large_t() {
    // Stress test with many tokens
    let (t, e, k) = (2048usize, 128usize, 4usize);
    let topk_ids = gen_random_topk_ids(t, e, k, 42);

    let (offsets_ref, sum_ref, counts_ref) = get_output::<Cpu>(&topk_ids, t, e, k);
    for_each_non_cpu_backend!(|B| {
        let (offsets, sum_k, partials) = get_output::<B>(&topk_ids, t, e, k);
        let backend_name = std::any::type_name::<B>();
        assert_eq!(offsets, offsets_ref, "offsets mismatch backend={}", backend_name);
        assert_eq!(sum_k, sum_ref, "sum mismatch backend={}", backend_name);
        assert_eq!(partials, counts_ref, "partials mismatch backend={}", backend_name);
    });
}

#[uzu_test]
fn test_counts_offsets_fused_offsets_monotonic() {
    // Verify offsets are monotonically non-decreasing for various shapes
    let shapes = vec![(10, 8, 2), (100, 32, 4), (5, 128, 1)];

    for (t, e, k) in shapes {
        let topk_ids = gen_random_topk_ids(t, e, k, 99);

        for_each_backend!(|B| {
            let (offsets, sum_k, _) = get_output::<B>(&topk_ids, t, e, k);
            let backend_name = std::any::type_name::<B>();
            for i in 1..offsets.len() {
                assert!(
                    offsets[i] >= offsets[i - 1],
                    "offsets not monotonic at i={} ({} < {}) T={} E={} K={} backend={}",
                    i,
                    offsets[i],
                    offsets[i - 1],
                    t,
                    e,
                    k,
                    backend_name,
                );
            }
            assert_eq!(
                *offsets.last().unwrap(),
                sum_k,
                "last offset should equal sum_k T={} E={} K={} backend={}",
                t,
                e,
                k,
                backend_name,
            );
        });
    }
}
