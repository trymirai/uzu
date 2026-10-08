use std::collections::BTreeMap;

use uzu_engine_macros::uzu_test;

use super::{
    MatmulCase as Case, exact_witnesses, kernel_fixture::KernelFixture, overflow_and_nonfinite_witnesses,
    soft_cap_edges, soft_cap_follows_bias,
};
use crate::{
    backends::vulkan::{Error, GemmVulkanKernel},
    data_type::DataType,
};

/// The case's `check` through Gemm; without a soft cap, Vulkan also stores the CPU's bits, as its K fold is the
/// CPU's order.
fn check(
    fixture: &KernelFixture,
    label: &str,
    case: &Case,
    totals: &mut BTreeMap<String, [f64; 3]>,
) {
    let (cpu, vulkan) = case.check(fixture, label, Case::gemm, totals);
    if case.soft_cap.is_none() {
        KernelFixture::assert_bits(&cpu, &vulkan, &format!("{label} {} CPU bits", case.label()));
    }
}

fn check_all(
    label: &str,
    cases: impl IntoIterator<Item = Case>,
) {
    let fixture = KernelFixture::new();
    let mut totals = BTreeMap::new();
    for case in cases {
        check(&fixture, label, &case, &mut totals);
    }
    Case::report("Gemm", &totals);
    fixture.assert_clean();
}

/// Every triple under all 16 flag masks over 1 x 8 x 7, 3 x 33 x 129 and 65 x 63 x 17 (m x n x k), A starting 1 to 3
/// elements into its range.
#[uzu_test]
fn all_triples_and_masks() {
    let cases = itertools::iproduct!(Case::triples(), 0..16, [(1, 8, 7), (3, 33, 129), (65, 63, 17)])
        .map(|(types, mask, (m, n, k))| Case::new(types, m, n, k, mask, mask + m));
    check_all("masks", cases);
}

/// K 0, 1, 31, 32, 33 and 257 around the 16-element slice under no and every flag; M and N of 1, 63, 64, 65 and 129
/// around the 64-row tile; and 0.8b_qkv at m 16.
#[uzu_test]
fn shapes_and_tails() {
    let mut cases = Vec::new();
    let types = [[DataType::BF16; 3], [DataType::F32; 3], [DataType::BF16, DataType::F32, DataType::F32]];
    for (types, k, mask) in itertools::iproduct!(types, [0, 1, 31, 32, 33, 257], [0, 15]) {
        cases.push(Case::new(types, 65, 63, k, mask, k));
    }
    for (m, n) in itertools::iproduct!([1, 63, 64, 65, 129], [1, 63, 64, 65, 129]) {
        cases.push(Case::new(types[(m + n) as usize % 3], m, n, 33, 15, m + n));
    }
    for types in [types[0], types[2]] {
        cases.push(Case::new(types, 16, 3072, 1024, Case::SCALE | Case::BIAS, 16));
    }
    check_all("shapes", cases);
}

/// No rows or no columns record nothing, leaving D's guards and inputs unchanged; K = 0 is the epilogue on +0.
#[uzu_test]
fn zero_dispatch_records_nothing() {
    let fixture = KernelFixture::new();
    for case in [Case::new([DataType::BF16; 3], 0, 5, 7, 15, 1), Case::new([DataType::F32; 3], 3, 0, 7, 15, 2)] {
        assert!(case.gemm(&fixture, 1).is_empty(), "{}", case.label());
    }
    for types in Case::triples() {
        Case::new(types, 2, 3, 0, 0, 4).exact(&fixture, "K 0", Case::gemm, &[0.0; 6]);
    }
    fixture.assert_clean();
}

/// Two accumulating dispatches in one command buffer over exact integers across two tiles each way: D + 2 A Bᵀ bit for
/// bit.
#[uzu_test]
fn successive_accumulate() {
    let fixture = KernelFixture::new();
    let mut case = Case::new([DataType::BF16, DataType::BF16, DataType::F32], 70, 67, 40, Case::ACCUMULATE, 6);
    case.b = (0..case.b.len()).map(|index| (index % 7) as f32 - 3.0).collect();
    case.a = (0..case.a.len()).map(|index| (index % 5) as f32 - 2.0).collect();
    case.d = (0..case.d.len()).map(|index| index as f32).collect();
    let (n, k) = (case.n as usize, case.k as usize);
    let expected = (0..case.d.len())
        .map(|index| {
            let (row, column) = (index / n, index % n);
            let dot = case.a[row * k..][..k].iter().zip(&case.b[column * k..][..k]).map(|(x, w)| x * w).sum::<f32>();
            case.d[index] + 2.0 * dot
        })
        .collect::<Vec<_>>();
    KernelFixture::assert_bits(&expected, &case.cpu(2, false), "CPU chain");
    KernelFixture::assert_bits(&expected, &case.gemm(&fixture, 2), "Vulkan chain");
    fixture.assert_clean();
}

/// Gemv's exact witnesses through Gemm, then one tile mixing every class: rows of A 0 ordinary, 1 zero but 2^-140 (BF16
/// 2^-133) at K 9, 2 ordinary with +inf at K 13, 3 zero and 4 ordinary; B rows (columns) 0 ordinary, 1 ordinary with
/// 2^-140 (BF16 2^-133) at K 13, 2 zero but 2^120 at K 9, 3 ordinary with 0 at K 13, 4 and 5 ordinary. Row 1 against
/// column 2 is the productive subnormal product alone, 2^-20 (2^-13 in BF16); row 2 is replayed, against column 1 inf x
/// subnormal infinite, against column 3 inf x 0 NaN; rows 0 and 4 against column 0 are ordinary; the decisive operands
/// sit in the slice's second to fourth loaders, never the first. Bounds, CPU bits and these classes on both.
#[uzu_test]
fn exact_witnesses_gemm() {
    exact_witnesses(Case::gemm);
    let fixture = KernelFixture::new();
    let mut totals = BTreeMap::new();
    for (types, tiny, productive) in
        [([DataType::F32; 3], 2f32.powi(-140), 2f32.powi(-20)), ([DataType::BF16; 3], 2f32.powi(-133), 2f32.powi(-13))]
    {
        let mut tile = Case::new(types, 5, 6, 40, 0, 3);
        let k = 40;
        tile.a = (0..5 * k).map(|index| Case::stored(((index % 11) as f32 - 5.0) * 0.25, types[1])).collect();
        tile.b = (0..6 * k).map(|index| Case::stored(((index % 13) as f32 - 6.0) * 0.125, types[0])).collect();
        tile.a[k..2 * k].fill(0.0);
        tile.a[3 * k..4 * k].fill(0.0);
        tile.b[2 * k..3 * k].fill(0.0);
        (tile.a[k + 9], tile.a[2 * k + 13]) = (tiny, f32::INFINITY);
        (tile.b[k + 13], tile.b[2 * k + 9], tile.b[3 * k + 13]) = (tiny, 2f32.powi(120), 0.0);
        let (cpu, vulkan) = tile.check(&fixture, "mixed tile", Case::gemm, &mut totals);
        for (backend, values) in [("CPU", &cpu), ("Vulkan", &vulkan)] {
            let (productive_value, infinite, nan) = (values[6 + 2], values[2 * 6 + 1], values[2 * 6 + 3]);
            assert_eq!(productive_value.to_bits(), productive.to_bits(), "{backend} productive {productive_value:e}");
            assert!(
                infinite == f32::INFINITY && nan.is_nan(),
                "{backend} inf x subnormal {infinite:e}, inf x 0 {nan:e}"
            );
            assert!(values[0].is_finite() && values[4 * 6].is_finite() && values[3 * 6] == 0.0, "{backend} {values:?}");
        }
        KernelFixture::assert_bits(&cpu, &vulkan, &format!("mixed tile {:?} CPU bits", types));
    }
    Case::report("Gemm", &totals);
    fixture.assert_clean();
}

#[uzu_test]
fn soft_cap_follows_bias_gemm() {
    soft_cap_follows_bias("Gemm", Case::gemm);
}

#[uzu_test]
fn soft_cap_edges_gemm() {
    soft_cap_edges("Gemm", Case::gemm);
}

#[uzu_test]
fn overflow_and_nonfinite_witnesses_gemm() {
    overflow_and_nonfinite_witnesses("Gemm", Case::gemm);
}

/// Construction rejects every data type but F32 and BF16.
#[uzu_test]
fn rejects_invalid_types() {
    let fixture = KernelFixture::new();
    for [a, b, d] in [[DataType::F16, DataType::F32, DataType::F32], [DataType::F32, DataType::F32, DataType::F16]] {
        let kernel = GemmVulkanKernel::new(&fixture.context, a, b, d, false, false, false, false);
        assert!(matches!(kernel, Err(Error::KernelVariant { .. })), "{:?} accepted", [a, b, d]);
    }
    fixture.assert_clean();
}

/// Gemm against the accepted Gemv on the same buffers: 0.8b_qkv (k 1024, n 3072) and 2b_up (k 2048, n 12288) at rows
/// `ms`, in BF16, F32, and BF16 weights with F32 input and output, hashed data and no flags; with `extras` an irregular
/// 77 x 3001 x 1001 tail and 2b_up at m 64 mixing careful outputs (A row 0 scaled by 2^-110) and replayed ones (a NaN in
/// B row 5) in every tile. Each case is first checked against the CPU, its bounds and the CPU's bits through Gemm and
/// against its bounds through Gemv, then timed in 4 interleaved Gemv, Gemm pairs, each the median GPU and wall time of
/// 10 submissions after 3 warm-up ones. Bytes count A, B and D once, FLOPs 2 m n k: logical rates, not DRAM traffic.
fn throughput(
    ms: &[u32],
    extras: bool,
) {
    let fixture = KernelFixture::new();
    let mut cases = Vec::new();
    let types = [[DataType::BF16; 3], [DataType::F32; 3], [DataType::BF16, DataType::F32, DataType::F32]];
    for ((label, k, n), &m, types) in
        itertools::iproduct!([("0.8b_qkv", 1024, 3072), ("2b_up", 2048, 12288)], ms, types)
    {
        cases.push((label, Case::new(types, m, n, k, 0, m)));
    }
    if extras {
        cases.push(("tail", Case::new(types[0], 77, 3001, 1001, 0, 77)));
        let mut mixed = Case::new(types[0], 64, 12288, 2048, 0, 64);
        mixed.a[..2048].iter_mut().for_each(|value| *value = Case::stored(*value * 2f32.powi(-110), DataType::BF16));
        mixed.b[5 * 2048] = f32::NAN;
        cases.push(("2b_up mixed", mixed));
    }
    let mut totals = BTreeMap::new();
    let mut gemv_totals = BTreeMap::new();
    for (label, case) in &cases {
        check(&fixture, "throughput", case, &mut totals);
        case.check(&fixture, "throughput", Case::gemv, &mut gemv_totals);
        let (gemv, gemm) = (case.gemv_kernel(&fixture), case.gemm_kernel(&fixture));
        let buffers = [(&case.b, 0), (&case.a, 1), (&case.d, 2)]
            .map(|(values, index)| fixture.buffer(&Case::bytes(values, case.types[index])));
        let ranges = || buffers.each_ref().map(|buffer| (buffer, 0..buffer.size()));
        let (m, n, k) = (u64::from(case.m), u64::from(case.n), u64::from(case.k));
        let [b, a, d] = case.types.map(|data_type| data_type.size_in_bytes() as u64);
        let (bytes, flops) = (n * k * b + m * k * a + m * n * d, 2 * m * n * k);
        for pair in 0..4 {
            // SAFETY: whole buffers of the case's B, A and D; D aliases nothing.
            let times = [
                fixture.median_times(|encoding| unsafe { case.encode_gemv(&gemv, ranges(), None, None, encoding) }),
                fixture.median_times(|encoding| unsafe { case.encode_gemm(&gemm, ranges(), None, encoding) }),
            ];
            let [[gemv_gpu, gemv_wall], [gemm_gpu, gemm_wall]] =
                times.map(|(gpu, wall)| [gpu, wall].map(|time| time.as_secs_f64() * 1e6));
            eprintln!(
                "MEASURE {label} m {m} n {n} k {k} {:?} pair {pair}: Gemv GPU {gemv_gpu:.1} us wall {gemv_wall:.1} us, Gemm \
                 GPU {gemm_gpu:.1} us wall {gemm_wall:.1} us, Gemm {:.1} GB/s {:.1} GFLOP/s, {bytes} B {flops} FLOP",
                case.types,
                bytes as f64 / gemm_gpu / 1e3,
                flops as f64 / gemm_gpu / 1e3,
            );
        }
    }
    Case::report("Gemm", &totals);
    Case::report("Gemv", &gemv_totals);
    fixture.assert_clean();
}

/// Run alone with `--ignored --nocapture`, as the three throughput_m tests.
#[uzu_test]
#[ignore]
fn throughput_m16() {
    throughput(&[16], false);
}

#[uzu_test]
#[ignore]
fn throughput_m64() {
    throughput(&[64], false);
}

#[uzu_test]
#[ignore]
fn throughput_m128() {
    throughput(&[128], true);
}
