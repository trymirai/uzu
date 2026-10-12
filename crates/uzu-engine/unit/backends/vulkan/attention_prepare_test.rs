use std::{
    ops::Range,
    panic::{AssertUnwindSafe, catch_unwind},
    sync::Arc,
    time::Instant,
};

use half::bf16;
use uzu_engine_macros::uzu_test;

use super::{AttentionPrepareCase as Case, kernel_fixture::KernelFixture, round32};
use crate::{
    backends::vulkan::{AttentionPrepareVulkanKernel, Error, VkBuffer},
    data_type::DataType,
};

/// Every element of an output against the oracle: rotated NaN by class, everything else bit for bit (copies keep NaN
/// payloads, rotated results their signed zeros). Returns the number of mismatches after printing the first ones.
fn mismatches(
    expected: &[(bf16, bool)],
    actual: &[bf16],
    case: &str,
) -> usize {
    assert_eq!(expected.len(), actual.len(), "{case}: length");
    let mut count = 0;
    for (index, (&(expected, rotated), &actual)) in expected.iter().zip(actual).enumerate() {
        let same = match rotated && expected.is_nan() {
            true => actual.is_nan(),
            false => expected.to_bits() == actual.to_bits(),
        };
        if !same {
            count += 1;
            if count <= 5 {
                eprintln!(
                    "{case}: element {index} (rotated {rotated}): expected {:#06x}, got {:#06x}",
                    expected.to_bits(),
                    actual.to_bits()
                );
            }
        }
    }
    count
}

/// Runs every case on Vulkan in one command buffer and on the CPU; both must match the staged oracle exactly, its
/// stages being single correctly rounded operations.
fn check(
    fixture: &KernelFixture,
    cases: &[Case],
) {
    let outputs = Case::gpu(fixture, cases, fixture.encoding());
    let mut failures = 0;
    for (case, gpu) in cases.iter().zip(outputs) {
        let (expected, cpu) = (case.oracle(), case.cpu());
        for (slot, name) in ["queries", "keys", "values"].into_iter().enumerate() {
            failures += mismatches(&expected[slot], &cpu[slot], &format!("{} {name} CPU", case.label()));
            failures += mismatches(&expected[slot], &gpu[slot], &format!("{} {name} Vulkan", case.label()));
        }
    }
    assert_eq!(failures, 0, "elements differ from the staged oracle");
}

/// Query-only and KV-only layouts, no, partial and full rotation, heads past one tile of 128 and odd widths, and a batch
/// whose second row and table row duplicate the first; an empty batch records nothing.
#[uzu_test]
fn layouts_match_oracle() {
    let fixture = KernelFixture::new();
    let mut cases = vec![
        Case::new(3, None, 80, None, 3, 1),
        Case::new(2, None, 96, Some(64), 2, 2),
        Case::new(0, Some(2), 64, Some(64), 3, 3),
        Case::new(0, Some(1), 130, None, 2, 4),
        Case::new(4, Some(2), 128, Some(128), 3, 5),
        Case::new(4, Some(1), 160, Some(96), 2, 6),
        Case::new(2, Some(2), 200, None, 2, 7),
        Case::new(1, None, 1, None, 4, 8),
        Case::new(2, Some(1), 2, Some(2), 3, 9),
        Case::new(3, Some(3), 256, Some(130), 2, 10),
        Case::new(4, Some(2), 64, Some(32), 0, 11),
        Case::new(0, Some(2), 64, Some(64), 0, 12),
    ];
    let mut duplicated = Case::new(2, Some(1), 96, Some(64), 2, 13);
    let stride = duplicated.input_row_stride as usize;
    duplicated.qkvg.copy_within(..stride, stride);
    for table in [&mut duplicated.cosines, &mut duplicated.sines] {
        table.as_mut().unwrap().copy_within(..64, 64);
    }
    cases.push(duplicated);
    check(&fixture, &cases);
    fixture.assert_clean();
}

/// Every BF16 class in rotated and copied positions of query, key and value heads: quiet and signaling NaN payloads,
/// infinities, signed zeros, subnormals and the largest finite values, with special tables (zero, negative zero,
/// subnormal and infinite entries) giving NaN products and exact cancellations.
#[uzu_test]
fn special_values_match_oracle() {
    let fixture = KernelFixture::new();
    let bits =
        [0x7fc1u16, 0x7f81, 0xffc0, 0x7f80, 0xff80, 0x0000, 0x8000, 0x0001, 0x807f, 0x0080, 0x7f7f, 0xff7f, 0x3f80];
    let tables = [0.0f32, -0.0, f32::from_bits(1), f32::INFINITY, 1.0, -1.0, 0.5, f32::MAX];
    let mut cases = Vec::new();
    for (seed, rope_dim) in [(0, Some(16)), (5, Some(32)), (9, None)] {
        let mut case = Case::new(2, Some(1), 32, rope_dim, 3, seed);
        for (index, value) in case.qkvg.iter_mut().enumerate() {
            *value = bf16::from_bits(bits[(index * 7 + seed) % bits.len()]);
        }
        for table in [&mut case.cosines, &mut case.sines].into_iter().flatten() {
            for (index, value) in table.iter_mut().enumerate() {
                *value = tables[(index * 3 + seed + 1) % tables.len()];
            }
        }
        cases.push(case);
    }
    check(&fixture, &cases);
    fixture.assert_clean();
}

/// Cases where the canonical staging differs from a contracted or flushed one, each difference asserted before CPU and
/// Vulkan are checked: arbitrary-mantissa products that cancel, where a fused multiply-add of either product into the
/// sum keeps bits the CPU rounds away; products and sums below the smallest normal that survive as BF16 subnormals;
/// and a subnormal paired element negated for the first half.
#[uzu_test]
fn staging_is_canonical() {
    let fixture = KernelFixture::new();
    let two = |exponent: i32| 2f64.powi(exponent);
    // [input, paired, cos(first), sin(first), cos(second), sin(second)] per row of a two-element rotated head.
    let rows: [[f64; 6]; 4] = [
        [
            1.0 + two(-7),
            1.0 + two(-7),
            1.0 + 3.0 * two(-23),
            1.0 + 5.0 * two(-23),
            1.0 + 3.0 * two(-23),
            -(1.0 + 5.0 * two(-23)),
        ],
        [
            -1.5 - two(-7),
            1.5 + two(-7),
            1.0 + 7.0 * two(-23),
            -(1.0 + 3.0 * two(-23)),
            1.0 + 9.0 * two(-23),
            1.0 + 11.0 * two(-23),
        ],
        [two(-63), 0.0, two(-70), 1.0, 1.0, two(-71)],
        [1.5 * two(-60), 1.5 * two(-60), two(-50), two(-50) * (1.0 + two(-23)), 0.0, two(-60)],
    ];
    let mut case = Case::new(1, Some(1), 2, Some(2), rows.len() as u32, 1);
    let stride = case.input_row_stride as usize;
    for (batch, row) in rows.iter().enumerate() {
        // The query and key heads hold the row; the value head a subnormal copied as is.
        for head in 0..2 {
            case.qkvg[batch * stride + head * 2] = bf16::from_f64(row[0]);
            case.qkvg[batch * stride + head * 2 + 1] = bf16::from_f64(row[1]);
        }
        case.cosines.as_mut().unwrap()[batch * 2..batch * 2 + 2].copy_from_slice(&[row[2] as f32, row[4] as f32]);
        case.sines.as_mut().unwrap()[batch * 2..batch * 2 + 2].copy_from_slice(&[row[3] as f32, row[5] as f32]);
    }
    case.qkvg[3 * stride + 4] = bf16::from_bits(0x8003);
    // A subnormal paired element for the first half: -paired * 1 must stay a negative subnormal.
    case.qkvg[2 * stride + 1] = bf16::from_bits(0x0005);
    let staged = |input: f64, signed_paired: f64, cosine: f32, sine: f32| {
        bf16::from_f32((round32(input * f64::from(cosine)) + round32(signed_paired * f64::from(sine))) as f32)
    };
    let fused = |input: f64, signed_paired: f64, cosine: f32, sine: f32| {
        let (a, b) = (input * f64::from(cosine), signed_paired * f64::from(sine));
        [bf16::from_f32((a + round32(b)) as f32), bf16::from_f32((round32(a) + b) as f32)]
    };
    for batch in 0..2 {
        let (input, paired) = (case.qkvg[batch * stride].to_f64(), case.qkvg[batch * stride + 1].to_f64());
        let (cosines, sines) =
            (&case.cosines.as_ref().unwrap()[batch * 2..], &case.sines.as_ref().unwrap()[batch * 2..]);
        for (input, signed_paired, cosine, sine) in
            [(input, -paired, cosines[0], sines[0]), (paired, input, cosines[1], sines[1])]
        {
            let staged = staged(input, signed_paired, cosine, sine);
            for alternative in fused(input, signed_paired, cosine, sine) {
                assert_ne!(staged.to_bits(), alternative.to_bits(), "row {batch}: insensitive to contraction");
            }
        }
    }
    let expected = case.oracle();
    let subnormal = |value: bf16| value.to_bits() & 0x7f80 == 0 && value.to_bits() & 0x7f != 0;
    assert!(subnormal(expected[0][4].0) && subnormal(expected[0][5].0), "row 2 rotates to subnormals");
    assert!(expected[0][5].0.is_sign_negative() || expected[0][4].0.is_sign_negative(), "a negated subnormal pair");
    assert!(subnormal(expected[0][6].0), "row 3 cancels to a subnormal");
    check(&fixture, &[case]);
    fixture.assert_clean();
}

/// Construction admits only BF16 elements with FP32 tables. `encode` rejects every optional argument missing where
/// required or present where not, and each violated CPU precondition, also for empty batches and without overflowing
/// for huge head counts, before recording anything; the same command buffer then completes valid work and rejected
/// outputs stay untouched.
#[uzu_test]
fn rejects_invalid_contracts() {
    let fixture = KernelFixture::new();
    for types in [
        [DataType::F32, DataType::F32],
        [DataType::F16, DataType::F32],
        [DataType::BF16, DataType::BF16],
        [DataType::BF16, DataType::F16],
    ] {
        let result = AttentionPrepareVulkanKernel::new(&fixture.context, types[0], types[1], true, true);
        assert!(
            matches!(
                result,
                Err(Error::KernelVariant {
                    kernel: "AttentionPrepare",
                    ..
                })
            ),
            "{types:?}"
        );
    }
    let untouched = fixture.buffer(&[0u32; 2048]);
    let buffer = |index: u64| (&untouched, index * 1024..(index + 1) * 1024);
    let full = Case::new(2, Some(1), 64, Some(32), 2, 1);
    let kernel = full.vulkan_kernel(&fixture);
    let mut encoding = fixture.encoding();
    // (keys, values, cosines, sines, num_kv_heads, rope_dim, kv_token_offset): all present, then each flipped absent.
    let present = [true; 7];
    let invalid_presence = (0..7).map(|flip| {
        let mut arguments = present;
        arguments[flip] = false;
        (arguments, 2, 64, Some(1), Some(32), 384, "must be present exactly when")
    });
    // The last stride bound would overflow as 1 + 2 * (2^32 - 1) heads.
    let violated = [
        (present, 2, 0, Some(1), Some(32), 384, "head_dim > 0"),
        (present, 2, 64, Some(0), Some(32), 384, "num_kv_heads.unwrap_or(1) > 0"),
        (present, 2, 64, Some(1), Some(0), 384, "rope_dim.unwrap_or(2) > 0"),
        (present, 2, 64, Some(1), Some(31), 384, "rope_dim.unwrap_or(0).is_multiple_of(2)"),
        (present, 2, 64, Some(1), Some(66), 384, "rope_dim.unwrap_or(0) <= head_dim"),
        (present, 2, 64, Some(1), Some(32), 255, "input_row_stride / head_dim"),
        (present, 1, 2, Some(u32::MAX), Some(2), u32::MAX, "input_row_stride / head_dim"),
    ];
    let cases = invalid_presence.chain(violated);
    for ((arguments, q_heads, head_dim, kv_heads, rope_dim, stride, message), batch) in
        cases.flat_map(|case| [(case, 0), (case, 2)])
    {
        let [keys, values, cosines, sines, kv_present, rope_present, offset_present] = arguments;
        let result = catch_unwind(AssertUnwindSafe(|| unsafe {
            // SAFETY: never dispatched: an assertion fails before recording.
            kernel.encode(
                buffer(0),
                buffer(1),
                keys.then(|| buffer(2)),
                values.then(|| buffer(3)),
                cosines.then(|| buffer(4)),
                sines.then(|| buffer(5)),
                q_heads,
                kv_present.then_some(kv_heads.unwrap()),
                head_dim,
                rope_present.then_some(rope_dim.unwrap()),
                offset_present.then_some(0),
                stride,
                batch,
                &mut encoding,
            );
        }));
        let payload = result.expect_err("encode accepted");
        let text = payload.downcast_ref::<String>().expect("assertion message");
        assert!(text.contains(message), "{arguments:?} q {q_heads} head_dim {head_dim} batch {batch}: {text}");
    }
    // Without KV, a query head is required.
    let queries_only = Case::new(1, None, 64, None, 1, 1).vulkan_kernel(&fixture);
    for batch in [0, 2] {
        let result = catch_unwind(AssertUnwindSafe(|| unsafe {
            // SAFETY: as above.
            queries_only.encode(
                buffer(0),
                buffer(1),
                None,
                None,
                None,
                None,
                0,
                None,
                64,
                None,
                None,
                64,
                batch,
                &mut encoding,
            );
        }));
        let payload = result.expect_err("encode accepted");
        let text = payload.downcast_ref::<String>().expect("assertion message");
        assert!(text.contains("num_q_heads > 0 || has_kv"), "{text}");
    }
    let outputs = Case::gpu(&fixture, std::slice::from_ref(&full), encoding);
    // SAFETY: the completed command buffer recorded only the valid dispatch.
    assert!(unsafe { KernelFixture::read::<u32>(&untouched) }.iter().all(|&word| word == 0), "a rejected call wrote");
    let expected = full.oracle();
    for (slot, actual) in outputs[0].iter().enumerate() {
        assert_eq!(mismatches(&expected[slot], actual, "valid work"), 0, "valid work differs");
    }
    fixture.assert_clean();
}

/// Run alone: `cargo test ... attention_prepare_test::throughput -- --ignored --nocapture`. Construction cost, then
/// decode and prefill batches of model-shaped grouped-query heads of 128 and 64 with full rotation.
#[uzu_test]
#[ignore]
fn throughput() {
    let fixture = KernelFixture::new();
    let mut construction = (0..11)
        .map(|_| {
            let start = Instant::now();
            Case::new(1, Some(1), 2, Some(2), 1, 1).vulkan_kernel(&fixture);
            start.elapsed()
        })
        .collect::<Vec<_>>();
    let first = construction[0];
    construction.sort();
    eprintln!("AttentionPrepare construction: first {first:?}, median of 11 {:?}", construction[5]);
    for (q_heads, kv_heads, head_dim) in [(32, 8, 128), (16, 4, 64)] {
        for batch in [1u32, 128, 1024] {
            let case = Case::new(q_heads, Some(kv_heads), head_dim, Some(head_dim), batch, 1);
            let kernel = case.vulkan_kernel(&fixture);
            let [queries, keys, values] = case.initial().map(|values| fixture.buffer(&values));
            let (qkvg, cosines, sines) = (
                fixture.buffer(&case.qkvg),
                fixture.buffer(case.cosines.as_ref().unwrap()),
                fixture.buffer(case.sines.as_ref().unwrap()),
            );
            fn all(buffer: &Arc<VkBuffer>) -> (&Arc<VkBuffer>, Range<u64>) {
                (buffer, 0..buffer.size())
            }
            let (gpu, wall) = fixture.median_times(|encoding| unsafe {
                // SAFETY: every buffer holds the case's canonical payload; the outputs alias nothing.
                kernel.encode(
                    all(&qkvg),
                    all(&queries),
                    Some(all(&keys)),
                    Some(all(&values)),
                    Some(all(&cosines)),
                    Some(all(&sines)),
                    q_heads,
                    Some(kv_heads),
                    head_dim,
                    Some(head_dim),
                    case.kv_token_offset,
                    case.input_row_stride,
                    batch,
                    encoding,
                );
            });
            let elements = u64::from(batch * (q_heads + 2 * kv_heads) * head_dim);
            let bytes = 4 * elements + 8 * u64::from(batch * head_dim);
            eprintln!(
                "MEASURE AttentionPrepare q {q_heads} kv {kv_heads} head_dim {head_dim} batch {batch}: {bytes} B; median of 10 after 3 warm-up: GPU {gpu:?} ({:.1} GB/s), wall {wall:?}",
                bytes as f64 / gpu.as_secs_f64() / 1e9
            );
        }
    }
    fixture.assert_clean();
}
