use std::{
    any::TypeId,
    fmt::{Debug, Display},
    ops::Range,
    panic::{AssertUnwindSafe, catch_unwind},
};

use half::{bf16, f16};
use num_traits::Float;
use uzu_engine_macros::uzu_test;

use crate::{
    array::ArrayElement,
    backends::{
        common::{
            Backend, CommandBufferEncoding, CommandBufferExecutable, CommandBufferPending, Context, Kernels,
            gpu_types::ActivationType,
            kernel::{SSDPrefill64Kernel, SSDPrefillKernel},
        },
        cpu::Cpu,
    },
    data_type::DataType,
    tests::{
        assert::assert_eq_float,
        helpers::{buffer_prefix_to_vec, create_buffer_with_data, for_each_backend},
    },
};

struct Input<T: ArrayElement + Float> {
    x: Box<[T]>,
    dt: Box<[T]>,
    b: Box<[T]>,
    c: Box<[T]>,
    d: Box<[T]>,
    z: Box<[T]>,
    state: Box<[T]>,
    suffix_len: usize,
    num_heads: usize,
    head_dim: usize,
    state_dim: usize,
    group_size: u32,
    x_strides: [usize; 3],
    dt_strides: [usize; 2],
    cb_strides: [usize; 3],
    state_strides: [usize; 3],
}

struct Output<T: ArrayElement + Float> {
    y: Vec<T>,
    state: Vec<T>,
}

/// The value of every y element no token writes.
fn sentinel<T: Float>() -> T {
    T::from(-7.0).unwrap()
}

/// Contiguous [x; dt; B and C; state] strides of [Q, H, Dh, N, group_size], whose last group may be partial.
fn contiguous([_, heads, head_dim, state_dim, group_size]: [usize; 5]) -> [usize; 11] {
    let groups = heads.div_ceil(group_size.max(1));
    [heads * head_dim, head_dim, 1, heads, 1, groups * state_dim, state_dim, 1, head_dim * state_dim, state_dim, 1]
}

/// [Q, H, Dh, N, group_size] at `strides`, [x, dt, B, C, d, z, state] holding `value(array, element)` over the spans
/// they address; one element where nothing is addressed (no work, or dt, B, C and the state with N 0).
fn layout_input<T: ArrayElement + Float>(
    [suffix_len, num_heads, head_dim, state_dim, group_size]: [usize; 5],
    s: [usize; 11],
    value: impl Fn(usize, usize) -> f64,
) -> Input<T> {
    let work = suffix_len > 0 && num_heads > 0 && head_dim > 0;
    let span = |extents: &[(usize, usize)]| match work && extents.iter().all(|&(extent, _)| extent > 0) {
        true => extents.iter().map(|&(extent, stride)| (extent - 1) * stride).sum::<usize>() + 1,
        false => 1,
    };
    let groups = num_heads.div_ceil(group_size.max(1));
    let x = span(&[(suffix_len, s[0]), (num_heads, s[1]), (head_dim, s[2])]);
    let cb = span(&[(suffix_len, s[5]), (groups, s[6]), (state_dim, s[7])]);
    let dt = span(&[(suffix_len, s[3]), (num_heads, s[4]), (state_dim, 0)]);
    let lens =
        [x, dt, cb, cb, span(&[(num_heads, 1)]), x, span(&[(num_heads, s[8]), (head_dim, s[9]), (state_dim, s[10])])];
    let [x, dt, b, c, d, z, state] = std::array::from_fn(|array| {
        (0..lens[array]).map(|element| T::from(value(array, element)).unwrap()).collect::<Box<[T]>>()
    });
    Input {
        x,
        dt,
        b,
        c,
        d,
        z,
        state,
        suffix_len,
        num_heads,
        head_dim,
        state_dim,
        group_size: group_size as u32,
        x_strides: [s[0], s[1], s[2]],
        dt_strides: [s[3], s[4]],
        cb_strides: [s[5], s[6], s[7]],
        state_strides: [s[8], s[9], s[10]],
    }
}

/// The fixture's periodic values of [x, dt, B, C, d, z, state].
fn pattern(
    array: usize,
    element: usize,
) -> f64 {
    let period = [17, 13, 11, 19, 3, 23, 29][array];
    (element % period) as f64 * [0.01, 0.2, 0.02, 0.01, 0.05, 0.02, 0.03][array]
        + [-0.05, -1.5, -0.05, -0.02, -0.05, -0.1, -0.4][array]
}

fn get_input<T: ArrayElement + Float>(
    suffix_len: usize,
    num_heads: usize,
    head_dim: usize,
    state_dim: usize,
    group_size: u32,
) -> Input<T> {
    let dims = [suffix_len, num_heads, head_dim, state_dim, group_size as usize];
    layout_input(dims, contiguous(dims), pattern)
}

/// One dispatch of SSDPrefill, or SSDPrefill64 when `special64`, with y holding sentinels over x's span; panics if an
/// input buffer changes.
fn get_output<B: Backend, T: ArrayElement + Float + Debug>(
    input: &Input<T>,
    special64: bool,
) -> Output<T> {
    let context = B::Context::new().expect("Failed to create Context");
    let inputs = [&input.x, &input.dt, &input.b, &input.c, &input.d, &input.z];
    let [x, dt, b, c, d, z] = inputs.map(|data| create_buffer_with_data::<B, T>(&context, data));
    let mut state = create_buffer_with_data::<B, T>(&context, &input.state);
    let mut y = create_buffer_with_data::<B, T>(&context, &vec![sentinel::<T>(); input.x.len()]);

    let mut command_buffer = context.create_command_buffer(None, None).expect("Failed to create command buffer");
    // The generic kernel takes the state size at construction, SSDPrefill64 at encode.
    macro_rules! encode {
        ($kernel:ident, [$($new:expr),*], [$($state:expr),*]) => {
            <<B as Backend>::Kernels as Kernels>::$kernel::new(&context, T::data_type() $(, $new)*)
                .expect("Failed to create the SSD prefill kernel")
                .encode(
                    &x,
                    &dt,
                    &b,
                    &c,
                    &d,
                    &z,
                    &mut state,
                    &mut y,
                    input.suffix_len as u32,
                    input.group_size,
                    $($state,)*
                    &input.x_strides.map(|s| s as u32),
                    &input.dt_strides.map(|s| s as u32),
                    &input.cb_strides.map(|s| s as u32),
                    &input.state_strides.map(|s| s as u32),
                    input.num_heads as u32,
                    input.head_dim as u32,
                    &mut command_buffer,
                )
        };
    }
    match special64 {
        true => encode!(SSDPrefill64Kernel, [], [input.state_dim as u32]),
        false => encode!(SSDPrefillKernel, [input.state_dim as u32], []),
    }
    command_buffer.end_encoding().submit().wait_until_completed().expect("Failed to wait command buffer");

    for (index, (buffer, data)) in [&x, &dt, &b, &c, &d, &z].into_iter().zip(inputs).enumerate() {
        assert_same_bytes(&buffer_prefix_to_vec::<B, T>(buffer, data.len()), data, &format!("input {index}"));
    }
    Output {
        y: buffer_prefix_to_vec(&y, input.x.len()),
        state: buffer_prefix_to_vec(&state, input.state.len()),
    }
}

fn cpu_output<T: ArrayElement + Float + Debug>(input: &Input<T>) -> Output<T> {
    get_output::<Cpu, T>(input, false)
}

/// The contract in scalar f32, independent of the kernels: each state row widened once and stored as T once; per token
/// decay = exp(-softplus(dt_raw)), s = s decay + B x in ascending order, the dot from +0 over s C, and
/// y = T((dot + d x) T(SiLU(z))). Unwritten y elements keep the sentinel; nothing unaddressed is read.
fn reference_output<T: ArrayElement + Float>(input: &Input<T>) -> Output<T> {
    let f = |value: T| value.to_f32().unwrap();
    let mut y = vec![sentinel::<T>(); input.x.len()];
    let mut state = input.state.to_vec();
    let (n, xs, cbs, ss) = (input.state_dim, input.x_strides, input.cb_strides, input.state_strides);
    // Without tokens no head is visited, so nothing is read.
    for h in 0..input.num_heads * usize::from(input.suffix_len > 0) {
        for e in 0..input.head_dim {
            let row = (0..n).map(|i| h * ss[0] + e * ss[1] + i * ss[2]).collect::<Vec<_>>();
            let mut s = row.iter().map(|&index| f(state[index])).collect::<Vec<_>>();
            for t in 0..input.suffix_len {
                let x_index = t * xs[0] + h * xs[1] + e * xs[2];
                let x = f(input.x[x_index]);
                let mut dot = 0.0f32;
                if n > 0 {
                    let dt_raw = f(input.dt[t * input.dt_strides[0] + h * input.dt_strides[1]]);
                    let decay = (-ActivationType::SOFTPLUS.activate(dt_raw)).exp();
                    let cb = t * cbs[0] + h / input.group_size.max(1) as usize * cbs[1];
                    for i in 0..n {
                        s[i] = s[i] * decay + f(input.b[cb + i * cbs[2]]) * x;
                        dot += s[i] * f(input.c[cb + i * cbs[2]]);
                    }
                }
                let gate = f(ActivationType::SILU.activate(input.z[x_index]));
                y[x_index] = T::from((dot + f(input.d[h]) * x) * gate).unwrap();
            }
            for (&index, &value) in row.iter().zip(&s) {
                state[index] = T::from(value).unwrap();
            }
        }
    }
    Output {
        y,
        state,
    }
}

/// `actual` has exactly the bytes of `expected`.
fn assert_same_bytes<T: ArrayElement + Debug>(
    actual: &[T],
    expected: &[T],
    label: &str,
) {
    let bytes = |values: &[T]| bytemuck::cast_slice::<T, u8>(values).to_vec();
    assert!(bytes(actual) == bytes(expected), "{label}: {actual:?} is not {expected:?}");
}

/// `actual` has the bits of `expected`, except that an expected NaN asks only for a NaN: the contract fixes classes, not
/// NaN payloads.
fn assert_bits<T: ArrayElement + Float + Debug>(
    expected: &[T],
    actual: &[T],
    label: &str,
) {
    assert_eq!(expected.len(), actual.len(), "{label}: length");
    for (index, (&expected, &actual)) in expected.iter().zip(actual).enumerate() {
        let same = match expected.is_nan() {
            true => actual.is_nan(),
            false => bytemuck::bytes_of(&expected) == bytemuck::bytes_of(&actual),
        };
        assert!(same, "{label}: element {index} is {actual:?}, not {expected:?}");
    }
}

fn test_internal<T: ArrayElement + Float + Debug + Display>(
    input: &Input<T>,
    expected: &Output<T>,
    label: &str,
) {
    let eps = if matches!(T::data_type(), DataType::F16 | DataType::BF16) {
        2e-2f32
    } else {
        5e-5
    };

    for_each_backend!(|B| {
        let output = get_output::<B, T>(input, false);
        let backend_name = std::any::type_name::<B>();
        let type_name = std::any::type_name::<T>();
        let y_label = format!("SSDPrefill y {backend_name} {label} (type={type_name})");
        let state_label = format!("SSDPrefill state {backend_name} {label} (type={type_name})");

        if TypeId::of::<B>() == TypeId::of::<Cpu>() {
            assert_bits(&expected.y, &output.y, &y_label);
            assert_bits(&expected.state, &output.state, &state_label);
        } else {
            assert_eq_float::<T>(&expected.y, &output.y, eps, &y_label);
            assert_eq_float::<T>(&expected.state, &output.state, eps, &state_label);
        }
    });
}

// --- test shapes ---

fn test_shape(
    suffix_len: usize,
    num_heads: usize,
    head_dim: usize,
    state_dim: usize,
    group_size: u32,
    label: &str,
) {
    fn run<T: ArrayElement + Float + Debug + Display>(
        suffix_len: usize,
        num_heads: usize,
        head_dim: usize,
        state_dim: usize,
        group_size: u32,
        label: &str,
    ) {
        let input = get_input::<T>(suffix_len, num_heads, head_dim, state_dim, group_size);
        let expected = reference_output(&input);
        test_internal(&input, &expected, label);
    }
    run::<f32>(suffix_len, num_heads, head_dim, state_dim, group_size, label);
    run::<f16>(suffix_len, num_heads, head_dim, state_dim, group_size, label);
    run::<bf16>(suffix_len, num_heads, head_dim, state_dim, group_size, label);
}

// --- Prefill ---
#[uzu_test]
fn test_prefill_basic() {
    test_shape(512, 32, 64, 64, 1, "prefill_basic");
}

#[uzu_test]
fn test_prefill_small() {
    test_shape(4, 4, 4, 8, 1, "prefill_small");
}

#[uzu_test]
fn test_prefill_minimal() {
    test_shape(1, 1, 1, 1, 1, "prefill_minimal");
}

#[uzu_test]
fn test_prefill_multi_group() {
    test_shape(8, 8, 4, 16, 4, "prefill_multi_group");
}

#[uzu_test]
fn test_prefill_group_per_head() {
    test_shape(8, 4, 4, 8, 1, "prefill_group_per_head");
}

// --- CPU contract witnesses ---

/// The tokens `tokens` of a contiguous input, starting from `state`.
fn token_range<T: ArrayElement + Float>(
    input: &Input<T>,
    tokens: Range<usize>,
    state: &[T],
) -> Input<T> {
    let (xs, dts, cbs, ss) = (input.x_strides, input.dt_strides, input.cb_strides, input.state_strides);
    let dims = [tokens.len(), input.num_heads, input.head_dim, input.state_dim, input.group_size as usize];
    let strides = [xs[0], xs[1], xs[2], dts[0], dts[1], cbs[0], cbs[1], cbs[2], ss[0], ss[1], ss[2]];
    let arrays = [&input.x[..], &input.dt, &input.b, &input.c, &input.d, &input.z, state];
    let starts = [xs[0], dts[0], cbs[0], cbs[0], 0, xs[0], 0].map(|stride| tokens.start * stride);
    layout_input(dims, strides, |array, element| arrays[array][starts[array] + element].to_f64().unwrap())
}

/// `run` over a contiguous input in two dispatches split at token `split`, the second from the first's state.
fn chained<T: ArrayElement + Float>(
    input: &Input<T>,
    split: usize,
    run: impl Fn(&Input<T>) -> Output<T>,
) -> Output<T> {
    let first = run(&token_range(input, 0..split, &input.state));
    let second = run(&token_range(input, split..input.suffix_len, &first.state));
    Output {
        y: [first.y, second.y].concat(),
        state: second.state,
    }
}

fn from_bits<T: ArrayElement>(bits: u16) -> T {
    bytemuck::pod_read_unaligned(&bits.to_ne_bytes())
}

/// Q 2 with decay exactly 1 (softplus(-104) rounds to +0), x = C = state = 1, d = 0, gate 32 (z = 32) and B half of
/// 1's spacing in T (2^-11 F16, 2^-8 BF16): kept in f32 the state ends at 1 + 2B, the next T value, and y1 = 32 (1 + 2B);
/// rounded to T between two Q 1 dispatches it stays 1 and both y are 32.
#[uzu_test]
fn retention_witnesses() {
    fn check<T: ArrayElement + Float + Debug>(
        b: f64,
        [one, above_one, y_32, y_above]: [u16; 4],
    ) {
        let dims = [2, 1, 1, 1, 1];
        let input = layout_input::<T>(dims, contiguous(dims), |array, _| [1.0, -104.0, b, 1.0, 0.0, 32.0, 1.0][array]);
        let retained = [from_bits::<T>(y_32), from_bits(y_above), from_bits(above_one)];
        let rounded = [from_bits::<T>(y_32), from_bits(y_32), from_bits(one)];
        let runs = [
            ("CPU", cpu_output(&input), retained),
            ("reference", reference_output(&input), retained),
            ("CPU in two dispatches", chained(&input, 1, cpu_output), rounded),
            ("reference in two dispatches", chained(&input, 1, reference_output), rounded),
        ];
        for (name, output, [y0, y1, state]) in runs {
            let label = format!("{:?} {name}", T::data_type());
            assert_bits(&[y0, y1], &output.y, &label);
            assert_bits(&[state], &output.state, &label);
        }
    }
    check::<f16>(2f64.powi(-11), [0x3c00, 0x3c01, 0x5000, 0x5001]);
    check::<bf16>(2f64.powi(-8), [0x3f80, 0x3f81, 0x4200, 0x4201]);
}

/// y of one element with x = C = 1 and B = d = 0 from 16-bit (dt_raw, s, z), with `stage` staged as the contract does
/// not: "dt" softplus rounded to T, "decay" the decay rounded to T, "gate" SiLU kept in f32. For positive s the
/// contract's added zeros and unit products are exact, leaving y = T((s decay) gate).
fn alternative_y<T: ArrayElement + Float>(
    stage: &str,
    [dt_raw, s, z]: [T; 3],
) -> T {
    let f = |value: T| value.to_f32().unwrap();
    let round = |value: f32| f(T::from(value).unwrap());
    let dt = ActivationType::SOFTPLUS.activate(f(dt_raw));
    let decay = (-[dt, round(dt)][usize::from(stage == "dt")]).exp();
    let decay = [decay, round(decay)][usize::from(stage == "decay")];
    let gate = match stage {
        "gate" => ActivationType::SILU.activate(f(z)),
        _ => f(ActivationType::SILU.activate(z)),
    };
    T::from(f(s) * decay * gate).unwrap()
}

/// Bounded 16-bit grids: dt_raw in ±[1/8, 8) with s 1 (dt) or s in {5/4, 3/2, 7/4, 3} (decay), z 32; s in [1, 4), z in
/// {1/2, 1, 2, 3}, dt_raw -104 (gate). The CPU gives the reference y for up to 64 cases per stage where it differs.
#[uzu_test]
fn staging_witnesses() {
    fn check<T: ArrayElement + Float + Debug>() {
        let v = |value: f64| T::from(value).unwrap();
        let grid = |low: f64, high: f64| {
            let values = (0..=u16::MAX).map(|bits| from_bits::<T>(bits));
            values.filter(move |value| (low..high).contains(&value.to_f64().unwrap().abs()))
        };
        let batch = |cases: &[[T; 3]]| {
            let dims = [1, cases.len(), 1, 1, 1];
            let case = |i: usize| cases[i].map(|value| value.to_f64().unwrap());
            layout_input::<T>(dims, contiguous(dims), |array, i| {
                [1.0, case(i)[0], 0.0, 1.0, 0.0, case(i)[2], case(i)[1]][array]
            })
        };
        for stage in ["dt", "decay", "gate"] {
            let candidates: Vec<[T; 3]> = match stage {
                "dt" => grid(0.125, 8.0).map(|dt| [dt, v(1.0), v(32.0)]).collect(),
                "decay" => {
                    grid(0.125, 8.0).flat_map(|dt| [1.25, 1.5, 1.75, 3.0].map(|s| [dt, v(s), v(32.0)])).collect()
                },
                _ => grid(1.0, 4.0)
                    .filter(|s| s.is_sign_positive())
                    .flat_map(|s| [0.5, 1.0, 2.0, 3.0].map(|z| [v(-104.0), s, v(z)]))
                    .collect(),
            };
            let reference = reference_output(&batch(&candidates)).y;
            let differs =
                |&(case, y): &([T; 3], T)| bytemuck::bytes_of(&alternative_y(stage, case)) != bytemuck::bytes_of(&y);
            let found =
                candidates.iter().copied().zip(reference).filter(differs).map(|(case, _)| case).collect::<Vec<_>>();
            let cases = &found[..found.len().min(64)];
            let label = format!("{:?} {stage}: {} of {} differ", T::data_type(), found.len(), candidates.len());
            eprintln!("{label}; run {cases:?}");
            assert!(!cases.is_empty(), "{label}: no witness");
            let output = cpu_output(&batch(cases));
            assert_bits(&reference_output(&batch(cases)).y, &output.y, &label);
            assert!(cases.iter().copied().zip(output.y).all(|pair| differs(&pair)), "{label}: alternative");
        }
    }
    check::<f16>();
    check::<bf16>();
}

/// SSDUpdate's accepted f32 order witnesses at Q 1, x = 1, dt_raw +inf (decay +0, so each state becomes B) and z 2^-27
/// (gate 2^-28): d x after the dot (first: 2^-28), ascending (descending: (2^24 + 2) 2^-28), the +0 start (N 0 and a -0
/// term), unfused (an FMA: 2^-52).
#[uzu_test]
fn order_witnesses() {
    let p = |e: i32| 2f32.powi(e);
    let cases: [(&str, f32, &[f32], &[f32], &[f32], f32); 5] = [
        ("d x after the dot", -p(24), &[p(24), 1.0], &[1.0; 2], &[1.0; 2], 0.0),
        ("ascending dot", 0.0, &[p(24), 1.0, 1.0], &[1.0; 3], &[0.0; 3], p(-4)),
        ("+0 start without state", -0.0, &[], &[], &[], 0.0),
        ("+0 start before a -0 term", -0.0, &[-0.0], &[1.0], &[-1.0], 0.0),
        ("unfused", 0.0, &[-(1.0 + p(-11)), 1.0 + p(-12)], &[1.0, 1.0 + p(-12)], &[0.0; 2], 0.0),
    ];
    for (name, d, b, c, state, y) in cases {
        let dims = [1, 1, 1, b.len(), 1];
        let values = [1.0, f64::INFINITY, 0.0, 0.0, d as f64, 2f64.powi(-27), 0.0];
        let mut input = layout_input::<f32>(dims, contiguous(dims), |array, _| values[array]);
        if !b.is_empty() {
            (input.b, input.c, input.state) = (b.into(), c.into(), state.into());
        }
        let next_state = [&input.state[..], b][usize::from(!b.is_empty())].to_vec();
        for (side, output) in [("CPU", cpu_output(&input)), ("reference", reference_output(&input))] {
            assert_bits(&[y], &output.y, &format!("{name}: {side} y"));
            assert_bits(&next_state, &output.state, &format!("{name}: {side} state"));
        }
    }
}

/// Padded strides (inner B, C and state strides 2 and 3, a final partial group), dt, B and C broadcast over tokens,
/// interleaved heads and elements: unwritten y, state and input elements keep their values.
#[uzu_test]
fn strided_layouts() {
    fn check<T: ArrayElement + Float + Debug>() {
        let layouts = [
            ([3, 3, 2, 4, 2], [16, 5, 2, 4, 1, 20, 9, 2, 30, 13, 3]),
            ([2, 2, 2, 4, 1], [8, 4, 1, 0, 1, 0, 4, 1, 8, 4, 1]),
            ([2, 2, 3, 2, 2], [8, 1, 2, 2, 1, 2, 2, 1, 1, 2, 6]),
        ];
        for (dims, strides) in layouts {
            let input = layout_input::<T>(dims, strides, pattern);
            let (expected, output) = (reference_output(&input), cpu_output(&input));
            let label = format!("{:?} {dims:?} {strides:?}", T::data_type());
            assert_bits(&expected.y, &output.y, &format!("{label} y"));
            assert_bits(&expected.state, &output.state, &format!("{label} state"));
        }
    }
    check::<f32>();
    check::<f16>();
    check::<bf16>();
}

/// Without work (Q, H or Dh 0, u32::MAX strides, NaN data) neither kernel writes, SSDPrefill64 also with N 63. With N 0,
/// y = T((+0 + d x) gate) beside NaN dt_raw, B, C and state at u32::MAX strides, whose bytes stay. This observes data
/// dependence only; that those pointers are never read is the kernel's structure.
#[uzu_test]
fn no_work_and_empty_state() {
    let max = u32::MAX as usize;
    for dims in [[0, 2, 2, 63, 1], [3, 0, 2, 63, 1], [3, 2, 0, 63, 1]] {
        let input = layout_input::<f32>(dims, [max; 11], |_, _| f64::NAN);
        for special64 in [false, true] {
            let output = get_output::<Cpu, f32>(&input, special64);
            assert_same_bytes(&output.y, &[sentinel()], &format!("{dims:?} y"));
            assert_same_bytes(&output.state, &input.state, &format!("{dims:?} state"));
        }
    }
    fn empty_state<T: ArrayElement + Float + Debug>() {
        let mut strides = [u32::MAX as usize; 11];
        strides[..3].copy_from_slice(&[4, 2, 1]);
        let input = layout_input::<T>([2, 2, 2, 0, 1], strides, |array, element| match array {
            0 | 4 | 5 => pattern(array, element),
            _ => f64::NAN,
        });
        let f = |value: T| value.to_f32().unwrap();
        let gate = |index: usize| f(ActivationType::SILU.activate(input.z[index]));
        let y = (0..8)
            .map(|i| T::from((0.0 + f(input.d[i / 2 % 2]) * f(input.x[i])) * gate(i)).unwrap())
            .collect::<Vec<_>>();
        let output = cpu_output(&input);
        assert!(y.iter().all(|value| value.is_finite()), "{y:?}");
        assert_bits(&y, &output.y, &format!("{:?} N 0 y", T::data_type()));
        assert_bits(&reference_output(&input).y, &output.y, &format!("{:?} N 0 reference y", T::data_type()));
        assert_same_bytes(&output.state, &input.state, &format!("{:?} N 0 state", T::data_type()));
    }
    empty_state::<f32>();
    empty_state::<f16>();
    empty_state::<bf16>();
}

/// group_size 0 reads B and C as group_size 1 does.
#[uzu_test]
fn group_size_zero_is_one() {
    let [zero, one] =
        [0, 1].map(|group| cpu_output(&layout_input::<f32>([3, 3, 2, 4, group], contiguous([3, 3, 2, 4, 1]), pattern)));
    assert_bits(&one.y, &zero.y, "y");
    assert_bits(&one.state, &zero.state, "state");
}

/// SSDPrefill64 gives SSDPrefill's bits at N 64 and rejects productive N 63 and 65 (surfacing as the failed wait).
#[uzu_test]
fn prefill64_matches_prefill() {
    let input = get_input::<f32>(3, 2, 2, 64, 1);
    let [generic, special] = [false, true].map(|special64| get_output::<Cpu, f32>(&input, special64));
    assert_bits(&generic.y, &special.y, "y");
    assert_bits(&generic.state, &special.state, "state");
    for n in [63, 65] {
        let input = get_input::<f32>(1, 1, 1, n, 1);
        let special = catch_unwind(AssertUnwindSafe(|| get_output::<Cpu, f32>(&input, true)));
        assert!(special.is_err(), "SSDPrefill64 accepted N {n}");
        assert_bits(&reference_output(&input).y, &cpu_output(&input).y, &format!("SSDPrefill N {n}"));
    }
}

/// A NaN reaches exactly its dependents in [Q 2, H 1, Dh 1, N 2]: z, C and d only y; x, dt_raw, B and the initial state
/// the state elements they feed and y from their token on.
#[uzu_test]
fn nan_dependencies() {
    fn check<T: ArrayElement + Float + Debug>() {
        let dims = [2, 1, 1, 2, 1];
        let cases = [
            (5, 0, [true, false], [false, false]),
            (3, 0, [true, false], [false, false]),
            (4, 0, [true, true], [false, false]),
            (0, 0, [true, true], [true, true]),
            (1, 1, [false, true], [true, true]),
            (2, 3, [false, true], [false, true]),
            (6, 0, [true, true], [true, false]),
        ];
        for (array, element, y_nan, state_nan) in cases {
            let mut input = layout_input::<T>(dims, contiguous(dims), pattern);
            let arrays =
                [&mut input.x, &mut input.dt, &mut input.b, &mut input.c, &mut input.d, &mut input.z, &mut input.state];
            arrays.into_iter().nth(array).unwrap()[element] = T::nan();
            let output = cpu_output(&input);
            let label = format!("{:?} NaN in array {array} element {element}", T::data_type());
            assert_bits(&reference_output(&input).y, &output.y, &label);
            assert_eq!(output.y.iter().map(|value| value.is_nan()).collect::<Vec<_>>(), y_nan, "{label} y");
            assert_eq!(output.state.iter().map(|value| value.is_nan()).collect::<Vec<_>>(), state_nan, "{label} state");
        }
    }
    check::<f32>();
    check::<f16>();
    check::<bf16>();
}

/// T narrowing only at the stores: an F16 state passing 65504 in f32 (60000 + 60000 - 60000) ends 60000, infinite when
/// rounded between two dispatches; an f32 subnormal state 2^-140 survives two tokens, y = 2^-140 2^100 32 = 2^-35.
#[uzu_test]
fn narrowing_at_stores() {
    let dims = [2, 1, 1, 1, 1];
    let mut input = layout_input::<f16>(dims, contiguous(dims), |array, _| {
        [1.0, -104.0, 0.0, 2f64.powi(-10), 0.0, 32.0, 60000.0][array]
    });
    input.b = [60000.0, -60000.0].map(f16::from_f64).into();
    let output = cpu_output(&input);
    assert_bits(&[f16::from_f64(3750.0), f16::from_f64(1875.0)], &output.y, "F16 y");
    assert_bits(&[f16::from_f64(60000.0)], &output.state, "F16 state");
    assert_bits(&[f16::INFINITY], &chained(&input, 1, cpu_output).state, "F16 state in two dispatches");

    let tiny = f32::from_bits(1 << 9);
    let input = layout_input::<f32>(dims, contiguous(dims), |array, _| {
        [1.0, -104.0, 0.0, 2f64.powi(100), 0.0, 32.0, tiny as f64][array]
    });
    let output = cpu_output(&input);
    assert_bits(&[2f32.powi(-35); 2], &output.y, "f32 subnormal y");
    assert_bits(&[tiny], &output.state, "f32 subnormal state");
}

/// f32 in two dispatches (tokens 0..2, 2..5) gives one dispatch's bits; F16 and BF16 give the reference that rounds
/// the state to T at the boundary (retention_witnesses shows the boundary changing a result).
#[uzu_test]
fn chained_dispatches() {
    let input = get_input::<f32>(5, 3, 4, 8, 3);
    let (one, two) = (cpu_output(&input), chained(&input, 2, cpu_output));
    assert_bits(&one.y, &two.y, "f32 y");
    assert_bits(&one.state, &two.state, "f32 state");
    fn segmented<T: ArrayElement + Float + Debug>() {
        let input = get_input::<T>(5, 3, 4, 8, 3);
        let (expected, output) = (chained(&input, 2, reference_output), chained(&input, 2, cpu_output));
        assert_bits(&expected.y, &output.y, &format!("{:?} y", T::data_type()));
        assert_bits(&expected.state, &output.state, &format!("{:?} state", T::data_type()));
    }
    segmented::<f16>();
    segmented::<bf16>();
}

/// The state update s decay + B x is two rounded products and a sum; fusing either product keeps its rounding error.
/// At dt_raw -1, s = 1 + 2^-23 and B = -(s decay) cancel to +0 unless s decay is fused; at decay 1 (dt_raw -104),
/// s = -(1 + 2^-11) and x = B = 1 + 2^-12 cancel to +0 unless B x (exactly 1 + 2^-11 + 2^-24) is fused, leaving 2^-24.
/// With C 1, d 0 and gate 32 the contract's y and state are +0; each fused alternative is asserted nonzero.
#[uzu_test]
fn recurrence_fma_witnesses() {
    let p = |e: i32| 2f32.powi(e);
    let decay = |dt_raw: f32| (-ActivationType::SOFTPLUS.activate(dt_raw)).exp();
    let (s, b) = (1.0 + p(-23), 1.0 + p(-12));
    // (dt_raw, s, x, B, the next state with one product fused)
    let cases = [
        (-1.0, s, 1.0, -(s * decay(-1.0)), s.mul_add(decay(-1.0), -(s * decay(-1.0)) * 1.0)),
        (-104.0, -(1.0 + p(-11)), b, b, b.mul_add(b, -(1.0 + p(-11)) * decay(-104.0))),
    ];
    for (dt_raw, s, x, b, fused) in cases {
        let label = format!("dt_raw {dt_raw}: fused next state {fused:e}");
        eprintln!("{label}");
        assert!(fused != 0.0, "{label}: the fused alternative does not differ");
        let dims = [1, 1, 1, 1, 1];
        let input =
            layout_input::<f32>(dims, contiguous(dims), |array, _| [x, dt_raw, b, 1.0, 0.0, 32.0, s][array] as f64);
        for (side, output) in [("CPU", cpu_output(&input)), ("reference", reference_output(&input))] {
            assert_bits(&[0.0], &output.y, &format!("{label}: {side} y"));
            assert_bits(&[0.0], &output.state, &format!("{label}: {side} state"));
        }
    }
}

/// One infinity at a time at [Q 1, H 1, Dh 1, N 1] with x = B = C = state = 1, decay 1 (dt_raw -104), d 0 and gate 32
/// (z 32), where the contract gives state 2 and y 64: x reaches the state, and y is NaN since d x = 0 inf; B and the
/// state reach both; C, d and z reach only y, where SiLU(-inf) = -inf / inf is NaN; dt_raw +inf decays to exactly 0
/// (state 1, y 32) and -inf to 1. An infinite state times that zero decay is NaN, never dropped.
#[uzu_test]
fn infinity_dependencies() {
    fn check<T: ArrayElement + Float + Debug>() {
        let (inf, nan) = (f64::INFINITY, f64::NAN);
        let dims = [1, 1, 1, 1, 1];
        let base = [1.0, -104.0, 1.0, 1.0, 0.0, 32.0, 1.0];
        // ([x, dt_raw, B, C, d, z, state] overrides, y, state)
        let cases = [
            (vec![(0, inf)], nan, inf),
            (vec![(0, -inf)], nan, -inf),
            (vec![(2, inf)], inf, inf),
            (vec![(2, -inf)], -inf, -inf),
            (vec![(3, inf)], inf, 2.0),
            (vec![(3, -inf)], -inf, 2.0),
            (vec![(4, inf)], inf, 2.0),
            (vec![(4, -inf)], -inf, 2.0),
            (vec![(5, inf)], inf, 2.0),
            (vec![(5, -inf)], nan, 2.0),
            (vec![(6, inf)], inf, inf),
            (vec![(6, -inf)], -inf, -inf),
            (vec![(1, inf)], 32.0, 1.0),
            (vec![(1, -inf)], 64.0, 2.0),
            (vec![(1, inf), (6, inf)], nan, nan),
        ];
        for (overrides, y, state) in cases {
            let mut values = base;
            for &(array, value) in &overrides {
                values[array] = value;
            }
            let input = layout_input::<T>(dims, contiguous(dims), |array, _| values[array]);
            let label = format!("{:?} {overrides:?}", T::data_type());
            let [y, state] = [y, state].map(|value| T::from(value).unwrap());
            for (side, output) in [("CPU", cpu_output(&input)), ("reference", reference_output(&input))] {
                assert_bits(&[y], &output.y, &format!("{label}: {side} y"));
                assert_bits(&[state], &output.state, &format!("{label}: {side} state"));
            }
        }
    }
    check::<f32>();
    check::<f16>();
    check::<bf16>();
}
