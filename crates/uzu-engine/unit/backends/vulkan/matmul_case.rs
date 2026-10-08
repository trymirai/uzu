use std::{collections::BTreeMap, ops::Range, sync::Arc};

use half::bf16;

use super::{hashed, kernel_fixture::KernelFixture, tanh_interval};
use crate::{
    backends::{
        common::{
            Backend, Context, Kernels,
            kernel::matmul::{MatmulA, MatmulArguments, MatmulB, MatmulDOps, MatmulKernel, MatmulOutput},
        },
        cpu::Cpu,
        vulkan::{GemmVulkanKernel, GemvVulkanKernel, VkBuffer, VkCommandBufferEncoding},
    },
    data_type::DataType,
    tests::helpers::{buffer_to_vec, create_buffer_with_data, create_context, submit_command_buffer},
};

/// One raw Gemv or Gemm dispatch shared by the CPU and Vulkan runners and the FP64 oracle: values held as FP32, each
/// exactly representable in its storage type, `types` of B, A and D. B holds rows of `k`, A `a_offset` NaN
/// elements before its `[m, k]` rows, D the prior `[m, n]` output; bias is one B-typed value per output column and
/// `gather` names the B row of every output (Gemv only). Mask bits: 1 scale, 2 accumulate, 4 bias, 8 soft cap, 16 gather.
#[derive(Clone)]
pub struct MatmulCase {
    pub types: [DataType; 3],
    pub m: u32,
    pub n: u32,
    pub k: u32,
    pub a_offset: usize,
    pub b: Vec<f32>,
    pub a: Vec<f32>,
    pub d: Vec<f32>,
    pub bias: Option<Vec<f32>>,
    pub gather: Option<Vec<u32>>,
    pub ab_scale: Option<f32>,
    pub accumulate: bool,
    pub soft_cap: Option<f32>,
}

impl MatmulCase {
    pub const SCALE: u32 = 1;
    pub const ACCUMULATE: u32 = 2;
    pub const BIAS: u32 = 4;
    pub const SOFT_CAP: u32 = 8;
    pub const GATHER: u32 = 16;

    /// Every B, A and D triple of F32 and BF16.
    pub fn triples() -> impl Iterator<Item = [DataType; 3]> {
        itertools::iproduct!(
            [DataType::F32, DataType::BF16],
            [DataType::F32, DataType::BF16],
            [DataType::F32, DataType::BF16]
        )
        .map(|(b, a, d)| [b, a, d])
    }

    /// One A row against `rows` of B under `mask`, without gather; other fields as `new` makes them.
    pub fn witness(
        types: [DataType; 3],
        a: &[f32],
        rows: &[&[f32]],
        mask: u32,
    ) -> Self {
        let mut case = Self::new(types, 1, rows.len() as u32, a.len() as u32, mask & !Self::GATHER, 0);
        (case.a, case.b) = (a.to_vec(), rows.concat());
        case
    }

    /// Hashed operands in [-2, 2), prior outputs and biases in [-1, 1), scale 0.75 and cap 3 under `mask`; gathered
    /// outputs name B rows out of order and repeatedly among `n + 3`; A starts 1 to 3 elements into its range.
    pub fn new(
        types: [DataType; 3],
        m: u32,
        n: u32,
        k: u32,
        mask: u32,
        seed: u32,
    ) -> Self {
        let gathered = mask & 16 != 0;
        let weight_rows = if gathered {
            n + 3
        } else {
            n
        };
        let data = |length: u32, scale: f32, seed: u32, data_type: DataType| {
            (0..length).map(|index| Self::stored(scale * hashed(index, seed), data_type)).collect::<Vec<_>>()
        };
        let [b_type, a_type, d_type] = types;
        Self {
            types,
            m,
            n,
            k,
            a_offset: 1 + seed as usize % 3,
            b: data(weight_rows * k, 2.0, seed + 1, b_type),
            a: data(m * k, 2.0, seed + 2, a_type),
            d: data(m * n, 1.0, seed + 3, d_type),
            bias: (mask & 4 != 0).then(|| data(n, 1.0, seed + 4, b_type)),
            gather: gathered.then(|| (0..m * n).map(|index| (index * 7 + seed) % weight_rows).collect()),
            ab_scale: (mask & 1 != 0).then_some(0.75),
            accumulate: mask & 2 != 0,
            soft_cap: (mask & 8 != 0).then_some(3.0),
        }
    }

    /// `value` rounded to nearest even in `data_type`.
    pub fn stored(
        value: f32,
        data_type: DataType,
    ) -> f32 {
        match data_type {
            DataType::BF16 => bf16::from_f32(value).to_f32(),
            _ => value,
        }
    }

    /// Storage bytes of `values`, each exactly representable in `data_type`; BF16 keeps the upper bits as they are, NaN
    /// payloads included.
    pub fn bytes(
        values: &[f32],
        data_type: DataType,
    ) -> Vec<u8> {
        match data_type {
            DataType::BF16 => {
                let halves = values
                    .iter()
                    .map(|value| {
                        assert_eq!(value.to_bits() & 0xffff, 0, "{value:e} is not a BF16 value");
                        (value.to_bits() >> 16) as u16
                    })
                    .collect::<Vec<_>>();
                bytemuck::cast_slice(&halves).to_vec()
            },
            _ => bytemuck::cast_slice(values).to_vec(),
        }
    }

    fn values(
        bytes: &[u8],
        data_type: DataType,
    ) -> Vec<f32> {
        match data_type {
            DataType::BF16 => bytemuck::pod_collect_to_vec::<u8, u16>(bytes)
                .into_iter()
                .map(|half| f32::from_bits(u32::from(half) << 16))
                .collect(),
            _ => bytemuck::pod_collect_to_vec(bytes),
        }
    }

    fn mask(&self) -> [bool; 5] {
        [self.ab_scale.is_some(), self.accumulate, self.bias.is_some(), self.soft_cap.is_some(), self.gather.is_some()]
    }

    /// A's range: `a_offset` NaN elements, then A.
    fn a_storage(&self) -> Vec<f32> {
        [vec![f32::NAN; self.a_offset], self.a.clone()].concat()
    }

    /// The B row of output `index`.
    pub fn weight_row(
        &self,
        index: usize,
    ) -> usize {
        self.gather.as_ref().map_or(index % self.n as usize, |gather| gather[index] as usize)
    }

    /// D after `repeat` dispatches of the CPU MatmulKernel in one command buffer; `before_soft_cap` drops the soft cap
    /// and stores FP32, so the result is the FP32 value the soft cap receives. CPU buffers cannot be empty.
    pub fn cpu(
        &self,
        repeat: usize,
        before_soft_cap: bool,
    ) -> Vec<f32> {
        let [b_type, a_type, d_type] = self.types;
        let d_type = if before_soft_cap {
            DataType::F32
        } else {
            d_type
        };
        let context = create_context::<Cpu>();
        let mut kernel = <<Cpu as Backend>::Kernels as Kernels>::MatmulKernel::new(&context, b_type, a_type, d_type)
            .expect("CPU MatmulKernel");
        let buffer = |values: &[f32], data_type: DataType| {
            let values = if values.is_empty() {
                &[0.0][..]
            } else {
                values
            };
            match data_type {
                DataType::BF16 => create_buffer_with_data::<Cpu, bf16>(
                    &context,
                    &values.iter().map(|&value| bf16::from_f32(value)).collect::<Vec<_>>(),
                ),
                _ => create_buffer_with_data::<Cpu, f32>(&context, values),
            }
        };
        let (b, a) = (buffer(&self.b, b_type), buffer(&self.a_storage(), a_type));
        let mut d = buffer(&self.d, d_type);
        let bias = self.bias.as_ref().map(|bias| buffer(bias, b_type));
        let gather = self.gather.as_ref().map(|gather| create_buffer_with_data::<Cpu, u32>(&context, gather));
        let mut command_buffer = context.create_command_buffer(None, None).expect("CPU command buffer");
        for _ in 0..repeat {
            let ops = MatmulDOps {
                ab_scale: self.ab_scale.unwrap_or(1.0),
                accumulate: self.accumulate,
                bias: bias.as_ref(),
                rht_factors: None,
                soft_cap: self.soft_cap.filter(|_| !before_soft_cap),
            };
            let arguments = MatmulArguments {
                a: MatmulA::FullPrecision {
                    values: &a,
                    offset: self.a_offset,
                },
                b: MatmulB::FullPrecision {
                    b: &b,
                },
                b_leading_dimension: None,
                b_transpose: true,
                output: MatmulOutput::new(&mut d, ops),
                gather_indices: gather.as_ref(),
                m: self.m,
                n: self.n,
                k: self.k,
            };
            kernel.encode(arguments, &mut command_buffer).expect("CPU Gemv");
        }
        submit_command_buffer(command_buffer);
        let length = self.d.len();
        match d_type {
            DataType::BF16 => buffer_to_vec::<Cpu, bf16>(&d)[..length].iter().map(|value| value.to_f32()).collect(),
            _ => buffer_to_vec::<Cpu, f32>(&d)[..length].to_vec(),
        }
    }

    pub fn gemv_kernel(
        &self,
        fixture: &KernelFixture,
    ) -> GemvVulkanKernel {
        let [b_type, a_type, d_type] = self.types;
        let [has_scale, accumulate, has_bias, has_soft_cap, gathered] = self.mask();
        GemvVulkanKernel::new(
            &fixture.context,
            a_type,
            b_type,
            d_type,
            has_scale,
            accumulate,
            has_bias,
            has_soft_cap,
            gathered,
        )
        .expect("Vulkan Gemv")
    }

    pub fn gemm_kernel(
        &self,
        fixture: &KernelFixture,
    ) -> GemmVulkanKernel {
        assert!(self.gather.is_none(), "Gemm takes no gather");
        let [b_type, a_type, d_type] = self.types;
        let [has_scale, accumulate, has_bias, has_soft_cap, _] = self.mask();
        GemmVulkanKernel::new(&fixture.context, a_type, b_type, d_type, has_scale, accumulate, has_bias, has_soft_cap)
            .expect("Vulkan Gemm")
    }

    /// Records the Gemv dispatch over `[b, a, d]` and the optional bias and gather ranges; A's range starts at A's
    /// first element.
    ///
    /// # Safety
    /// The ranges hold every element the case indexes, aligned, gather indices are below `weight_rows`, and D aliases
    /// nothing.
    pub unsafe fn encode_gemv(
        &self,
        kernel: &GemvVulkanKernel,
        [b, a, d]: [(&Arc<VkBuffer>, Range<u64>); 3],
        bias: Option<(&Arc<VkBuffer>, Range<u64>)>,
        gather: Option<(&Arc<VkBuffer>, Range<u64>)>,
        encoding: &mut VkCommandBufferEncoding,
    ) {
        // SAFETY: forwarded from the caller.
        unsafe { kernel.encode(b, a, d, bias, gather, self.k, self.n, self.m, self.ab_scale, self.soft_cap, encoding) }
    }

    /// Records the Gemm dispatch as `encode_gemv` does, without gather.
    ///
    /// # Safety
    /// As for `encode_gemv`.
    pub unsafe fn encode_gemm(
        &self,
        kernel: &GemmVulkanKernel,
        [b, a, d]: [(&Arc<VkBuffer>, Range<u64>); 3],
        bias: Option<(&Arc<VkBuffer>, Range<u64>)>,
        encoding: &mut VkCommandBufferEncoding,
    ) {
        // SAFETY: forwarded from the caller.
        unsafe { kernel.encode(b, a, d, bias, self.k, self.n, self.m, self.ab_scale, self.soft_cap, encoding) }
    }

    /// D after `repeat` Gemv dispatches, as `gpu` records them.
    pub fn gemv(
        &self,
        fixture: &KernelFixture,
        repeat: usize,
    ) -> Vec<f32> {
        let kernel = self.gemv_kernel(fixture);
        // SAFETY: `gpu` passes guarded ranges holding the case's elements and indices; D aliases nothing.
        self.gpu(fixture, repeat, |ranges, bias, gather, encoding| unsafe {
            self.encode_gemv(&kernel, ranges, bias, gather, encoding)
        })
    }

    /// D after `repeat` Gemm dispatches, as `gpu` records them.
    pub fn gemm(
        &self,
        fixture: &KernelFixture,
        repeat: usize,
    ) -> Vec<f32> {
        let kernel = self.gemm_kernel(fixture);
        // SAFETY: `gpu` passes guarded ranges holding the case's elements; D aliases nothing.
        self.gpu(fixture, repeat, |ranges, bias, _, encoding| unsafe {
            self.encode_gemm(&kernel, ranges, bias, encoding)
        })
    }

    /// D after `repeat` dispatches `encode` records in one command buffer over guarded byte ranges of `[b, a, d]`, bias
    /// and gather, A's starting `a_offset` elements in, after asserting every guard and input unchanged.
    pub fn gpu(
        &self,
        fixture: &KernelFixture,
        repeat: usize,
        mut encode: impl FnMut(
            [(&Arc<VkBuffer>, Range<u64>); 3],
            Option<(&Arc<VkBuffer>, Range<u64>)>,
            Option<(&Arc<VkBuffer>, Range<u64>)>,
            &mut VkCommandBufferEncoding,
        ),
    ) -> Vec<f32> {
        let [b_type, a_type, d_type] = self.types;
        let sentinel = 0xa5u8;
        let inputs =
            [(&self.b, b_type), (&self.a_storage(), a_type)].map(|(values, data_type)| Self::bytes(values, data_type));
        let [b, a] = inputs.each_ref().map(|bytes| fixture.guarded(bytes, sentinel));
        let d = fixture.guarded(&Self::bytes(&self.d, d_type), sentinel);
        let bias = self.bias.as_ref().map(|bias| Self::bytes(bias, b_type));
        let bias_range = bias.as_ref().map(|bytes| fixture.guarded(bytes, sentinel));
        let gather = self.gather.as_ref().map(|gather| bytemuck::cast_slice::<u32, u8>(gather).to_vec());
        let gather_range = gather.as_ref().map(|bytes| fixture.guarded(bytes, sentinel));
        let skip = (self.a_offset * a_type.size_in_bytes()) as u64;
        fn range((buffer, range): &(Arc<VkBuffer>, Range<u64>)) -> (&Arc<VkBuffer>, Range<u64>) {
            (buffer, range.clone())
        }
        // A recorded dispatch retains every buffer it declares until completion.
        let owners = || [&b, &a, &d].map(|(buffer, _)| Arc::strong_count(buffer));
        let unrecorded = owners();
        let mut encoding = fixture.encoding();
        for _ in 0..repeat {
            let a_range = (&a.0, a.1.start + skip..a.1.end);
            encode(
                [range(&b), a_range, range(&d)],
                bias_range.as_ref().map(range),
                gather_range.as_ref().map(range),
                &mut encoding,
            );
        }
        if self.m == 0 || self.n == 0 {
            assert_eq!(owners(), unrecorded, "{}: an empty dispatch was recorded", self.label());
        }
        KernelFixture::complete(encoding);
        // SAFETY: the only command buffer using these buffers has completed.
        unsafe {
            for (guarded, payload, name) in [(&b, &inputs[0], "B"), (&a, &inputs[1], "A")] {
                KernelFixture::assert_unchanged(guarded, sentinel, payload, name);
            }
            if let (Some(guarded), Some(payload)) = (&bias_range, &bias) {
                KernelFixture::assert_unchanged(guarded, sentinel, payload, "bias");
            }
            if let (Some(guarded), Some(payload)) = (&gather_range, &gather) {
                KernelFixture::assert_unchanged(guarded, sentinel, payload, "gather");
            }
            Self::values(&KernelFixture::read_guarded(&d, sentinel), d_type)
        }
    }

    /// FP64 bounds of every output's FP32 value before its rounding to D, `None` where they cannot prove the CPU's order
    /// and every reordering free of overflow, infinities and NaN, which leaves class and bits to the CPU's order, or
    /// where the cap is zero or not finite. FP32 products are exact in FP64. With S = Σ|ab| and u = 2^-24, (k + 2) u at
    /// most 1/2: any order of the products' roundings, the sums, the scaled tiny sum and its merge keeps every partial
    /// within (1 + u)^(k + 2) S <= (1 + 2 (k + 2) u) S and the dot within 2 (k + 2) u S + (k + 1) 2^-150 of the exact
    /// sum, the second term half the subnormal spacing for each product rounded onto it and the tiny sum's scaling down.
    /// The FP64 sums Ŝ and Ê of the exact products are each within γ64(k) S <= g S, g = 2 k 2^-53, of S and of the exact
    /// sum, an absolute error that cancellation in the sum does not shrink, so S <= Ŝ (1 + 2 g) and the reference error is
    /// at most g S. The partials and every epilogue step must stay below
    /// 2^127, under FP32 overflow; each step is one FP32 rounding, u of its magnitude plus 2^-150. Every FP64 operation
    /// on a bound is rounded outward by one FP64 step, at least its own rounding error.
    pub fn bounds(&self) -> Vec<Option<(f64, f64)>> {
        let (unit, tiny, limit) = (2f64.powi(-24), 2f64.powi(-150), 2f64.powi(127));
        let rounded = |(lo, hi): (f64, f64)| {
            ((lo - (unit * lo.abs() + tiny).next_up()).next_down(), (hi + (unit * hi.abs() + tiny).next_up()).next_up())
        };
        let below = |(lo, hi): (f64, f64)| lo.abs().max(hi.abs()) < limit;
        let (m, n, k) = (self.m as usize, self.n as usize, self.k as usize);
        let count = k as f64;
        (0..m * n)
            .map(|index| {
                let (row, column) = (index / n, index % n);
                let operands = self.a[row * k..][..k].iter().zip(&self.b[self.weight_row(index) * k..][..k]);
                if (count + 2.0) * unit > 0.5 || !operands.clone().all(|(x, w)| x.is_finite() && w.is_finite()) {
                    return None;
                }
                let products = operands.map(|(&x, &w)| f64::from(x) * f64::from(w));
                let (exact, magnitude) =
                    products.fold((0.0, 0.0), |(sum, magnitude), p| (sum + p, magnitude + p.abs()));
                let reference = 2.0 * count * 2f64.powi(-53);
                let magnitude = (magnitude * (1.0 + 2.0 * reference)).next_up();
                // Finite operands give no NaN, at most an infinite bound.
                if (magnitude * (1.0 + 2.0 * (count + 2.0) * unit)).next_up() >= limit {
                    return None;
                }
                let error = (2.0 * (count + 2.0) * unit * magnitude).next_up() + (count + 1.0) * tiny;
                let error = (error.next_up() + (reference * magnitude).next_up()).next_up();
                let mut value = ((exact - error).next_down(), (exact + error).next_up());
                if !below(value) {
                    return None;
                }
                let mut steps = Vec::new();
                steps.extend(self.ab_scale.map(|scale| (f64::from(scale), true)));
                steps.extend(self.accumulate.then(|| (f64::from(self.d[index]), false)));
                steps.extend(self.bias.as_ref().map(|bias| (f64::from(bias[column]), false)));
                for (operand, product) in steps {
                    let (lo, hi) = value;
                    if !operand.is_finite() {
                        return None;
                    }
                    value = rounded(match product {
                        true => {
                            ((lo * operand).min(hi * operand).next_down(), (lo * operand).max(hi * operand).next_up())
                        },
                        false => ((lo + operand).next_down(), (hi + operand).next_up()),
                    });
                    if !below(value) {
                        return None;
                    }
                }
                match self.soft_cap {
                    Some(cap) => Self::soft_cap_bounds(value, cap),
                    None => Some(value),
                }
            })
            .collect()
    }

    /// FP64 bounds of the soft cap cap tanh(v / cap) over FP32 values v in `value`, `None` for a zero or nonfinite
    /// cap: the quotient within Vulkan's 2.5 ULPs, of the scaled quotient below the smallest normal, the shader's
    /// tanh within `tanh_interval` at the quotient's extremes and the polynomial and builtin switches between them,
    /// widened by one more FP32 ULP for the CPU's libm, and the product one FP32 rounding. FP64 operations round outward.
    pub fn soft_cap_bounds(
        (lo, hi): (f64, f64),
        cap: f32,
    ) -> Option<(f64, f64)> {
        if cap == 0.0 || !cap.is_finite() {
            return None;
        }
        let cap = f64::from(cap);
        let widen = |(lo, hi): (f64, f64), ulps: f64, absolute: f64| {
            let error = |value: f64| (ulps * value.abs() * 2f64.powi(-23) + absolute).next_up();
            ((lo - error(lo)).next_down(), (hi + error(hi)).next_up())
        };
        let span = |a: f64, b: f64| (a.min(b).next_down(), a.max(b).next_up());
        let (q_lo, q_hi) = widen(span(lo / cap, hi / cap), 2.5, 2f64.powi(-149));
        let switches = [0.0, 0.1, 0.549_306_15].into_iter().flat_map(|x| [x, -x]);
        let points = [q_lo, q_hi].into_iter().chain(switches.filter(|x| (q_lo..=q_hi).contains(x)));
        let (t_lo, t_hi) = points
            .map(tanh_interval)
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), (a, b)| (lo.min(a), hi.max(b)));
        let (t_lo, t_hi) = widen((t_lo, t_hi), 1.0, 2f64.powi(-149));
        let (out_lo, out_hi) = span(cap * t_lo, cap * t_hi);
        Some(widen((out_lo, out_hi), 0.5, 2f64.powi(-150)))
    }

    /// `value` rounded to the nearest BF16 value, ties to even. half's `from_f64` rounds on the upper 32 bits of the FP64
    /// value only, so just past a midpoint it rounds as at the midpoint; its result is within one BF16 step, and of it and
    /// its neighbours the nearest is taken, by distances exact in FP64. An infinity stands at 2^128 as rounding sees it,
    /// so from 2^128 - 2^119 on, the tie with the odd largest finite value included, values round to it.
    fn nearest_bf16(value: f64) -> f64 {
        let ordinal = |bits: u16| {
            if bits & 0x8000 != 0 {
                -i32::from(bits & 0x7fff)
            } else {
                i32::from(bits)
            }
        };
        let bits = |ordinal: i32| {
            if ordinal < 0 {
                0x8000 | (-ordinal) as u16
            } else {
                ordinal as u16
            }
        };
        let guess = ordinal(bf16::from_f64(value).to_bits());
        let distance = |candidate: &bf16| {
            let position = if candidate.is_infinite() {
                2f64.powi(128).copysign(candidate.to_f64())
            } else {
                candidate.to_f64()
            };
            (position - value).abs()
        };
        (guess - 1..=guess + 1)
            .filter(|candidate| candidate.abs() <= 0x7f80)
            .map(|candidate| bf16::from_bits(bits(candidate)))
            .min_by(|x, y| distance(x).total_cmp(&distance(y)).then((x.to_bits() & 1).cmp(&(y.to_bits() & 1))))
            .expect("three candidates")
            .to_f64()
    }

    /// Whether a stored D value lies within FP64 bounds of the FP32 value it stores: F32 values within them, BF16 values
    /// between the bounds rounded to nearest even, which is monotonic.
    pub fn within(
        &self,
        stored: f32,
        (lo, hi): (f64, f64),
    ) -> bool {
        let (lo, hi) = match self.types[2] {
            DataType::BF16 => (Self::nearest_bf16(lo), Self::nearest_bf16(hi)),
            _ => (lo, hi),
        };
        lo <= f64::from(stored) && f64::from(stored) <= hi
    }

    pub fn label(&self) -> String {
        let flags = ["scale", "accumulate", "bias", "soft cap", "gather"];
        let mask = self.mask().iter().zip(flags).filter(|(on, _)| **on).map(|(_, flag)| flag).collect::<Vec<_>>();
        format!("B/A/D {:?} m {} n {} k {} A offset {} {mask:?}", self.types, self.m, self.n, self.k, self.a_offset)
    }

    /// The case on the CPU and through `run` against its FP64 bounds: where they exist both outputs within them, else
    /// Vulkan's class and bits the CPU's (NaN any NaN), and under a soft cap the bounds of the soft cap at the CPU's
    /// exact FP32 input where that is finite. Counts bounded and exact outputs and the largest Vulkan-CPU difference
    /// relative to the CPU's magnitude (at least the smallest normal) per label; returns the CPU and Vulkan outputs.
    pub fn check(
        &self,
        fixture: &KernelFixture,
        label: &str,
        run: fn(&Self, &KernelFixture, usize) -> Vec<f32>,
        totals: &mut BTreeMap<String, [f64; 3]>,
    ) -> (Vec<f32>, Vec<f32>) {
        let vulkan = run(self, fixture, 1);
        let cpu = self.cpu(1, false);
        let bounds = self.bounds();
        let before = (self.soft_cap.is_some() && bounds.iter().any(Option::is_none)).then(|| self.cpu(1, true));
        let total = totals.entry(label.to_owned()).or_default();
        for (index, bound) in bounds.into_iter().enumerate() {
            let name = format!("{label} {} output {index}", self.label());
            let bound = bound.or_else(|| {
                let value = f64::from(before.as_ref()?[index]);
                value.is_finite().then(|| Self::soft_cap_bounds((value, value), self.soft_cap?)).flatten()
            });
            match bound {
                Some(bound) => {
                    for (backend, value) in [("CPU", cpu[index]), ("Vulkan", vulkan[index])] {
                        assert!(
                            self.within(value, bound),
                            "{name}: {backend} {value:e} outside [{:e}, {:e}]",
                            bound.0,
                            bound.1
                        );
                    }
                    total[0] += 1.0;
                    let difference = f64::from(vulkan[index]) - f64::from(cpu[index]);
                    total[2] =
                        total[2].max(difference.abs() / f64::from(cpu[index].abs()).max(f64::from(f32::MIN_POSITIVE)));
                },
                None => {
                    KernelFixture::assert_bits(&cpu[index..=index], &vulkan[index..=index], &name);
                    total[1] += 1.0;
                },
            }
        }
        (cpu, vulkan)
    }

    pub fn report(
        kernel: &str,
        totals: &BTreeMap<String, [f64; 3]>,
    ) {
        for (label, [bounded, exact, difference]) in totals {
            eprintln!(
                "{kernel} {label}: {bounded} bounded outputs, {exact} exact outputs, max relative |Vulkan - CPU| \
                 {difference:.3e}"
            );
        }
    }

    pub fn check_all(
        kernel: &str,
        label: &str,
        run: fn(&Self, &KernelFixture, usize) -> Vec<f32>,
        cases: impl IntoIterator<Item = Self>,
    ) {
        let fixture = KernelFixture::new();
        let mut totals = BTreeMap::new();
        for case in cases {
            case.check(&fixture, label, run, &mut totals);
        }
        Self::report(kernel, &totals);
        fixture.assert_clean();
    }

    /// Both the CPU and `run` store exactly `expected`, bit for bit (NaN any NaN).
    pub fn exact(
        &self,
        fixture: &KernelFixture,
        label: &str,
        run: fn(&Self, &KernelFixture, usize) -> Vec<f32>,
        expected: &[f32],
    ) {
        let vulkan = run(self, fixture, 1);
        KernelFixture::assert_bits(expected, &self.cpu(1, false), &format!("{label} CPU"));
        KernelFixture::assert_bits(expected, &vulkan, &format!("{label} Vulkan"));
    }
}
