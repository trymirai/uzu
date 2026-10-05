//! Matmul: the GEMV kernel of the Metal backend for single rows, the native WMMA qmv for a few rows of
//! 4-bit weights (`qmv_wmma.rs`), and GEMM (WMMA, dequantized once per tile) from `UZU_AMDGPU_GEMM_MIN_M` rows.

mod gemm;
pub(crate) mod gemm_w4;
mod gemv;
pub(crate) mod gemv_w4;
mod policy;
pub(crate) mod qmv_wmma;

use self::{
    gemm::GemmKernel,
    gemv::{GemvKernel, GemvSpecialization},
};
use crate::{
    backends::{
        amdgpu::{Amdgpu, context::AmdgpuContext, error::AmdgpuError},
        common::{
            Backend, BufferMut, BufferRef, CommandBuffer,
            kernel::{
                ActivationQuantization, ActivationTransform,
                matmul::{ActivationFormat, MatmulArguments, MatmulError, MatmulKernel, MatmulShape},
            },
        },
    },
    data_type::DataType,
};

type Encoding = <<Amdgpu as Backend>::CommandBuffer as CommandBuffer>::Encoding;

/// Output random Hadamard transform that the GEMV tile could not fuse.
pub struct MatmulOutputWork {
    output_rht: ActivationTransform<Amdgpu>,
    output_rht_with_bias: ActivationTransform<Amdgpu>,
}

impl MatmulOutputWork {
    fn new(
        context: &AmdgpuContext,
        weights_data_type: DataType,
        output_data_type: DataType,
    ) -> Result<Self, AmdgpuError> {
        Ok(Self {
            output_rht: ActivationTransform::output_rht(context, output_data_type, None, true)?,
            output_rht_with_bias: ActivationTransform::output_rht(
                context,
                output_data_type,
                Some(weights_data_type),
                true,
            )?,
        })
    }

    fn apply(
        &self,
        output: impl BufferMut<Backend = Amdgpu>,
        factors: impl BufferRef<Backend = Amdgpu>,
        bias: Option<&<Amdgpu as Backend>::GlobalBuffer>,
        m: u32,
        n: u32,
        command_buffer: &mut Encoding,
    ) {
        let transform = if bias.is_some() {
            &self.output_rht_with_bias
        } else {
            &self.output_rht
        };
        transform.encode_fp_in_place(output, factors, bias, m, n, command_buffer);
    }
}

/// Batch size from which GEMM replaces GEMV. Measured on the 890M at the 9B gate_up shape:
/// - bf16 weights: WMMA GEMM beats GEMV 4x at M = 16 and 13x at M = 64..256 (11.1 vs 46.9 ms at M = 16);
/// - quantized weights (W4 zero-point, vectorized dequantizing loader): GEMM 16.5 ms at M = 16 against
///   6.5 for the multi-row GEMV, 21.3 at M = 64 against ~26, 77.5 at M = 256.
/// `UZU_AMDGPU_GEMM_MIN_M` / `UZU_AMDGPU_GEMM_MIN_M_QUANTIZED` override them.
fn gemm_min_m(quantized: bool) -> u32 {
    static FULL_PRECISION: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    static QUANTIZED: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    let read =
        |name: &str, default: u32| std::env::var(name).ok().and_then(|value| value.parse().ok()).unwrap_or(default);
    if quantized {
        *QUANTIZED.get_or_init(|| read("UZU_AMDGPU_GEMM_MIN_M_QUANTIZED", 48))
    } else {
        *FULL_PRECISION.get_or_init(|| read("UZU_AMDGPU_GEMM_MIN_M", 8))
    }
}

/// `UZU_AMDGPU_MATMUL_TRACE=1`: every distinct matmul shape and the path it takes, once, on stderr.
#[cfg(test)]
thread_local! {
    /// Routes taken by `encode` on this thread since the last `take_routes`.
    static ROUTES: std::cell::RefCell<Vec<String>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// The matmul routes taken on this thread since the last call, as `trace_route` labels: `"GemvW4"`,
/// `"GemmW4"`, `"QmvWmma"`, `"QmvWmma int8"`, a GEMM tiling (`Tile...`) or an MSL GEMV specialization
/// (`Some(GemvSpecialization ...)`). Unit tests assert with it which kernel ran, so a native path that stops
/// being selected cannot pass its tests through a fallback.
#[cfg(test)]
pub fn take_routes() -> Vec<String> {
    ROUTES.with(|routes| std::mem::take(&mut *routes.borrow_mut()))
}

fn trace_route(
    shape: &MatmulShape,
    data_types: [DataType; 3],
    route: &dyn std::fmt::Debug,
) {
    #[cfg(test)]
    ROUTES.with(|routes| routes.borrow_mut().push(format!("{route:?}")));
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    static SEEN: std::sync::Mutex<Option<std::collections::HashSet<String>>> = std::sync::Mutex::new(None);
    if !*ENABLED.get_or_init(|| std::env::var("UZU_AMDGPU_MATMUL_TRACE").is_ok_and(|value| value != "0")) {
        return;
    }
    let line = format!(
        "m={} n={} k={} types={data_types:?} b_transpose={} ld={:?} prologue={:?} bits={:?} group={:?} a_full_precision={} gathered={} d_transform={:?} -> {route:?}",
        shape.m,
        shape.n,
        shape.k,
        shape.b_transpose,
        shape.b_leading_dimension,
        shape.b_prologue,
        shape.b_bits,
        shape.b_group_size,
        shape.a_full_precision,
        shape.gathered,
        shape.d_transform,
    );
    let mut seen = SEEN.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    if seen.get_or_insert_default().insert(line.clone()) {
        eprintln!("[amdgpu matmul] {line}");
    }
}

pub struct AmdgpuMatmulKernel {
    gemv: GemvKernel,
    gemm: GemmKernel,
    output_work: MatmulOutputWork,
    weights_data_type: DataType,
    input_data_type: DataType,
    output_data_type: DataType,
}

impl MatmulKernel for AmdgpuMatmulKernel {
    type Backend = Amdgpu;

    fn select_activation_quantization(
        &self,
        candidate: &MatmulShape,
        _context: &AmdgpuContext,
    ) -> Option<ActivationQuantization> {
        qmv_wmma::activation_quantization(
            candidate,
            [self.weights_data_type, self.input_data_type, self.output_data_type],
        )
    }

    /// Int8 activations for the batch sizes the WMMA qmv takes (verification of speculation trees, short
    /// prefills): its int8 variant needs less VALU than the bf16 one, and the engine quantizes in the input
    /// transform it runs anyway. Decode (m = 1) stays bf16 for the MSL GEMV.
    fn select_activation_format(
        &self,
        bf16_shape: &MatmulShape,
        _context: &AmdgpuContext,
    ) -> ActivationFormat {
        let (qmv_min_m, qmv_max_m) = qmv_wmma::m_range();
        let a8_shape = MatmulShape {
            a_full_precision: false,
            ..*bf16_shape
        };
        let data_types = [self.weights_data_type, self.input_data_type, self.output_data_type];
        if qmv_wmma::a8_enabled()
            && (qmv_min_m..=qmv_max_m).contains(&bf16_shape.m)
            && bf16_shape.m < gemm_min_m(true)
            && qmv_wmma::supports(&a8_shape, data_types)
        {
            ActivationFormat::Int8
        } else {
            ActivationFormat::Bf16
        }
    }

    fn new(
        context: &AmdgpuContext,
        weights_data_type: DataType,
        input_data_type: DataType,
        output_data_type: DataType,
    ) -> Result<Self, AmdgpuError> {
        for data_type in [weights_data_type, input_data_type, output_data_type] {
            if !matches!(data_type, DataType::BF16 | DataType::F32) {
                return Err(MatmulError::<Amdgpu>::UnsupportedDataType(data_type).into());
            }
        }
        Ok(Self {
            gemv: GemvKernel::new(weights_data_type, input_data_type, output_data_type),
            gemm: GemmKernel::new(weights_data_type, input_data_type, output_data_type),
            output_work: MatmulOutputWork::new(context, weights_data_type, output_data_type)?,
            weights_data_type,
            input_data_type,
            output_data_type,
        })
    }

    fn encode(
        &mut self,
        arguments: MatmulArguments<
            '_,
            Amdgpu,
            impl BufferRef<Backend = Amdgpu>,
            impl BufferRef<Backend = Amdgpu>,
            impl BufferMut<Backend = Amdgpu>,
            impl BufferRef<Backend = Amdgpu>,
        >,
        command_buffer: &mut Encoding,
    ) -> Result<(), AmdgpuError> {
        let shape = MatmulShape::from_arguments(&arguments);
        let gemv =
            GemvSpecialization::select(&shape, self.weights_data_type, self.input_data_type, self.output_data_type);
        let gemm_tiling =
            gemm::select_tiling(&shape, [self.weights_data_type, self.input_data_type, self.output_data_type]);
        let data_types = [self.weights_data_type, self.input_data_type, self.output_data_type];
        // prefill: the native W4 GEMM for the shapes it takes, the MSL GEMM for the rest
        if shape.m >= gemm_min_m(true) && gemm_w4::supports(&shape, data_types) && gemm_w4::aligned(&arguments) {
            trace_route(&shape, data_types, &"GemmW4");
            return gemm_w4::encode(arguments, command_buffer).map_err(AmdgpuError::from);
        }
        if let Some(tiling) = gemm_tiling
            && (shape.m >= gemm_min_m(shape.is_quant()) || gemv.is_none())
        {
            trace_route(&shape, data_types, &tiling);
            return self.gemm.encode(arguments, tiling, command_buffer);
        }
        // int8 activations come only where select_activation_format chose them, and only this kernel takes them
        if !shape.a_full_precision {
            if !qmv_wmma::supports(&shape, data_types) || !qmv_wmma::aligned(&arguments) {
                return Err(AmdgpuError::KernelDispatchFailed(
                    format!("no AMDGPU path for int8 activations at m={} n={} k={}", shape.m, shape.n, shape.k).into(),
                ));
            }
            trace_route(&shape, data_types, &"QmvWmma int8");
            return qmv_wmma::encode(arguments, command_buffer).map_err(AmdgpuError::from);
        }
        let (qmv_min_m, qmv_max_m) = qmv_wmma::m_range();
        if (qmv_min_m..=qmv_max_m).contains(&shape.m)
            && qmv_wmma::supports(&shape, data_types)
            && qmv_wmma::aligned(&arguments)
        {
            trace_route(&shape, data_types, &"QmvWmma");
            return qmv_wmma::encode(arguments, command_buffer).map_err(AmdgpuError::from);
        }
        if gemv_w4::supports(&shape, data_types) && gemv_w4::aligned(&arguments) {
            trace_route(&shape, data_types, &"GemvW4");
            return gemv_w4::encode(arguments, command_buffer).map_err(AmdgpuError::from);
        }
        trace_route(&shape, data_types, &gemv);
        let Some(specialization) = gemv else {
            return Err(AmdgpuError::KernelDispatchFailed(
                format!(
                    "no AMDGPU matmul path yet for m={} n={} k={} (b_transpose={}, quantized={})",
                    shape.m,
                    shape.n,
                    shape.k,
                    shape.b_transpose,
                    shape.is_quant()
                )
                .into(),
            ));
        };
        self.gemv.encode(arguments, specialization, &self.output_work, command_buffer).map_err(AmdgpuError::from)
    }
}
