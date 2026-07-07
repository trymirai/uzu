use thiserror::Error;

use crate::{
    array::size_for_shape,
    backends::common::{
        Allocation, Backend, Context, Encoder, Kernels,
        kernel::{
            Conv1dPackKernel, DeltaNetChunkedCumsumKernel, DeltaNetChunkedGramKernel, DeltaNetChunkedMegaApplyKernel,
            DeltaNetChunkedPrepKernel, DeltaNetChunkedSolveKernel, DeltaNetChunkedSolveTKernel, DeltaNetConvScanKernel,
            DeltaNetConvUpdateKernel, DeltaNetNormGateKernel, DeltaNetPrefillKernel, DeltaNetPrefillPrepKernel,
            DeltaNetUpdateKernel,
        },
    },
    config::token_mixer::delta_net::DeltaNetConfig,
    data_type::DataType,
    encodable_block::linear::{Linear, LinearBlockError},
    forward_pass::delta_net_layer::DeltaNetLayer,
    parameters::{ParameterLoaderError, ParameterTree},
};

#[derive(Debug, Error)]
pub enum DeltaNetMixerError<B: Backend> {
    #[error("Backend error: {0}")]
    BackendError(#[source] B::Error),
    #[error("Unsupported configuration: {0}")]
    UnsupportedConfiguration(String),
    #[error("Linear error: {0}")]
    InnerLinearError(#[from] LinearBlockError<B>),
    #[error("Parameter loader error: {0}")]
    ParameterLoaderError(#[from] ParameterLoaderError<B>),
}

/// Minimum prefill length (tokens) for chunked Mode-L with the MXU backend.
pub const CHUNKED_MXU_MIN_T: usize = 256;

/// Minimum prefill length for chunked Mode-L with the simdgroup backend.
pub const CHUNKED_SIMD_MIN_T: usize = 1024;

/// Chunk length of the Mode-L pipeline (tokens per chunk).
const CHUNKED_CHUNK_SIZE: usize = 64;
/// Block size of the triangular block-inverse solve inside Mode-L.
const CHUNKED_BLOCK_SIZE: usize = 16;
/// V-dimension tile of the MegaApply / SolveT kernels.
const CHUNKED_VT: usize = 32;

/// How GDN prefill should be executed for a given device + sequence length.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GdnPrefillPath {
    /// Recurrent scan (the always-available default). Used on M1/M2/CPU and for
    /// short sequences everywhere.
    Recurrent,
    /// Chunked Mode-L pipeline. `use_mxu` selects the MegaApply backend and MUST
    /// match the kernel that was actually constructed for the device
    /// (`ctx.supports_mxu()`).
    ChunkedModeL {
        use_mxu: bool,
    },
}

/// Device-tier + sequence-length router for GDN prefill. Generic over the
/// backend `Context` so it works for any `DeltaNetMixer<B>`.
///
/// - M5-class (MXU available): chunked Mode-L (`use_mxu=true`) for
///   `suffix_len >= CHUNKED_MXU_MIN_T`, else recurrent.
/// - Apple family 9 (M3/M4, dynamic caching, no MXU): chunked Mode-L
///   (`use_mxu=false`) for `suffix_len >= CHUNKED_SIMD_MIN_T`, else recurrent.
/// - Apple family <= 8 (M1/M2) and CPU: recurrent always (both predicates are
///   `false`).
pub fn select_gdn_prefill_path<C: Context>(
    context: &C,
    suffix_len: usize,
) -> GdnPrefillPath {
    route_gdn_prefill(context.supports_mxu(), context.supports_dynamic_caching(), suffix_len)
}

/// Pure decision core of [`select_gdn_prefill_path`], factored out so the full
/// device-tier truth table can be unit-tested without a live device (mirrors
/// `device_tier::device_tier_for`).
fn route_gdn_prefill(
    supports_mxu: bool,
    supports_dynamic_caching: bool,
    suffix_len: usize,
) -> GdnPrefillPath {
    if supports_dynamic_caching && supports_mxu && suffix_len >= CHUNKED_MXU_MIN_T {
        GdnPrefillPath::ChunkedModeL {
            use_mxu: true,
        }
    } else if supports_dynamic_caching && suffix_len >= CHUNKED_SIMD_MIN_T {
        GdnPrefillPath::ChunkedModeL {
            use_mxu: false,
        }
    } else {
        GdnPrefillPath::Recurrent
    }
}

pub struct DeltaNetMixer<B: Backend> {
    kernel_size: usize,
    num_heads: usize,
    num_groups: usize,
    value_head_dim: usize,
    key_dim: usize,
    value_dim: usize,
    conv_dim: usize,
    total_proj_dim: usize,
    norm_epsilon: f32,
    in_projection: Box<dyn Linear<B>>,
    out_projection: Box<dyn Linear<B>>,
    // Decode kernels
    conv_update: <B::Kernels as Kernels>::DeltaNetConvUpdateKernel,
    delta_net_update: <B::Kernels as Kernels>::DeltaNetUpdateKernel,
    // Prefill kernels
    conv_pack: <B::Kernels as Kernels>::Conv1dPackKernel,
    conv_scan: <B::Kernels as Kernels>::DeltaNetConvScanKernel,
    prefill_prep: <B::Kernels as Kernels>::DeltaNetPrefillPrepKernel,
    delta_net_prefill: <B::Kernels as Kernels>::DeltaNetPrefillKernel,
    norm_gate: <B::Kernels as Kernels>::DeltaNetNormGateKernel,
    // Chunked Mode-L prefill kernels. `Some` only on chunked-eligible devices
    // (Apple family 9+); `None` on M1/M2/CPU, where the router always selects
    // the recurrent path so these are never touched. MegaApply is constructed
    // with `use_mxu = ctx.supports_mxu()`, so the MXU PSO is never built on a
    // non-MXU device.
    chunked_prep: Option<<B::Kernels as Kernels>::DeltaNetChunkedPrepKernel>,
    chunked_cumsum: Option<<B::Kernels as Kernels>::DeltaNetChunkedCumsumKernel>,
    chunked_gram: Option<<B::Kernels as Kernels>::DeltaNetChunkedGramKernel>,
    chunked_solve: Option<<B::Kernels as Kernels>::DeltaNetChunkedSolveKernel>,
    chunked_solve_t: Option<<B::Kernels as Kernels>::DeltaNetChunkedSolveTKernel>,
    chunked_mega: Option<<B::Kernels as Kernels>::DeltaNetChunkedMegaApplyKernel>,
    // Parameters
    conv_weight: Allocation<B>,
    conv_bias: Option<Allocation<B>>,
    a_log: Allocation<B>,
    dt_bias: Allocation<B>,
    norm_weight: Allocation<B>,
    // Activation/scratch dtype (model dtype, e.g. bf16). The recurrent ssm_state
    // and small weights (a_log/dt_bias/norm/conv) stay f32 inside the kernels.
    outer_data_type: DataType,
}

pub struct DeltaNetArguments<'a, B: Backend> {
    pub active_row_count: usize,
    pub layer: &'a mut DeltaNetLayer<B>,
}

impl<B: Backend> DeltaNetMixer<B> {
    pub fn new(
        context: &B::Context,
        config: &DeltaNetConfig,
        model_dim: usize,
        parameter_tree: &ParameterTree<B>,
        outer_data_type: DataType,
    ) -> Result<(Self, Option<Allocation<B>>), DeltaNetMixerError<B>> {
        let inner_data_type = DataType::F32;

        if config.kernel_size < 2 {
            return Err(DeltaNetMixerError::UnsupportedConfiguration(format!(
                "kernel_size must be >= 2, got {}",
                config.kernel_size
            )));
        }
        if config.head_dim != 128 {
            return Err(DeltaNetMixerError::UnsupportedConfiguration(format!(
                "head_dim must be 128, got {}",
                config.head_dim
            )));
        }
        if config.value_head_dim != 128 {
            return Err(DeltaNetMixerError::UnsupportedConfiguration(format!(
                "value_head_dim must be 128, got {}",
                config.value_head_dim
            )));
        }

        let has_bias = config.conv_config.has_biases;
        let key_dim = config.key_dim();
        let value_dim = config.value_dim();
        let conv_dim = config.conv_dim();
        let total_proj_dim = config.total_proj_dim();

        // Load weights
        let conv_tree = parameter_tree.subtree("conv")?;

        let (in_projection, in_projection_input_hadamard_factors) =
            <dyn Linear<B>>::new_extracting_input_hadamard_mixed_precision(
                model_dim,
                [total_proj_dim],
                false,
                context,
                outer_data_type,
                outer_data_type,
                outer_data_type,
                &parameter_tree.subtree("in_proj")?,
            )?;

        let out_projection = <dyn Linear<B>>::new_mixed_precision(
            value_dim,
            [model_dim],
            false,
            context,
            outer_data_type,
            outer_data_type,
            outer_data_type,
            &parameter_tree.subtree("out_proj")?,
        )?;

        let conv_weight =
            conv_tree.leaf("weights")?.validate(&[conv_dim, config.kernel_size], inner_data_type)?.read_allocation()?;
        let conv_bias = if has_bias {
            Some(conv_tree.leaf("biases")?.validate(&[conv_dim], inner_data_type)?.read_allocation()?)
        } else {
            None
        };

        let a_log = parameter_tree.leaf("a_log")?.validate(&[config.num_heads], inner_data_type)?.read_allocation()?;
        let dt_bias =
            parameter_tree.leaf("dt_bias")?.validate(&[config.num_heads], inner_data_type)?.read_allocation()?;
        let norm_weight = parameter_tree
            .leaf("norm.scales")?
            .validate(&[config.value_head_dim], inner_data_type)?
            .read_allocation()?;

        // Create kernels
        let conv_update = <B::Kernels as Kernels>::DeltaNetConvUpdateKernel::new(context, outer_data_type, has_bias)
            .map_err(DeltaNetMixerError::BackendError)?;
        let delta_net_update =
            <B::Kernels as Kernels>::DeltaNetUpdateKernel::new(context, outer_data_type, config.head_dim as u32)
                .map_err(DeltaNetMixerError::BackendError)?;
        let conv_pack = <B::Kernels as Kernels>::Conv1dPackKernel::new(context, inner_data_type, outer_data_type)
            .map_err(DeltaNetMixerError::BackendError)?;
        let conv_scan = <B::Kernels as Kernels>::DeltaNetConvScanKernel::new(context, outer_data_type, has_bias)
            .map_err(DeltaNetMixerError::BackendError)?;
        let prefill_prep =
            <B::Kernels as Kernels>::DeltaNetPrefillPrepKernel::new(context, outer_data_type, config.head_dim as u32)
                .map_err(DeltaNetMixerError::BackendError)?;
        let delta_net_prefill =
            <B::Kernels as Kernels>::DeltaNetPrefillKernel::new(context, outer_data_type, config.head_dim as u32)
                .map_err(DeltaNetMixerError::BackendError)?;
        let norm_gate = <B::Kernels as Kernels>::DeltaNetNormGateKernel::new(context, outer_data_type)
            .map_err(DeltaNetMixerError::BackendError)?;

        // Chunked Mode-L kernels. Construct ONLY on chunked-eligible devices
        // (Apple family 9+). Crucially, MegaApply's USE_MXU flag mirrors the
        // device capability, so the MXU-specialized PSO is only ever built on a
        // device that actually has an MXU (M5+) — building it on M1-M4 would
        // fail PSO creation. On M1/M2/CPU everything below stays `None` and the
        // router falls back to the recurrent path.
        let chunked_eligible = context.supports_dynamic_caching();
        let (chunked_prep, chunked_cumsum, chunked_gram, chunked_solve, chunked_solve_t, chunked_mega) =
            if chunked_eligible {
                let use_mxu = context.supports_mxu();
                let prep = <B::Kernels as Kernels>::DeltaNetChunkedPrepKernel::new(
                    context,
                    outer_data_type,
                    config.head_dim as u32,
                )
                .map_err(DeltaNetMixerError::BackendError)?;
                let cumsum = <B::Kernels as Kernels>::DeltaNetChunkedCumsumKernel::new(context)
                    .map_err(DeltaNetMixerError::BackendError)?;
                let gram = <B::Kernels as Kernels>::DeltaNetChunkedGramKernel::new(
                    context,
                    config.head_dim as u32,
                    CHUNKED_CHUNK_SIZE as u32,
                )
                .map_err(DeltaNetMixerError::BackendError)?;
                let solve =
                    <B::Kernels as Kernels>::DeltaNetChunkedSolveKernel::new(context, CHUNKED_CHUNK_SIZE as u32, false)
                        .map_err(DeltaNetMixerError::BackendError)?;
                let solve_t = <B::Kernels as Kernels>::DeltaNetChunkedSolveTKernel::new(
                    context,
                    CHUNKED_CHUNK_SIZE as u32,
                    CHUNKED_VT as u32,
                )
                .map_err(DeltaNetMixerError::BackendError)?;
                let mega = <B::Kernels as Kernels>::DeltaNetChunkedMegaApplyKernel::new(
                    context,
                    outer_data_type,
                    outer_data_type,
                    CHUNKED_VT as u32,
                    use_mxu,
                )
                .map_err(DeltaNetMixerError::BackendError)?;
                (Some(prep), Some(cumsum), Some(gram), Some(solve), Some(solve_t), Some(mega))
            } else {
                (None, None, None, None, None, None)
            };

        Ok((
            Self {
                kernel_size: config.kernel_size,
                num_heads: config.num_heads,
                num_groups: config.num_groups,
                value_head_dim: config.value_head_dim,
                key_dim,
                value_dim,
                conv_dim,
                total_proj_dim,
                norm_epsilon: config.norm_config.epsilon,
                in_projection,
                out_projection,
                conv_update,
                delta_net_update,
                conv_pack,
                conv_scan,
                prefill_prep,
                delta_net_prefill,
                norm_gate,
                chunked_prep,
                chunked_cumsum,
                chunked_gram,
                chunked_solve,
                chunked_solve_t,
                chunked_mega,
                conv_weight,
                conv_bias,
                a_log,
                dt_bias,
                norm_weight,
                outer_data_type,
            },
            in_projection_input_hadamard_factors,
        ))
    }

    fn run_conv_update(
        &self,
        layer: &mut DeltaNetLayer<B>,
        in_proj: &mut Allocation<B>,
        encoder: &mut Encoder<B>,
    ) {
        let kernel_size = self.kernel_size;
        self.conv_update.encode(
            &self.conv_weight,
            self.conv_bias.as_ref(),
            in_proj,
            &mut layer.conv_state,
            kernel_size as u32,
            self.conv_dim as u32,
            (kernel_size - 1) as u32,
            encoder,
        );
    }

    fn run_conv_scan(
        &self,
        layer: &mut DeltaNetLayer<B>,
        in_proj: &mut Allocation<B>,
        encoder: &mut Encoder<B>,
        suffix_length: usize,
    ) -> Result<(), B::Error> {
        let kernel_size = self.kernel_size;
        let state_stride = kernel_size - 1;
        let conv_dim = self.conv_dim;
        let total_proj_dim = self.total_proj_dim;
        let mut padded =
            encoder.allocate_scratch(size_for_shape(&[suffix_length + state_stride, total_proj_dim], DataType::F32))?;

        self.conv_pack.encode(
            &layer.conv_state,
            &*in_proj,
            &mut padded,
            state_stride as u32,
            total_proj_dim as u32,
            suffix_length as u32,
            conv_dim as u32,
            encoder,
        );

        self.conv_scan.encode(
            &padded,
            &self.conv_weight,
            self.conv_bias.as_ref(),
            in_proj,
            &mut layer.conv_state,
            suffix_length as u32,
            kernel_size as u32,
            total_proj_dim as u32,
            state_stride as u32,
            conv_dim as u32,
            total_proj_dim as u32,
            encoder,
        );
        Ok(())
    }

    fn run_delta_rule(
        &self,
        layer: &mut DeltaNetLayer<B>,
        in_proj: &Allocation<B>,
        active_row_count: usize,
        encoder: &mut Encoder<B>,
    ) -> Result<Allocation<B>, B::Error> {
        let mut out =
            encoder.allocate_scratch(size_for_shape(&[active_row_count, self.value_dim], self.outer_data_type))?;
        self.delta_net_update.encode(
            in_proj,
            &self.a_log,
            &self.dt_bias,
            &self.norm_weight,
            &mut layer.ssm_state,
            &mut out,
            self.num_heads as u32,
            self.num_groups as u32,
            self.value_head_dim as u32,
            self.key_dim as u32,
            self.value_dim as u32,
            self.norm_epsilon,
            encoder,
        );
        Ok(out)
    }

    fn run_delta_rule_prefill(
        &self,
        layer: &mut DeltaNetLayer<B>,
        in_proj: &Allocation<B>,
        suffix_length: usize,
        encoder: &mut Encoder<B>,
    ) -> Result<Allocation<B>, B::Error> {
        let num_dv_groups = self.value_head_dim.div_ceil(16) as u32;
        let mut prep_q_norm =
            encoder.allocate_scratch(size_for_shape(&[suffix_length * self.key_dim], DataType::F32))?;
        let mut prep_k_norm =
            encoder.allocate_scratch(size_for_shape(&[suffix_length * self.key_dim], DataType::F32))?;
        let mut prep_beta =
            encoder.allocate_scratch(size_for_shape(&[suffix_length * self.num_heads], DataType::F32))?;
        let mut prep_decay =
            encoder.allocate_scratch(size_for_shape(&[suffix_length * self.num_heads], DataType::F32))?;

        self.prefill_prep.encode(
            in_proj,
            &self.a_log,
            &self.dt_bias,
            &mut prep_q_norm,
            &mut prep_k_norm,
            &mut prep_beta,
            &mut prep_decay,
            self.num_heads as u32,
            self.num_groups as u32,
            self.key_dim as u32,
            self.value_dim as u32,
            suffix_length as u32,
            encoder,
        );

        let mut out =
            encoder.allocate_scratch(size_for_shape(&[suffix_length, self.value_dim], self.outer_data_type))?;
        self.delta_net_prefill.encode(
            &prep_q_norm,
            &prep_k_norm,
            &prep_beta,
            &prep_decay,
            in_proj,
            &mut layer.ssm_state,
            &mut out,
            self.num_heads as u32,
            self.num_groups as u32,
            self.value_head_dim as u32,
            self.key_dim as u32,
            self.value_dim as u32,
            suffix_length as u32,
            num_dv_groups,
            encoder,
        );

        self.norm_gate.encode(
            &mut out,
            in_proj,
            &self.norm_weight,
            self.num_heads as u32,
            self.value_head_dim as u32,
            self.value_dim as u32,
            self.conv_dim as u32,
            self.total_proj_dim as u32,
            self.norm_epsilon,
            suffix_length as u32,
            encoder,
        );
        Ok(out)
    }

    /// Chunked Mode-L prefill. Same state-in/state-out contract as
    /// `run_delta_rule_prefill`: the initial state is read from
    /// `layer.ssm_state`, and the final chunk state is written back to it
    /// in-place (MegaApply reads S0 and writes the final S over the same
    /// buffer), so decode continues correctly afterward. All intermediates are
    /// pooled scratch — no new persistent buffers.
    ///
    /// The MegaApply kernel is passed in so callers (production vs. tests) can
    /// select the MXU or simdgroup backend; production passes
    /// `self.chunked_mega`, which was built to match the device.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn run_delta_rule_prefill_chunked(
        &self,
        mega: &<B::Kernels as Kernels>::DeltaNetChunkedMegaApplyKernel,
        layer: &mut DeltaNetLayer<B>,
        in_proj: &Allocation<B>,
        suffix_length: usize,
        encoder: &mut Encoder<B>,
    ) -> Result<Allocation<B>, B::Error> {
        let prep = self.chunked_prep.as_ref().expect("chunked prep kernel present when routed to chunked path");
        let cumsum = self.chunked_cumsum.as_ref().expect("chunked cumsum kernel present when routed to chunked path");
        let gram = self.chunked_gram.as_ref().expect("chunked gram kernel present when routed to chunked path");
        let solve = self.chunked_solve.as_ref().expect("chunked solve kernel present when routed to chunked path");
        let solve_t = self.chunked_solve_t.as_ref().expect("chunked solveT kernel present when routed to chunked path");

        let num_v_heads = self.num_heads;
        let num_k_heads = self.num_groups;
        let head_v_dim = self.value_head_dim;
        let key_dim = self.key_dim;
        let value_dim = self.value_dim;

        let chunk_size = CHUNKED_CHUNK_SIZE;
        let block_size = CHUNKED_BLOCK_SIZE;
        let num_chunks = suffix_length.div_ceil(chunk_size);
        let num_blocks = chunk_size.div_ceil(block_size);
        let num_col_pairs = num_blocks.div_ceil(2);

        // Prep outputs (per-token, f32).
        let mut q_norm = encoder.allocate_scratch(size_for_shape(&[suffix_length * key_dim], DataType::F32))?;
        let mut k_norm = encoder.allocate_scratch(size_for_shape(&[suffix_length * key_dim], DataType::F32))?;
        let mut beta = encoder.allocate_scratch(size_for_shape(&[suffix_length * num_v_heads], DataType::F32))?;
        let mut log_decay = encoder.allocate_scratch(size_for_shape(&[suffix_length * num_v_heads], DataType::F32))?;
        // Chunk-local cumulative log-decay prefix (Cumsum output, f32).
        let mut g = encoder.allocate_scratch(size_for_shape(&[suffix_length * num_v_heads], DataType::F32))?;
        // Gram outputs.
        let mut kk = encoder
            .allocate_scratch(size_for_shape(&[num_chunks * num_k_heads * chunk_size * chunk_size], DataType::F32))?;
        let mut qk = encoder
            .allocate_scratch(size_for_shape(&[num_chunks * num_v_heads * chunk_size * chunk_size], DataType::F32))?;
        // Solve outputs (block-packed strips + diagonal block inverses).
        let mut a_packed = encoder.allocate_scratch(size_for_shape(
            &[num_chunks * num_v_heads * num_blocks * num_col_pairs * block_size * 2 * block_size],
            DataType::F32,
        ))?;
        let mut a_inv = encoder.allocate_scratch(size_for_shape(
            &[num_chunks * num_v_heads * num_blocks * block_size * block_size],
            DataType::F32,
        ))?;
        // Dense unit-lower-triangular inverse T per (chunk, v-head), bf16.
        let mut t_mat = encoder
            .allocate_scratch(size_for_shape(&[num_chunks * num_v_heads * chunk_size * chunk_size], DataType::BF16))?;

        let mut out =
            encoder.allocate_scratch(size_for_shape(&[suffix_length, self.value_dim], self.outer_data_type))?;

        prep.encode(
            in_proj,
            &self.a_log,
            &self.dt_bias,
            &mut q_norm,
            &mut k_norm,
            &mut beta,
            &mut log_decay,
            num_v_heads as u32,
            num_k_heads as u32,
            key_dim as u32,
            value_dim as u32,
            suffix_length as u32,
            encoder,
        );
        cumsum.encode(&log_decay, &mut g, num_v_heads as u32, suffix_length as u32, chunk_size as u32, encoder);
        gram.encode(
            &q_norm,
            &k_norm,
            &g,
            &mut kk,
            &mut qk,
            num_v_heads as u32,
            num_k_heads as u32,
            key_dim as u32,
            suffix_length as u32,
            encoder,
        );
        solve.encode(
            &kk,
            &beta,
            &g,
            &mut a_packed,
            &mut a_inv,
            num_v_heads as u32,
            num_k_heads as u32,
            suffix_length as u32,
            encoder,
        );
        solve_t.encode(&a_packed, &a_inv, &mut t_mat, num_v_heads as u32, suffix_length as u32, encoder);
        mega.encode(
            &q_norm,
            &k_norm,
            in_proj,
            &qk,
            &t_mat,
            &g,
            &beta,
            &mut layer.ssm_state,
            &mut out,
            num_v_heads as u32,
            num_k_heads as u32,
            head_v_dim as u32,
            key_dim as u32,
            value_dim as u32,
            suffix_length as u32,
            encoder,
        );

        self.norm_gate.encode(
            &mut out,
            in_proj,
            &self.norm_weight,
            self.num_heads as u32,
            self.value_head_dim as u32,
            self.value_dim as u32,
            self.conv_dim as u32,
            self.total_proj_dim as u32,
            self.norm_epsilon,
            suffix_length as u32,
            encoder,
        );
        Ok(out)
    }

    pub fn encode(
        &self,
        args: DeltaNetArguments<B>,
        input: Allocation<B>,
        encoder: &mut Encoder<B>,
    ) -> Result<Allocation<B>, B::Error> {
        let DeltaNetArguments {
            active_row_count,
            layer,
        } = args;
        assert!(active_row_count > 0, "DeltaNet mixer requires at least one active row");

        let mut in_proj = self.in_projection.encode(input, active_row_count, encoder)?;

        let delta_output = if active_row_count == 1 {
            self.run_conv_update(layer, &mut in_proj, encoder);
            self.run_delta_rule(layer, &in_proj, active_row_count, encoder)?
        } else {
            self.run_conv_scan(layer, &mut in_proj, encoder, active_row_count)?;
            match select_gdn_prefill_path(encoder.context(), active_row_count) {
                GdnPrefillPath::ChunkedModeL {
                    ..
                } => {
                    // Router only returns ChunkedModeL on chunked-eligible
                    // devices, where `chunked_mega` (and the rest) are `Some`.
                    let mega = self
                        .chunked_mega
                        .as_ref()
                        .expect("chunked mega kernel present when router selects the chunked path");
                    self.run_delta_rule_prefill_chunked(mega, layer, &in_proj, active_row_count, encoder)?
                },
                GdnPrefillPath::Recurrent => self.run_delta_rule_prefill(layer, &in_proj, active_row_count, encoder)?,
            }
        };

        self.out_projection.encode(delta_output, active_row_count, encoder)
    }
}

#[cfg(test)]
mod router_tests {
    use proc_macros::uzu_test;

    use super::{CHUNKED_MXU_MIN_T, CHUNKED_SIMD_MIN_T, GdnPrefillPath, route_gdn_prefill};

    const MXU: GdnPrefillPath = GdnPrefillPath::ChunkedModeL {
        use_mxu: true,
    };
    const SIMD: GdnPrefillPath = GdnPrefillPath::ChunkedModeL {
        use_mxu: false,
    };
    const REC: GdnPrefillPath = GdnPrefillPath::Recurrent;

    // M5-class: MXU available (implies dynamic caching). Chunked+MXU at
    // T >= CHUNKED_MXU_MIN_T, recurrent below.
    #[uzu_test]
    fn m5_class_routes_mxu_above_threshold() {
        let mxu = true;
        let dyn_cache = true;
        assert_eq!(route_gdn_prefill(mxu, dyn_cache, 1), REC);
        assert_eq!(route_gdn_prefill(mxu, dyn_cache, CHUNKED_MXU_MIN_T - 1), REC);
        assert_eq!(route_gdn_prefill(mxu, dyn_cache, CHUNKED_MXU_MIN_T), MXU);
        assert_eq!(route_gdn_prefill(mxu, dyn_cache, CHUNKED_SIMD_MIN_T), MXU);
        assert_eq!(route_gdn_prefill(mxu, dyn_cache, 32768), MXU);
    }

    // Apple family 9 (M3/M4): dynamic caching, no MXU. Chunked+simd at
    // T >= CHUNKED_SIMD_MIN_T, recurrent below (incl. the whole MXU window).
    #[uzu_test]
    fn family9_routes_simd_above_larger_threshold() {
        let mxu = false;
        let dyn_cache = true;
        assert_eq!(route_gdn_prefill(mxu, dyn_cache, 1), REC);
        assert_eq!(route_gdn_prefill(mxu, dyn_cache, CHUNKED_MXU_MIN_T), REC);
        assert_eq!(route_gdn_prefill(mxu, dyn_cache, CHUNKED_SIMD_MIN_T - 1), REC);
        assert_eq!(route_gdn_prefill(mxu, dyn_cache, CHUNKED_SIMD_MIN_T), SIMD);
        assert_eq!(route_gdn_prefill(mxu, dyn_cache, 32768), SIMD);
    }

    // Apple family <= 8 (M1/M2) and CPU: neither predicate holds -> recurrent
    // ALWAYS. This is the no-op guarantee for pre-family-9 devices.
    #[uzu_test]
    fn pre_family9_and_cpu_always_recurrent() {
        let mxu = false;
        let dyn_cache = false;
        for &t in &[1usize, 63, 256, 512, 1024, 4096, 32768, usize::MAX] {
            assert_eq!(route_gdn_prefill(mxu, dyn_cache, t), REC, "T={t}");
        }
    }

    #[uzu_test]
    fn incoherent_mxu_without_dynamic_caching_stays_recurrent() {
        let mxu = true;
        let dyn_cache = false;
        for &t in &[CHUNKED_MXU_MIN_T, CHUNKED_SIMD_MIN_T, 32768] {
            assert_eq!(route_gdn_prefill(mxu, dyn_cache, t), REC, "T={t}");
        }
    }
}
