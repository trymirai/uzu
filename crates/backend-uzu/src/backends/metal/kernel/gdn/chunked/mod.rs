use super::super::{
    DeltaNetChunkedCumsumMetalKernel, DeltaNetChunkedDenseCausalInverseMetalKernel, DeltaNetChunkedGramMetalKernel,
    DeltaNetChunkedOutputAndStateMetalKernel, DeltaNetChunkedPackedAAndDiaInvMetalKernel,
    DeltaNetPrefillPrepMetalKernel,
};
use crate::{
    array::size_for_shape,
    backends::{
        common::{Backend, Encoder, kernel::DeltaNetPrefillPrepKernel},
        metal::{Metal, MetalContext},
    },
    data_type::DataType,
    encodable_block::mixer::delta_net::chunked_prefill::{DeltaNetChunkedPrefill, DeltaNetChunkedPrefillArgs},
};

const MXU_MIN_T: usize = 256;
const SIMD_MIN_T: usize = 1024;
const CHUNK_SIZE: usize = 64;
const BLOCK_SIZE: usize = 16;
const VT: usize = 32;

pub struct MetalDeltaNetChunkedPrefill {
    min_t: usize,
    prep: DeltaNetPrefillPrepMetalKernel,
    cumsum: DeltaNetChunkedCumsumMetalKernel,
    gram: DeltaNetChunkedGramMetalKernel,
    packed_a_and_dia_inv: DeltaNetChunkedPackedAAndDiaInvMetalKernel,
    dense_causal_inverse: DeltaNetChunkedDenseCausalInverseMetalKernel,
    output_and_state: DeltaNetChunkedOutputAndStateMetalKernel,
}

impl DeltaNetChunkedPrefill<Metal> for MetalDeltaNetChunkedPrefill {
    fn new(
        context: &MetalContext,
        outer_data_type: DataType,
        head_dim: u32,
    ) -> Result<Option<Self>, <Metal as Backend>::Error> {
        if outer_data_type == DataType::F16 {
            return Ok(None);
        }

        let use_mxu = context.supports_mxu();
        let min_t = if use_mxu {
            MXU_MIN_T
        } else if context.supports_dynamic_caching() {
            SIMD_MIN_T
        } else {
            return Ok(None);
        };

        Ok(Some(Self {
            min_t,
            prep: DeltaNetPrefillPrepMetalKernel::new(context, outer_data_type, head_dim, true)?,
            cumsum: DeltaNetChunkedCumsumMetalKernel::new(context, CHUNK_SIZE as u32)?,
            gram: DeltaNetChunkedGramMetalKernel::new(context, head_dim, CHUNK_SIZE as u32)?,
            packed_a_and_dia_inv: DeltaNetChunkedPackedAAndDiaInvMetalKernel::new(context, CHUNK_SIZE as u32)?,
            dense_causal_inverse: DeltaNetChunkedDenseCausalInverseMetalKernel::new(
                context,
                CHUNK_SIZE as u32,
                VT as u32,
            )?,
            output_and_state: DeltaNetChunkedOutputAndStateMetalKernel::new(
                context,
                outer_data_type,
                outer_data_type,
                VT as u32,
                use_mxu,
            )?,
        }))
    }

    fn should_use(
        &self,
        suffix_len: usize,
    ) -> bool {
        suffix_len >= self.min_t
    }

    fn encode(
        &self,
        args: DeltaNetChunkedPrefillArgs<'_, Metal>,
        encoder: &mut Encoder<Metal>,
    ) -> Result<(), <Metal as Backend>::Error> {
        let suffix_len = args.suffix_len;
        let num_chunks = suffix_len.div_ceil(CHUNK_SIZE);
        let num_blocks = CHUNK_SIZE.div_ceil(BLOCK_SIZE);
        let num_col_pairs = num_blocks.div_ceil(2);

        let mut q_norm =
            encoder.allocate_scratch(size_for_shape(&[suffix_len * args.key_dim as usize], DataType::F32))?;
        let mut k_norm =
            encoder.allocate_scratch(size_for_shape(&[suffix_len * args.key_dim as usize], DataType::F32))?;
        let mut beta =
            encoder.allocate_scratch(size_for_shape(&[suffix_len * args.num_heads as usize], DataType::F32))?;
        let mut log_decay =
            encoder.allocate_scratch(size_for_shape(&[suffix_len * args.num_heads as usize], DataType::F32))?;
        let mut g = encoder.allocate_scratch(size_for_shape(&[suffix_len * args.num_heads as usize], DataType::F32))?;
        let mut kk = encoder.allocate_scratch(size_for_shape(
            &[num_chunks * args.num_groups as usize * CHUNK_SIZE * CHUNK_SIZE],
            DataType::F32,
        ))?;
        let mut qk = encoder.allocate_scratch(size_for_shape(
            &[num_chunks * args.num_heads as usize * CHUNK_SIZE * CHUNK_SIZE],
            DataType::F32,
        ))?;
        let mut a_packed = encoder.allocate_scratch(size_for_shape(
            &[num_chunks * args.num_heads as usize * num_blocks * num_col_pairs * BLOCK_SIZE * 2 * BLOCK_SIZE],
            DataType::F32,
        ))?;
        let mut a_inv = encoder.allocate_scratch(size_for_shape(
            &[num_chunks * args.num_heads as usize * num_blocks * BLOCK_SIZE * BLOCK_SIZE],
            DataType::F32,
        ))?;
        let mut t_mat = encoder.allocate_scratch(size_for_shape(
            &[num_chunks * args.num_heads as usize * CHUNK_SIZE * CHUNK_SIZE],
            DataType::BF16,
        ))?;

        self.prep.encode(
            args.in_projected,
            args.a_log,
            args.dt_bias,
            &mut q_norm,
            &mut k_norm,
            &mut beta,
            &mut log_decay,
            args.num_heads,
            args.num_groups,
            args.key_dim,
            args.value_dim,
            suffix_len as u32,
            encoder,
        );
        self.cumsum.encode(&log_decay, &mut g, args.num_heads, suffix_len as u32, encoder);
        self.gram.encode(
            &q_norm,
            &k_norm,
            &g,
            &mut kk,
            &mut qk,
            args.num_heads,
            args.num_groups,
            args.key_dim,
            suffix_len as u32,
            encoder,
        );
        self.packed_a_and_dia_inv.encode(
            &kk,
            &beta,
            &g,
            &mut a_packed,
            &mut a_inv,
            args.num_heads,
            args.num_groups,
            suffix_len as u32,
            encoder,
        );
        self.dense_causal_inverse.encode(&a_packed, &a_inv, &mut t_mat, args.num_heads, suffix_len as u32, encoder);
        self.output_and_state.encode(
            &q_norm,
            &k_norm,
            args.in_projected,
            &qk,
            &t_mat,
            &g,
            &beta,
            args.ssm_state,
            args.delta_output,
            args.num_heads,
            args.num_groups,
            args.value_head_dim,
            args.key_dim,
            args.value_dim,
            suffix_len as u32,
            encoder,
        );
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use half::bf16;
    use proc_macros::uzu_test;

    use super::*;
    use crate::{
        backends::{
            common::{
                Context, Kernels,
                kernel::{DeltaNetNormGateKernel, DeltaNetPrefillKernel, DeltaNetPrefillPrepKernel},
            },
            cpu::Cpu,
        },
        tests::helpers::{alloc_allocation, alloc_allocation_with_data, allocation_to_vec},
    };

    const TEST_NUM_V_HEADS: usize = 4;
    const TEST_NUM_K_HEADS: usize = 1;
    const TEST_HEAD_K_DIM: usize = 128;
    const TEST_HEAD_V_DIM: usize = 32;
    const TEST_SUFFIX_LEN: usize = CHUNK_SIZE + 1;

    struct TestCase {
        in_proj: Vec<f32>,
        a_log: Vec<f32>,
        dt_bias: Vec<f32>,
        norm_weight: Vec<f32>,
        state: Vec<f32>,
        key_dim: usize,
        value_dim: usize,
        conv_dim: usize,
        total_proj_dim: usize,
    }

    fn test_case() -> TestCase {
        let key_dim = TEST_NUM_K_HEADS * TEST_HEAD_K_DIM;
        let value_dim = TEST_NUM_V_HEADS * TEST_HEAD_V_DIM;
        let conv_dim = 2 * key_dim + value_dim;
        let total_proj_dim = conv_dim + value_dim + TEST_NUM_V_HEADS + TEST_NUM_V_HEADS;
        let state_len = TEST_NUM_V_HEADS * TEST_HEAD_V_DIM * TEST_HEAD_K_DIM;

        TestCase {
            in_proj: (0..TEST_SUFFIX_LEN * total_proj_dim).map(|i| ((i % 37) as f32) * 0.02 - 0.3).collect(),
            a_log: (0..TEST_NUM_V_HEADS).map(|i| -1.5 + (i as f32) * 0.05).collect(),
            dt_bias: (0..TEST_NUM_V_HEADS).map(|i| 0.3 + (i as f32) * 0.02).collect(),
            norm_weight: (0..TEST_HEAD_V_DIM).map(|i| 0.9 + (i as f32) * 0.001).collect(),
            state: (0..state_len).map(|i| ((i % 29) as f32) * 0.005 - 0.05).collect(),
            key_dim,
            value_dim,
            conv_dim,
            total_proj_dim,
        }
    }

    fn run_cpu_legacy(case: &TestCase) -> (Vec<f32>, Vec<f32>) {
        let context = <Cpu as Backend>::Context::new().expect("cpu context");
        let in_proj = alloc_allocation_with_data::<Cpu, f32>(&context, &case.in_proj);
        let a_log = alloc_allocation_with_data::<Cpu, f32>(&context, &case.a_log);
        let dt_bias = alloc_allocation_with_data::<Cpu, f32>(&context, &case.dt_bias);
        let norm_weight = alloc_allocation_with_data::<Cpu, f32>(&context, &case.norm_weight);
        let mut state = alloc_allocation_with_data::<Cpu, f32>(&context, &case.state);
        let mut out = alloc_allocation::<Cpu, f32>(&context, TEST_SUFFIX_LEN * case.value_dim);
        let mut q_norm = alloc_allocation::<Cpu, f32>(&context, TEST_SUFFIX_LEN * case.key_dim);
        let mut k_norm = alloc_allocation::<Cpu, f32>(&context, TEST_SUFFIX_LEN * case.key_dim);
        let mut beta = alloc_allocation::<Cpu, f32>(&context, TEST_SUFFIX_LEN * TEST_NUM_V_HEADS);
        let mut decay = alloc_allocation::<Cpu, f32>(&context, TEST_SUFFIX_LEN * TEST_NUM_V_HEADS);

        let prep = <<Cpu as Backend>::Kernels as Kernels>::DeltaNetPrefillPrepKernel::new(
            &context,
            DataType::F32,
            TEST_HEAD_K_DIM as u32,
            false,
        )
        .expect("cpu prep");
        let prefill = <<Cpu as Backend>::Kernels as Kernels>::DeltaNetPrefillKernel::new(
            &context,
            DataType::F32,
            TEST_HEAD_K_DIM as u32,
        )
        .expect("cpu prefill");
        let norm = <<Cpu as Backend>::Kernels as Kernels>::DeltaNetNormGateKernel::new(&context, DataType::F32)
            .expect("cpu norm");

        let mut encoder = Encoder::new(context.as_ref()).expect("cpu encoder");
        prep.encode(
            &in_proj,
            &a_log,
            &dt_bias,
            &mut q_norm,
            &mut k_norm,
            &mut beta,
            &mut decay,
            TEST_NUM_V_HEADS as u32,
            TEST_NUM_K_HEADS as u32,
            case.key_dim as u32,
            case.value_dim as u32,
            TEST_SUFFIX_LEN as u32,
            &mut encoder,
        );
        prefill.encode(
            &q_norm,
            &k_norm,
            &beta,
            &decay,
            &in_proj,
            &mut state,
            &mut out,
            TEST_NUM_V_HEADS as u32,
            TEST_NUM_K_HEADS as u32,
            TEST_HEAD_V_DIM as u32,
            case.key_dim as u32,
            case.value_dim as u32,
            TEST_SUFFIX_LEN as u32,
            TEST_HEAD_V_DIM.div_ceil(16) as u32,
            &mut encoder,
        );
        norm.encode(
            &mut out,
            &in_proj,
            &norm_weight,
            TEST_NUM_V_HEADS as u32,
            TEST_HEAD_V_DIM as u32,
            case.value_dim as u32,
            case.conv_dim as u32,
            case.total_proj_dim as u32,
            1e-6f32,
            TEST_SUFFIX_LEN as u32,
            &mut encoder,
        );
        encoder.end_encoding().submit().wait_until_completed().unwrap();

        (allocation_to_vec(&out), allocation_to_vec(&state))
    }

    fn run_metal_chunked(
        case: &TestCase,
        use_mxu: bool,
    ) -> (Vec<f32>, Vec<f32>) {
        let context = <Metal as Backend>::Context::new().expect("metal context");
        let in_proj = alloc_allocation_with_data::<Metal, f32>(&context, &case.in_proj);
        let a_log = alloc_allocation_with_data::<Metal, f32>(&context, &case.a_log);
        let dt_bias = alloc_allocation_with_data::<Metal, f32>(&context, &case.dt_bias);
        let norm_weight = alloc_allocation_with_data::<Metal, f32>(&context, &case.norm_weight);
        let mut state = alloc_allocation_with_data::<Metal, f32>(&context, &case.state);
        let mut out = alloc_allocation::<Metal, f32>(&context, TEST_SUFFIX_LEN * case.value_dim);

        let chunked = MetalDeltaNetChunkedPrefill {
            min_t: 0,
            prep: DeltaNetPrefillPrepMetalKernel::new(&context, DataType::F32, TEST_HEAD_K_DIM as u32, true)
                .expect("metal prep"),
            cumsum: DeltaNetChunkedCumsumMetalKernel::new(&context, CHUNK_SIZE as u32).expect("metal cumsum"),
            gram: DeltaNetChunkedGramMetalKernel::new(&context, TEST_HEAD_K_DIM as u32, CHUNK_SIZE as u32)
                .expect("metal gram"),
            packed_a_and_dia_inv: DeltaNetChunkedPackedAAndDiaInvMetalKernel::new(&context, CHUNK_SIZE as u32)
                .expect("metal packed_a_and_dia_inv"),
            dense_causal_inverse: DeltaNetChunkedDenseCausalInverseMetalKernel::new(
                &context,
                CHUNK_SIZE as u32,
                VT as u32,
            )
            .expect("metal dense_causal_inverse"),
            output_and_state: DeltaNetChunkedOutputAndStateMetalKernel::new(
                &context,
                DataType::F32,
                DataType::F32,
                VT as u32,
                use_mxu,
            )
            .expect("metal output_and_state"),
        };
        let norm = <<Metal as Backend>::Kernels as Kernels>::DeltaNetNormGateKernel::new(&context, DataType::F32)
            .expect("metal norm");

        let mut encoder = Encoder::new(context.as_ref()).expect("metal encoder");
        chunked
            .encode(
                DeltaNetChunkedPrefillArgs {
                    in_projected: &in_proj,
                    a_log: &a_log,
                    dt_bias: &dt_bias,
                    ssm_state: &mut state,
                    delta_output: &mut out,
                    num_heads: TEST_NUM_V_HEADS as u32,
                    num_groups: TEST_NUM_K_HEADS as u32,
                    value_head_dim: TEST_HEAD_V_DIM as u32,
                    key_dim: case.key_dim as u32,
                    value_dim: case.value_dim as u32,
                    suffix_len: TEST_SUFFIX_LEN,
                },
                &mut encoder,
            )
            .expect("chunked encode");
        norm.encode(
            &mut out,
            &in_proj,
            &norm_weight,
            TEST_NUM_V_HEADS as u32,
            TEST_HEAD_V_DIM as u32,
            case.value_dim as u32,
            case.conv_dim as u32,
            case.total_proj_dim as u32,
            1e-6f32,
            TEST_SUFFIX_LEN as u32,
            &mut encoder,
        );
        encoder.end_encoding().submit().wait_until_completed().unwrap();

        (allocation_to_vec(&out), allocation_to_vec(&state))
    }

    fn assert_close(
        actual: &[f32],
        expected: &[f32],
        label: &str,
    ) {
        assert_eq!(actual.len(), expected.len(), "{label}: length mismatch");
        assert!(
            is_close::default().abs_tol(1e-3).rel_tol(5e-2).all_close(actual.iter().copied(), expected.iter().copied()),
            "{label}: values not close enough"
        );
    }

    #[uzu_test]
    fn delta_net_chunked_prefill_matches_cpu_legacy_tail_chunk() {
        let case = test_case();
        let (cpu_out, cpu_state) = run_cpu_legacy(&case);

        let (simd_out, simd_state) = run_metal_chunked(&case, false);
        assert_close(&simd_out, &cpu_out, "simd output");
        assert_close(&simd_state, &cpu_state, "simd state");

        let context = <Metal as Backend>::Context::new().expect("metal context");
        if context.supports_mxu() {
            let (mxu_out, mxu_state) = run_metal_chunked(&case, true);
            assert_close(&mxu_out, &cpu_out, "mxu output");
            assert_close(&mxu_state, &cpu_state, "mxu state");
        }
    }

    #[uzu_test]
    #[ignore]
    fn bench_delta_net_chunked_prefill_e2e() {
        use std::time::{Duration, Instant};

        use test_runner::perf::run_perf_with_warmup;

        let num_v_heads = 48usize;
        let num_k_heads = 16usize;
        let head_k_dim = 128usize;
        let head_v_dim = 128usize;
        let key_dim = num_k_heads * head_k_dim;
        let value_dim = num_v_heads * head_v_dim;
        let conv_dim = 2 * key_dim + value_dim;
        let total_proj_dim = conv_dim + value_dim + num_v_heads + num_v_heads;
        let state_size = num_v_heads * head_v_dim * head_k_dim;
        let num_dv_groups = head_v_dim.div_ceil(16);
        let num_blocks = CHUNK_SIZE.div_ceil(BLOCK_SIZE);
        let num_col_pairs = num_blocks.div_ceil(2);

        eprintln!("\n=== DeltaNet chunked prefill e2e (preallocated scratch) ===");
        eprintln!("BENCH\tT\tpath\tmedian_ms\tmin_ms\tmax_ms\tstd_ms\tspeedup_vs_recurrent");

        for suffix_len in [32usize, 64, 128, 256, 512, 1024, 4096, 8192, 32768] {
            let context = <Metal as Backend>::Context::new().expect("metal context");
            let num_chunks = suffix_len.div_ceil(CHUNK_SIZE);

            let in_proj: Vec<bf16> =
                (0..suffix_len * total_proj_dim).map(|i| bf16::from_f32(((i % 23) as f32 - 11.0) * 0.002)).collect();
            let a_log: Vec<f32> = (0..num_v_heads).map(|i| -1.5 + (i as f32) * 0.01).collect();
            let dt_bias: Vec<f32> = (0..num_v_heads).map(|i| 0.1 + (i as f32) * 0.001).collect();
            let norm_weight: Vec<f32> = (0..head_v_dim).map(|i| 0.9 + (i as f32) * 0.001).collect();

            let in_proj_array = alloc_allocation_with_data::<Metal, bf16>(&context, &in_proj);
            let a_log_array = alloc_allocation_with_data::<Metal, f32>(&context, &a_log);
            let dt_bias_array = alloc_allocation_with_data::<Metal, f32>(&context, &dt_bias);
            let norm_weight_array = alloc_allocation_with_data::<Metal, f32>(&context, &norm_weight);
            let alloc = |data_type, len| match data_type {
                DataType::BF16 => alloc_allocation::<Metal, bf16>(&context, len),
                DataType::F32 => alloc_allocation::<Metal, f32>(&context, len),
                _ => unreachable!("bench only allocates BF16/F32 buffers"),
            };

            let mut rec_out = alloc(DataType::BF16, suffix_len * value_dim);
            let mut rec_q = alloc(DataType::F32, suffix_len * key_dim);
            let mut rec_k = alloc(DataType::F32, suffix_len * key_dim);
            let mut rec_beta = alloc(DataType::F32, suffix_len * num_v_heads);
            let mut rec_decay = alloc(DataType::F32, suffix_len * num_v_heads);
            let mut rec_state = alloc(DataType::F32, state_size);

            let mut q_m = alloc(DataType::F32, suffix_len * key_dim);
            let mut k_m = alloc(DataType::F32, suffix_len * key_dim);
            let mut beta_m = alloc(DataType::F32, suffix_len * num_v_heads);
            let mut log_decay_m = alloc(DataType::F32, suffix_len * num_v_heads);
            let mut g_m = alloc(DataType::F32, suffix_len * num_v_heads);
            let mut kk_m = alloc(DataType::F32, num_chunks * num_k_heads * CHUNK_SIZE * CHUNK_SIZE);
            let mut qk_m = alloc(DataType::F32, num_chunks * num_v_heads * CHUNK_SIZE * CHUNK_SIZE);
            let mut a_packed_m = alloc(
                DataType::F32,
                num_chunks * num_v_heads * num_blocks * num_col_pairs * BLOCK_SIZE * 2 * BLOCK_SIZE,
            );
            let mut a_inv_m = alloc(DataType::F32, num_chunks * num_v_heads * num_blocks * BLOCK_SIZE * BLOCK_SIZE);
            let mut t_mat_m = alloc(DataType::BF16, num_chunks * num_v_heads * CHUNK_SIZE * CHUNK_SIZE);
            let mut mode_l_state = alloc(DataType::F32, state_size);
            let mut mode_l_out = alloc(DataType::BF16, suffix_len * value_dim);

            let recurrent_prep = <<Metal as Backend>::Kernels as Kernels>::DeltaNetPrefillPrepKernel::new(
                &context,
                DataType::BF16,
                head_k_dim as u32,
                false,
            )
            .expect("recurrent prep");
            let recurrent_prefill = <<Metal as Backend>::Kernels as Kernels>::DeltaNetPrefillKernel::new(
                &context,
                DataType::BF16,
                head_k_dim as u32,
            )
            .expect("recurrent prefill");
            let norm = <<Metal as Backend>::Kernels as Kernels>::DeltaNetNormGateKernel::new(&context, DataType::BF16)
                .expect("norm");
            let chunked_simd = forced_chunked_for_bench(&context, head_k_dim as u32, false);
            let chunked_mxu =
                context.supports_mxu().then(|| forced_chunked_for_bench(&context, head_k_dim as u32, true));

            let warmup = |f: &mut dyn FnMut()| {
                let start = Instant::now();
                while start.elapsed() < Duration::from_millis(500) {
                    f();
                }
            };
            let iterations = if suffix_len >= 1024 {
                30
            } else {
                60
            };

            let mut run_recurrent = || {
                let mut encoder = Encoder::new(context.as_ref()).expect("encoder");
                encoder.encode_fill(&mut rec_state, 0);
                recurrent_prep.encode(
                    &in_proj_array,
                    &a_log_array,
                    &dt_bias_array,
                    &mut rec_q,
                    &mut rec_k,
                    &mut rec_beta,
                    &mut rec_decay,
                    num_v_heads as u32,
                    num_k_heads as u32,
                    key_dim as u32,
                    value_dim as u32,
                    suffix_len as u32,
                    &mut encoder,
                );
                recurrent_prefill.encode(
                    &rec_q,
                    &rec_k,
                    &rec_beta,
                    &rec_decay,
                    &in_proj_array,
                    &mut rec_state,
                    &mut rec_out,
                    num_v_heads as u32,
                    num_k_heads as u32,
                    head_v_dim as u32,
                    key_dim as u32,
                    value_dim as u32,
                    suffix_len as u32,
                    num_dv_groups as u32,
                    &mut encoder,
                );
                norm.encode(
                    &mut rec_out,
                    &in_proj_array,
                    &norm_weight_array,
                    num_v_heads as u32,
                    head_v_dim as u32,
                    value_dim as u32,
                    conv_dim as u32,
                    total_proj_dim as u32,
                    1e-6f32,
                    suffix_len as u32,
                    &mut encoder,
                );
                encoder.end_encoding().submit().wait_until_completed().unwrap();
            };
            warmup(&mut run_recurrent);
            let rec = run_perf_with_warmup("recurrent", 3, iterations, &mut run_recurrent);
            drop(run_recurrent);
            print_bench_row(suffix_len, "recurrent", &rec, 1.0);

            macro_rules! encode_mode_l {
                ($chunked:expr) => {{
                    let mut encoder = Encoder::new(context.as_ref()).expect("encoder");
                    encoder.encode_fill(&mut mode_l_state, 0);
                    $chunked.prep.encode(
                        &in_proj_array,
                        &a_log_array,
                        &dt_bias_array,
                        &mut q_m,
                        &mut k_m,
                        &mut beta_m,
                        &mut log_decay_m,
                        num_v_heads as u32,
                        num_k_heads as u32,
                        key_dim as u32,
                        value_dim as u32,
                        suffix_len as u32,
                        &mut encoder,
                    );
                    $chunked.cumsum.encode(&log_decay_m, &mut g_m, num_v_heads as u32, suffix_len as u32, &mut encoder);
                    $chunked.gram.encode(
                        &q_m,
                        &k_m,
                        &g_m,
                        &mut kk_m,
                        &mut qk_m,
                        num_v_heads as u32,
                        num_k_heads as u32,
                        key_dim as u32,
                        suffix_len as u32,
                        &mut encoder,
                    );
                    $chunked.packed_a_and_dia_inv.encode(
                        &kk_m,
                        &beta_m,
                        &g_m,
                        &mut a_packed_m,
                        &mut a_inv_m,
                        num_v_heads as u32,
                        num_k_heads as u32,
                        suffix_len as u32,
                        &mut encoder,
                    );
                    $chunked.dense_causal_inverse.encode(
                        &a_packed_m,
                        &a_inv_m,
                        &mut t_mat_m,
                        num_v_heads as u32,
                        suffix_len as u32,
                        &mut encoder,
                    );
                    $chunked.output_and_state.encode(
                        &q_m,
                        &k_m,
                        &in_proj_array,
                        &qk_m,
                        &t_mat_m,
                        &g_m,
                        &beta_m,
                        &mut mode_l_state,
                        &mut mode_l_out,
                        num_v_heads as u32,
                        num_k_heads as u32,
                        head_v_dim as u32,
                        key_dim as u32,
                        value_dim as u32,
                        suffix_len as u32,
                        &mut encoder,
                    );
                    norm.encode(
                        &mut mode_l_out,
                        &in_proj_array,
                        &norm_weight_array,
                        num_v_heads as u32,
                        head_v_dim as u32,
                        value_dim as u32,
                        conv_dim as u32,
                        total_proj_dim as u32,
                        1e-6f32,
                        suffix_len as u32,
                        &mut encoder,
                    );
                    encoder.end_encoding().submit().wait_until_completed().unwrap();
                }};
            }

            let mut run_simd = || encode_mode_l!(chunked_simd);
            warmup(&mut run_simd);
            let simd = run_perf_with_warmup("chunked_simd", 3, iterations, &mut run_simd);
            drop(run_simd);
            print_bench_row(suffix_len, "chunked_simd", &simd, rec.median_ms / simd.median_ms);

            if let Some(chunked_mxu) = chunked_mxu.as_ref() {
                let mut run_mxu = || encode_mode_l!(chunked_mxu);
                warmup(&mut run_mxu);
                let mxu = run_perf_with_warmup("chunked_mxu", 3, iterations, &mut run_mxu);
                print_bench_row(suffix_len, "chunked_mxu", &mxu, rec.median_ms / mxu.median_ms);
            }
        }
    }

    fn forced_chunked_for_bench(
        context: &MetalContext,
        head_k_dim: u32,
        use_mxu: bool,
    ) -> MetalDeltaNetChunkedPrefill {
        MetalDeltaNetChunkedPrefill {
            min_t: 0,
            prep: DeltaNetPrefillPrepMetalKernel::new(context, DataType::BF16, head_k_dim, true).expect("chunked prep"),
            cumsum: DeltaNetChunkedCumsumMetalKernel::new(context, CHUNK_SIZE as u32).expect("chunked cumsum"),
            gram: DeltaNetChunkedGramMetalKernel::new(context, head_k_dim, CHUNK_SIZE as u32).expect("chunked gram"),
            packed_a_and_dia_inv: DeltaNetChunkedPackedAAndDiaInvMetalKernel::new(context, CHUNK_SIZE as u32)
                .expect("chunked packed_a_and_dia_inv"),
            dense_causal_inverse: DeltaNetChunkedDenseCausalInverseMetalKernel::new(
                context,
                CHUNK_SIZE as u32,
                VT as u32,
            )
            .expect("chunked dense_causal_inverse"),
            output_and_state: DeltaNetChunkedOutputAndStateMetalKernel::new(
                context,
                DataType::BF16,
                DataType::BF16,
                VT as u32,
                use_mxu,
            )
            .expect("chunked output_and_state"),
        }
    }

    fn print_bench_row(
        suffix_len: usize,
        path: &str,
        result: &test_runner::perf::PerfResult,
        speedup: f64,
    ) {
        eprintln!(
            "BENCH\t{}\t{}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}x",
            suffix_len, path, result.median_ms, result.min_ms, result.max_ms, result.std_dev_ms, speedup
        );
    }
}
