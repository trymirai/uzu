use thiserror::Error;

use crate::{
    backends::common::{
        Allocation, AllocationType, Backend, Context, Encoder, Kernels,
        kernel::qtip_s_exact::{QtipGaussianArguments, QtipSExactKernel},
    },
    config::weight_matrix::{Layout, qtip_gaussian_spec::QtipGaussianSpec},
    data_type::DataType,
    encodable_block::linear::Linear,
    parameters::{ParameterLoaderError, ParameterTree},
};

#[derive(Debug, Error)]
pub enum QtipGaussianLinearError<B: Backend> {
    #[error("Backend error: {0}")]
    BackendError(#[source] B::Error),
    #[error("Parameter loading error: {0}")]
    ParameterError(#[from] ParameterLoaderError<B>),
}

pub struct QtipGaussianLinear<B: Backend> {
    kernel: <B::Kernels as Kernels>::QtipSExactKernel,
    codes: Allocation<B>,
    codebook: Allocation<B>,
    codebook_split: Option<Allocation<B>>,
    state_bits: u32,
    table_mode: u32,
    codebook_scale: f32,
    scales: Allocation<B>,
    gains: Allocation<B>,
    signs: Allocation<B>,
    small_q: Allocation<B>,
    input_dimension: u32,
    output_dimension: u32,
    vector_width: u32,
    transition_bits: u32,
    restart_columns: u32,
}

fn repack_v2_codes(
    physical: &[u8],
    rows: u32,
    bytes_per_row: u32,
    groups: u32,
    transition_bits: u32,
) -> Vec<u8> {
    assert!(matches!(transition_bits, 4 | 6));
    assert_eq!(physical.len(), rows as usize * bytes_per_row as usize);
    let mut repacked = vec![0u8; physical.len()];

    for row in 0..rows as usize {
        let start = row * bytes_per_row as usize;
        let source = &physical[start..start + bytes_per_row as usize];
        let destination = &mut repacked[start..start + bytes_per_row as usize];
        destination[0] = source[1];
        destination[1] = source[0];

        if transition_bits == 4 {
            for (output, input) in destination[2..].iter_mut().zip(&source[2..]) {
                *output = input.rotate_left(4);
            }
            continue;
        }

        let transition_count = groups - 1;
        for block in 0..transition_count.div_ceil(4) as usize {
            let byte = 2 + block * 3;
            let b0 = source[byte];
            let b1 = source[byte + 1];
            let b2 = source[byte + 2];
            let s0 = b0 & 0x3F;
            let s1 = (b0 >> 6) | ((b1 & 0x0F) << 2);
            let s2 = (b1 >> 4) | ((b2 & 0x03) << 4);
            let s3 = if block as u32 * 4 + 3 < transition_count {
                b2 >> 2
            } else {
                0
            };
            destination[byte] = (s0 << 2) | (s1 >> 4);
            destination[byte + 1] = (s1 << 4) | (s2 >> 2);
            destination[byte + 2] = (s2 << 6) | s3;
        }
    }
    repacked
}

impl<B: Backend> QtipGaussianLinear<B> {
    pub fn load(
        context: &B::Context,
        spec: QtipGaussianSpec,
        input_dimension: u32,
        output_dimension: u32,
        parameter_tree: &ParameterTree<B>,
    ) -> Result<Self, QtipGaussianLinearError<B>> {
        assert_eq!(spec.layout, Layout::OutputInput);
        assert!(matches!((spec.vector_width, spec.transition_bits, spec.restart_columns), (2, 4 | 6, 0) | (4, 8, 64)));
        let groups = input_dimension / spec.vector_width;
        let bytes_per_row = match spec.vector_width {
            2 => (16 + (groups - 1) * spec.transition_bits).div_ceil(8),
            4 => input_dimension / 64 * 17,
            _ => unreachable!(),
        };
        let shared = parameter_tree.root().subtree("qtip_shared");
        let dimension = input_dimension.to_string();
        let vector_width = spec.vector_width.to_string();
        let order = match input_dimension {
            5120 => 5,
            6144 => 3,
            17408 => 17,
            _ => panic!("unsupported full-incoherence dimension {input_dimension}"),
        };
        // V4 tables may be the 32768-state (L=15) refit; the state width follows the table
        let state_count = if shared
            .leaf(&format!("codebook_v{vector_width}"))?
            .validate(&[32_768, spec.vector_width], DataType::F32)
            .is_ok()
        {
            32_768u32
        } else {
            65_536u32
        };
        let state_bits = if state_count == 32_768 { 15 } else { 16 };
        let codebook_values = shared
            .leaf(&format!("codebook_v{vector_width}"))?
            .validate(&[state_count, spec.vector_width], DataType::F32)?
            .read_slice::<f32>()?;
        // An L=16 V4 table with T[s ^ 0x8000] == -T[s] is decoded from its first half (128 KiB) with a
        // sign select in the kernels; the stored package stays a plain L=16 package for other runtimes.
        let antipodal = spec.vector_width == 4
            && state_count == 65_536
            && (0..32_768usize).all(|state| {
                (0..4).all(|component| {
                    codebook_values[(state + 32_768) * 4 + component] == -codebook_values[state * 4 + component]
                })
            });
        // Two history sign bits: T[s ^ 0x8000] == -T[s] and T[s ^ 0x4000] negates components 0,1 -> 64 KiB stored
        let two_sign = antipodal
            && (0..16_384usize).all(|state| {
                let flipped = state + 16_384;
                (0..4).all(|component| {
                    let expected = if component < 2 { -codebook_values[state * 4 + component] } else { codebook_values[state * 4 + component] };
                    codebook_values[flipped * 4 + component] == expected
                })
            });
        // four history sign bits: rows 4096.. are rows 0..4095 with component c negated when bit 12+c is set
        let four_sign = spec.vector_width == 4
            && state_count == 65_536
            && (4_096usize..65_536).all(|state| {
                let base = state & 0x0FFF;
                let h = state >> 12;
                (0..4).all(|component| {
                    let expected = if (h >> component) & 1 != 0 { -codebook_values[base * 4 + component] } else { codebook_values[base * 4 + component] };
                    codebook_values[state * 4 + component] == expected
                })
            });
        let table_mode: u32 = if four_sign { 3 } else if two_sign { 2 } else if antipodal { 1 } else { 0 };
        if std::env::var("QTIP_RACE_DEBUG").is_ok() {
            eprintln!("qtip leaf V{} states {} table_mode {} state_bits {}", spec.vector_width, state_count, table_mode, state_bits);
        }
        let codebook_values = if four_sign {
            &codebook_values[..4_096 * 4]
        } else if two_sign {
            &codebook_values[..16_384 * 4]
        } else if antipodal {
            &codebook_values[..32_768 * 4]
        } else {
            &codebook_values[..]
        };
        let codebook_max = codebook_values.iter().fold(0.0f32, |maximum, value| maximum.max(value.abs()));
        let codebook_scale = codebook_max / 127.0;
        let codebook_values_q8 = codebook_values
            .iter()
            .map(|value| (value / codebook_scale).round().clamp(-127.0, 127.0) as i8)
            .collect::<Vec<_>>();
        let mut codebook = context
            .create_allocation(codebook_values_q8.len(), AllocationType::Global)
            .map_err(QtipGaussianLinearError::BackendError)?;
        codebook.copyin(&codebook_values_q8);
        // two-sign packages also carry the 128 KiB half-expanded table (bit 14 applied, bit 15 left to the
        // antipodal kernels), which wins on the K=17408 / 6144-row shapes; it travels in the split slot
        let codebook_wide = if two_sign && !four_sign {
            let mut wide = codebook_values_q8.clone();
            wide.extend((0..16_384usize).flat_map(|state| {
                let row = &codebook_values_q8[state * 4..state * 4 + 4];
                [-row[0], -row[1], row[2], row[3]]
            }));
            let mut allocation = context
                .create_allocation(wide.len(), AllocationType::Global)
                .map_err(QtipGaussianLinearError::BackendError)?;
            allocation.copyin(&wide);
            Some(allocation)
        } else {
            None
        };
        let codebook_split = if let Some(wide) = codebook_wide {
            Some(wide)
        } else if spec.vector_width == 4 && state_count == 65_536 && table_mode == 0 {
            let mut split = vec![0i8; codebook_values_q8.len()];
            for state in 0..65_536usize {
                split[2 * state] = codebook_values_q8[4 * state];
                split[2 * state + 1] = codebook_values_q8[4 * state + 1];
                split[131_072 + 2 * state] = codebook_values_q8[4 * state + 2];
                split[131_072 + 2 * state + 1] = codebook_values_q8[4 * state + 3];
            }
            let mut allocation = context
                .create_allocation(split.len(), AllocationType::Global)
                .map_err(QtipGaussianLinearError::BackendError)?;
            allocation.copyin(&split);
            Some(allocation)
        } else {
            None
        };

        let codes = if spec.vector_width == 2 {
            let physical = parameter_tree
                .leaf("codes")?
                .validate(&[output_dimension, bytes_per_row], DataType::U8)?
                .read_slice::<u8>()?;
            let repacked = repack_v2_codes(
                &physical,
                output_dimension,
                bytes_per_row,
                groups,
                spec.transition_bits,
            );
            let mut allocation = context
                .create_allocation(repacked.len(), AllocationType::Global)
                .map_err(QtipGaussianLinearError::BackendError)?;
            allocation.copyin(&repacked);
            allocation
        } else {
            parameter_tree
                .leaf("codes")?
                .validate(&[output_dimension, bytes_per_row], DataType::U8)?
                .read_allocation()?
        };
        Ok(Self {
            kernel: <B::Kernels as Kernels>::QtipSExactKernel::new(context)
                .map_err(QtipGaussianLinearError::BackendError)?,
            codes,
            codebook,
            codebook_split,
            state_bits,
            table_mode,
            codebook_scale,
            scales: parameter_tree.leaf("scales")?.validate(&[output_dimension], DataType::F16)?.read_allocation()?,
            gains: parameter_tree.leaf("gains")?.validate(&[output_dimension], DataType::BF16)?.read_allocation()?,
            signs: shared
                .leaf(&format!("signs_{dimension}"))?
                .validate(&[input_dimension], DataType::F32)?
                .read_allocation()?,
            small_q: shared
                .leaf(&format!("q_{dimension}"))?
                .validate(&[order, order], DataType::F32)?
                .read_allocation()?,
            input_dimension,
            output_dimension,
            vector_width: spec.vector_width,
            transition_bits: spec.transition_bits,
            restart_columns: spec.restart_columns,
        })
    }
}

impl<B: Backend> Linear<B> for QtipGaussianLinear<B> {
    fn encode(
        &self,
        input: Allocation<B>,
        batch_dim: u32,
        encoder: &mut Encoder<B>,
    ) -> Result<Allocation<B>, B::Error> {
        encoder.push_debug_group("qtip gaussian linear");
        let output = self.kernel.encode_qtip_gaussian(
            QtipGaussianArguments {
                input: &input,
                codes: &self.codes,
                codebook: &self.codebook,
                codebook_split: self.codebook_split.as_ref(),
                state_bits: self.state_bits,
                table_mode: self.table_mode,
                codebook_scale: self.codebook_scale,
                scales: &self.scales,
                gains: &self.gains,
                signs: &self.signs,
                small_q: &self.small_q,
                batch: batch_dim,
                rows: self.output_dimension,
                columns: self.input_dimension,
                vector_width: self.vector_width,
                transition_bits: self.transition_bits,
                restart_columns: self.restart_columns,
            },
            encoder,
        )?;
        encoder.pop_debug_group();
        Ok(output)
    }
}
