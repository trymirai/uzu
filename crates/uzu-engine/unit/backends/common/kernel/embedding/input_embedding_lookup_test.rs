use std::{fmt::Display, io::Write};

use futures::executor::block_on;
use half::{bf16, f16};
use num_traits::Float;
use rstest::rstest;
use uzu_engine_macros::uzu_test;

use crate::{
    backends::{
        common::{
            Backend, Context, Kernels,
            gpu_types::{
                d4s4,
                embedding::EmbeddingTableKind::{self, D4S4, Dense, Quantized},
                quantization::QuantizationMode::{self, I8, U4, U8},
                quantization_method::QuantizationMethod::{self, ScaleBias, ScaleSymmetric, ScaleZeroPoint},
            },
            kernel::InputEmbeddingLookupKernel,
        },
        cpu::Cpu,
    },
    data_type::{DataType, DataTypeRepr},
    encodable_block::embedding::Embedding,
    safetensors::ParameterLoader,
    tests::{
        assert::assert_eq_float,
        helpers::{
            buffer_readback, buffer_to_vec, create_buffer, create_buffer_with_data, for_each_backend,
            for_each_non_cpu_backend, submit_command_buffer,
        },
    },
};

#[uzu_test]
fn embedding_prefix_preserves_full_lookup() {
    for_each_backend!(|B| {
        for tied in [true, false] {
            let context = block_on(<B as Backend>::Context::new()).unwrap();
            let weights = floats::<bf16>(VOCAB_SIZE * MODEL_DIM, |index| (index as f32 * 0.37).sin() * 0.1);
            let bytes: &[u8] = bytemuck::cast_slice(&weights);
            let names: &[&str] = if tied {
                &["embedding"]
            } else {
                &["input_embedding", "output_embedding"]
            };
            let mut header = serde_json::json!({"__metadata__": {}});
            for (index, name) in names.iter().enumerate() {
                let layout = if *name == "output_embedding" {
                    "output_input"
                } else {
                    "input_output"
                };
                header["__metadata__"][format!("{name}.spec")] =
                    serde_json::json!(serde_json::json!({"type":"FullPrecisionSpec", "layout":layout}).to_string());
                header[format!("{name}.weights")] = serde_json::json!({
                    "dtype":"BF16", "shape":[VOCAB_SIZE, MODEL_DIM],
                    "data_offsets":[index * bytes.len(), (index + 1) * bytes.len()]
                });
            }
            let header = serde_json::to_vec(&header).unwrap();
            let path = std::env::temp_dir().join(format!("uzu-embedding-view-{}", std::process::id()));
            let mut file = std::fs::OpenOptions::new().read(true).write(true).create_new(true).open(&path).unwrap();
            file.write_all(&(header.len() as u64).to_le_bytes()).unwrap();
            file.write_all(&header).unwrap();
            for _ in names {
                file.write_all(bytes).unwrap();
            }
            drop(file);
            let loader = block_on(ParameterLoader::<B>::new(&path, &context)).unwrap();
            std::fs::remove_file(path).unwrap();
            let config = serde_json::from_value(serde_json::json!({
                "type":if tied { "TiedEmbeddingConfig" } else { "UntiedEmbeddingConfig" },
                "input_scale":null, "logit_scale":0.75, "logit_soft_cap":3.0
            }))
            .unwrap();
            let (target, _) =
                block_on(Embedding::new(VOCAB_SIZE, MODEL_DIM, DataType::BF16, &config, &loader.tree(), &context))
                    .unwrap();
            assert!(target.prefix(0).is_err());
            assert!(target.prefix(VOCAB_SIZE + 1).is_err());
            assert_eq!(target.prefix(VOCAB_SIZE).unwrap().vocab_size(), VOCAB_SIZE);
            let draft = target.prefix(3).unwrap();
            assert!(draft.prefix(0).is_err());
            assert!(draft.prefix(4).is_err());
            let nested = draft.prefix(2).unwrap();
            assert_eq!(nested.vocab_size(), 2);
            assert_eq!(nested.lookup_vocab_size(), VOCAB_SIZE);
            assert_eq!(target.vocab_size(), VOCAB_SIZE);
            assert_eq!(draft.vocab_size(), 3);
            assert_eq!(draft.lookup_vocab_size(), VOCAB_SIZE);
            let ids = create_buffer_with_data::<B, u32>(&context, &[3, 0]);
            let mut commands = context.create_command_buffer(None, None).unwrap();
            let full_lookup = target.encode_lookup(&ids, 2, &mut commands).unwrap();
            let draft_lookup = draft.encode_lookup(&ids, 2, &mut commands).unwrap();
            let full = target
                .encode_readout(
                    &full_lookup,
                    2,
                    target.vocab_size(),
                    None::<&<B as Backend>::ScratchBuffer>,
                    true,
                    &mut commands,
                )
                .unwrap();
            let prefix = draft
                .encode_readout(
                    &draft_lookup,
                    2,
                    draft.vocab_size(),
                    None::<&<B as Backend>::ScratchBuffer>,
                    true,
                    &mut commands,
                )
                .unwrap();
            let gather_ids = [3, 0, 2, 1, 1, 3, 0, 2];
            let indices = create_buffer_with_data::<B, u32>(&context, &gather_ids);
            let gathered = draft.encode_readout(&draft_lookup, 2, 4, Some(&indices), true, &mut commands).unwrap();
            let full_after_draft = target
                .encode_readout(
                    &full_lookup,
                    2,
                    target.vocab_size(),
                    None::<&<B as Backend>::ScratchBuffer>,
                    true,
                    &mut commands,
                )
                .unwrap();
            submit_command_buffer(commands);
            let read = |buffer: &<B as Backend>::ScratchBuffer| {
                buffer_to_vec::<B, bf16>(&context, &buffer_readback::<B>(&context, buffer))
                    .into_iter()
                    .map(f32::from)
                    .collect::<Vec<_>>()
            };
            assert_eq_float(&read(&full_lookup), &read(&draft_lookup), 0.0, "full lookup through prefix");
            let full = read(&full);
            let expected: Vec<_> =
                full.chunks_exact(VOCAB_SIZE as usize).flat_map(|row| row[..3].iter().copied()).collect();
            assert_eq_float(&expected, &read(&prefix), 0.001, "prefix readout");
            let full_ref = &full;
            let expected: Vec<_> = gather_ids
                .chunks_exact(4)
                .enumerate()
                .flat_map(|(row, ids)| ids.iter().map(move |id| full_ref[row * VOCAB_SIZE as usize + *id as usize]))
                .collect();
            assert_eq_float(&expected, &read(&gathered), 0.001, "gathered full-vocabulary IDs through prefix");
            assert_eq_float(&full, &read(&full_after_draft), 0.0, "target readout after prefix use");
        }
    });
}

#[uzu_test]
fn embedding_prefix_quantized_input_rht_readout() {
    const VOCAB: u32 = 64;
    const DIM: u32 = 512;
    const GROUP: u32 = 32;

    for_each_backend!(|B| {
        let context = block_on(<B as Backend>::Context::new()).unwrap();
        let input_weights = floats::<bf16>(VOCAB * DIM, |index| (index as f32 * 0.37).sin() * 0.1);
        let codes = pattern(VOCAB * DIM / 2, 37);
        let scales = floats::<bf16>(VOCAB * (DIM / GROUP), |index| 0.005 + (index % 5) as f32 * 0.001);
        let biases = floats::<bf16>(VOCAB * (DIM / GROUP), |index| -0.04 + (index % 3) as f32 * 0.002);
        let signs: Vec<i32> = (0..DIM)
            .map(|index| {
                if index % 3 == 0 {
                    -1
                } else {
                    1
                }
            })
            .collect();
        let tensors: &[(&str, &str, &[u32], &[u8])] = &[
            ("input_embedding.weights", "BF16", &[VOCAB, DIM], bytemuck::cast_slice(&input_weights)),
            ("output_embedding.quantized.weights", "U8", &[VOCAB, DIM / 2], &codes),
            ("output_embedding.quantized.scales", "BF16", &[DIM / GROUP, VOCAB], bytemuck::cast_slice(&scales)),
            ("output_embedding.quantized.biases", "BF16", &[DIM / GROUP, VOCAB], bytemuck::cast_slice(&biases)),
            ("output_embedding.incoherence_signs.input_signs", "I32", &[DIM], bytemuck::cast_slice(&signs)),
        ];
        let mut header = serde_json::json!({"__metadata__": {
            "input_embedding.spec": serde_json::json!({"type":"FullPrecisionSpec", "layout":"input_output"}).to_string(),
            "output_embedding.spec": serde_json::json!({
                "type":"HybridSpec", "adapter_spec":null, "incoherence_block_size":32,
                "incoherence_processing_mode":"input", "quantization_spec":{
                    "type":"MLXSpec", "bits":4, "group_size":GROUP,
                    "layout":"output_input"
                }
            }).to_string()
        }});
        let mut offset = 0;
        for (name, dtype, shape, bytes) in tensors {
            header[*name] = serde_json::json!({
                "dtype":dtype, "shape":shape, "data_offsets":[offset, offset + bytes.len()]
            });
            offset += bytes.len();
        }
        let header = serde_json::to_vec(&header).unwrap();
        let path = std::env::temp_dir().join(format!("uzu-embedding-prefix-rht-{}", std::process::id()));
        let mut file = std::fs::OpenOptions::new().read(true).write(true).create_new(true).open(&path).unwrap();
        file.write_all(&(header.len() as u64).to_le_bytes()).unwrap();
        file.write_all(&header).unwrap();
        for (_, _, _, bytes) in tensors {
            file.write_all(bytes).unwrap();
        }
        drop(file);
        let loader = block_on(ParameterLoader::<B>::new(&path, &context)).unwrap();
        std::fs::remove_file(path).unwrap();
        let config = serde_json::from_value(serde_json::json!({
            "type":"UntiedEmbeddingConfig", "input_scale":null, "logit_scale":null, "logit_soft_cap":null
        }))
        .unwrap();
        // Metal weight preparation waits synchronously, as it does under the engine's Tokio runtime.
        let (target, _) = tokio::runtime::Builder::new_current_thread()
            .build()
            .unwrap()
            .block_on(Embedding::new(VOCAB, DIM, DataType::BF16, &config, &loader.tree(), &context))
            .unwrap();
        loader.assert_all_tensors_read().unwrap();
        for (batch, prefix_width) in [1, 4, 9]
            .into_iter()
            .flat_map(|batch| [1, 15, 16, 17, 19, 31, 32, 33].into_iter().map(move |width| (batch, width)))
        {
            let draft = target.prefix(prefix_width).unwrap();
            let input = floats::<bf16>(batch * DIM, |index| (index as f32 * 0.17).cos() * 0.1);
            let input = create_buffer_with_data::<B, bf16>(&context, &input);
            let mut commands = context.create_command_buffer(None, None).unwrap();
            let full = target
                .encode_readout(&input, batch, VOCAB, None::<&<B as Backend>::ScratchBuffer>, false, &mut commands)
                .unwrap();
            let prefix = draft
                .encode_readout(
                    &input,
                    batch,
                    prefix_width,
                    None::<&<B as Backend>::ScratchBuffer>,
                    false,
                    &mut commands,
                )
                .unwrap();
            let full_after = target
                .encode_readout(&input, batch, VOCAB, None::<&<B as Backend>::ScratchBuffer>, false, &mut commands)
                .unwrap();
            submit_command_buffer(commands);
            let read = |buffer: &<B as Backend>::ScratchBuffer| {
                buffer_to_vec::<B, bf16>(&context, &buffer_readback::<B>(&context, buffer))
                    .into_iter()
                    .map(f32::from)
                    .collect::<Vec<_>>()
            };
            let full = read(&full);
            assert!(full.iter().any(|value| value.abs() > 0.001), "quantized readout is empty");
            let expected: Vec<_> = full
                .chunks_exact(VOCAB as usize)
                .flat_map(|row| row[..prefix_width as usize].iter().copied())
                .collect();
            assert_eq_float(
                &expected,
                &read(&prefix),
                0.003,
                &format!(
                    "quantized input-RHT prefix readout {} batch {batch} width {prefix_width}",
                    std::any::type_name::<B>()
                ),
            );
            assert_eq_float(&full, &read(&full_after), 0.0, "quantized target readout after prefix use");
        }
    });
}

const VOCAB_SIZE: u32 = 4;
const MODEL_DIM: u32 = 128;
const TOKEN_IDS: [u32; 3] = [0, 2, VOCAB_SIZE];

type Quantization = (QuantizationMode, QuantizationMethod, u32);

fn pattern(
    len: u32,
    seed: u32,
) -> Vec<u8> {
    (0..len).map(|index| (index * seed + 9) as u8).collect()
}

fn floats<T: DataTypeRepr + Float>(
    len: u32,
    value: impl Fn(u32) -> f32,
) -> Vec<T> {
    (0..len).map(|index| T::from(value(index)).unwrap()).collect()
}

fn lookup<B: Backend, T: DataTypeRepr + Float>(
    table_kind: EmbeddingTableKind,
    quantization: Option<Quantization>,
    use_hadamard: bool,
) -> Vec<T> {
    let context = block_on(<B as Backend>::Context::new()).unwrap();
    let context = context.as_ref();
    let bytes = |data: &[u8]| create_buffer_with_data::<B, u8>(context, data);
    let (mut scales, mut zero_points, mut biases) = (None, None, None);
    let (mut ladder_indices, mut ladder, mut codebook) = (None, None, None);
    let values = match table_kind {
        Dense => {
            bytes(bytemuck::cast_slice(&floats::<T>(VOCAB_SIZE * MODEL_DIM, |index| (index as f32 * 0.37).sin() * 3.0)))
        },
        Quantized => {
            let (mode, method, group_size) = quantization.expect("quantization settings");
            let groups = MODEL_DIM.div_ceil(group_size);
            let group_values =
                |value: fn(u32) -> f32| create_buffer_with_data::<B, T>(context, &floats(VOCAB_SIZE * groups, value));
            scales = Some(group_values(|index| 0.02 + (index % 5) as f32 * 0.01));
            if method == ScaleBias {
                biases = Some(group_values(|index| -0.1 + index as f32 * 0.03));
            } else if method == ScaleZeroPoint {
                zero_points = Some(bytes(&pattern(VOCAB_SIZE * groups.div_ceil(mode.packing_divisor()), 53)));
            }
            bytes(&pattern(VOCAB_SIZE * MODEL_DIM / mode.packing_divisor(), 37))
        },
        D4S4 => {
            let row_scales = floats::<T>(VOCAB_SIZE, |row| 0.02 + row as f32 * 0.01);
            let steps: Vec<f16> =
                (0..d4s4::LADDER_SIZE).map(|index| f16::from_f32(2f32.powf(index as f32 / 2.0 - 5.5))).collect();
            let points: Vec<i8> =
                (0..d4s4::CODEBOOK_SIZE * d4s4::VALUES_PER_CODE).map(|index| (index % 9) as i8 - 4).collect();
            scales = Some(create_buffer_with_data::<B, T>(context, &row_scales));
            ladder_indices = Some(bytes(&pattern(VOCAB_SIZE * MODEL_DIM / d4s4::COLUMNS_PER_LADDER_INDEX_BYTE, 5)));
            ladder = Some(create_buffer_with_data::<B, f16>(context, &steps));
            codebook = Some(create_buffer_with_data::<B, i8>(context, &points));
            bytes(&pattern(VOCAB_SIZE * MODEL_DIM / d4s4::VALUES_PER_CODE, 37))
        },
    };
    let hadamard_factors = if use_hadamard {
        let signs: Vec<i32> = (0..MODEL_DIM)
            .map(|index| {
                if index % 3 == 0 {
                    -1
                } else {
                    1
                }
            })
            .collect();
        Some(create_buffer_with_data::<B, i32>(context, &signs))
    } else {
        None
    };

    let (mode, method, group_size) = quantization
        .map_or((None, None, None), |(mode, method, group_size)| (Some(mode), Some(method), Some(group_size)));
    let kernel = <<B as Backend>::Kernels as Kernels>::InputEmbeddingLookupKernel::new(
        context,
        T::data_type(),
        table_kind,
        group_size,
        mode,
        method,
        use_hadamard,
    )
    .unwrap();
    let token_ids = create_buffer_with_data::<B, u32>(context, &TOKEN_IDS);
    let mut output = create_buffer::<B, T>(context, TOKEN_IDS.len() * MODEL_DIM as usize);
    let mut command_buffer = context.create_command_buffer(None, None).unwrap();
    kernel.encode(
        &token_ids,
        &values,
        scales.as_ref(),
        zero_points.as_ref(),
        biases.as_ref(),
        hadamard_factors.as_ref(),
        ladder_indices.as_ref(),
        ladder.as_ref(),
        codebook.as_ref(),
        &mut output,
        TOKEN_IDS.len() as u32,
        VOCAB_SIZE,
        MODEL_DIM,
        1.5,
        &mut command_buffer,
    );
    submit_command_buffer(command_buffer);
    buffer_to_vec::<B, T>(context, &output)
}

fn check<T: DataTypeRepr + Float + Display>(
    table_kind: EmbeddingTableKind,
    quantization: Option<Quantization>,
    use_hadamard: bool,
) {
    let expected = lookup::<Cpu, T>(table_kind, quantization, use_hadamard);
    let in_vocab = 2 * MODEL_DIM as usize;
    assert!(expected[..in_vocab].iter().any(|value| *value != T::zero()), "in-vocab rows are empty");
    assert!(expected[in_vocab..].iter().all(|value| *value == T::zero()), "out-of-vocab row is not zero");
    for_each_non_cpu_backend!(|B| {
        let actual = lookup::<B, T>(table_kind, quantization, use_hadamard);
        assert_eq_float(&expected, &actual, 0.02, std::any::type_name::<B>());
    });
}

#[rstest]
#[test_attr(uzu_test)]
#[case::dense(Dense, None, false)]
#[case::u4_bias(Quantized, Some((U4, ScaleBias, 32)), false)]
#[case::u4_zero_point_odd_group(Quantized, Some((U4, ScaleZeroPoint, 48)), false)]
#[case::u4_symmetric(Quantized, Some((U4, ScaleSymmetric, 32)), false)]
#[case::u8_bias(Quantized, Some((U8, ScaleBias, 32)), false)]
#[case::u8_zero_point(Quantized, Some((U8, ScaleZeroPoint, 32)), false)]
#[case::u8_symmetric(Quantized, Some((U8, ScaleSymmetric, 32)), false)]
#[case::i8_symmetric(Quantized, Some((I8, ScaleSymmetric, 32)), false)]
#[case::u4_bias_hadamard(Quantized, Some((U4, ScaleBias, 32)), true)]
#[case::d4s4(D4S4, None, true)]
fn input_embedding_lookup_bf16(
    #[case] table_kind: EmbeddingTableKind,
    #[case] quantization: Option<Quantization>,
    #[case] use_hadamard: bool,
) {
    check::<bf16>(table_kind, quantization, use_hadamard);
}

#[rstest]
#[test_attr(uzu_test)]
#[case::dense(Dense, None, false)]
#[case::u4_bias(Quantized, Some((U4, ScaleBias, 32)), false)]
fn input_embedding_lookup_f32(
    #[case] table_kind: EmbeddingTableKind,
    #[case] quantization: Option<Quantization>,
    #[case] use_hadamard: bool,
) {
    check::<f32>(table_kind, quantization, use_hadamard);
}
