#![cfg(metal_backend)]

use std::time::Instant;

use backend_uzu::{
    Array, ArrayContextExt, DataType,
    backends::{
        common::{
            Backend, Context, Encoder, Kernels,
            gpu_types::ActivationType,
            kernel::{
                ActivationKernel, AudioAddKernel, AudioCausalConv1dGroupedKernel,
                AudioCausalConv1dGroupedResidualKernel, AudioCausalConv1dKernel,
                AudioCausalConvTranspose1dCausalPadKernel, AudioConv1dKernel, AudioHalfSnakeKernel, AudioNormNcsKernel,
                AudioQuantizerDecodeKernel,
            },
        },
        metal::Metal,
    },
};

type Ctx = <Metal as Backend>::Context;

const WARMUP: usize = 5;
const ITERATIONS: usize = 20;

fn measure<F: FnMut()>(
    label: &str,
    mut f: F,
) -> f64 {
    for _ in 0..WARMUP {
        f();
    }
    let start = Instant::now();
    for _ in 0..ITERATIONS {
        f();
    }
    let elapsed = start.elapsed();
    let mean_ms = elapsed.as_secs_f64() * 1000.0 / ITERATIONS as f64;
    println!("  {label:<65} {mean_ms:>8.3} ms");
    mean_ms
}

fn make_lengths(
    context: &Ctx,
    batch_size: usize,
    value: i32,
) -> Array<Metal> {
    context.create_array_from(&[batch_size], &vec![value; batch_size], "lengths")
}

// ==========================================================================
// FishAudio s1-mini vocoder configuration (from model config JSON)
//
//   input_dim          = 1024
//   decoder_dim        = 1536
//   downsample_factor  = [2, 2]  (reversed for upsampler -> strides [2, 2])
//   decoder_rates      = [8, 8, 4, 2]
//   n_codebooks        = 9  (+ 1 semantic = 10 total)
//   codebook_dim       = 8
//   codebook_size      = 1024
//   semantic_codebook_size = 4096
//   samplerate         = 44100
//   upsample_factor    = 2*2 * 8*8*4*2 = 2048
//
// Channel progression (standard DAC with decoder_dim=1536):
//   Upsampler:   1024 -> 512 (s=2) -> 256 (s=2)
//   first_conv:  256  -> 1536  (k=7)
//   Decoder:     1536 -> 768 (s=8) -> 384 (s=8) -> 192 (s=4) -> 96 (s=2)
//   final_conv:  96   -> 1    (k=7)
//
// Residual units per decoder block: 3, with dilations [1, 3, 9], k=7
// ConvNeXt depthwise: k=7, per-channel
// Transpose conv kernel_size = stride * 2 (fast_two_tap path)
//
// For 10 seconds at 44100 Hz:
//   441000 samples / 2048 upsample_factor = ~215 semantic frames
//   We use 215 frames as the starting point.
// ==========================================================================

const INPUT_DIM: usize = 1024;
const DECODER_DIM: usize = 1536;
const CODEBOOK_DIM: usize = 8;
const TOTAL_CODEBOOKS: usize = 10;
const RESIDUAL_QUANTIZERS: usize = 9;
const SEMANTIC_CARDINALITY: usize = 4096;
const RESIDUAL_CARDINALITY: usize = 1024;

// Upsample block channel progression: input_dim -> input_dim/2 -> input_dim/4
const UP_CHANNELS: [usize; 3] = [1024, 512, 256];
const UP_STRIDES: [usize; 2] = [2, 2]; // downsample_factor reversed

// Decoder block channel progression: decoder_dim -> decoder_dim/2 -> /4 -> /8 -> /16
const DEC_CHANNELS: [usize; 5] = [1536, 768, 384, 192, 96];
const DEC_STRIDES: [usize; 4] = [8, 8, 4, 2]; // decoder_rates

const SEMANTIC_FRAMES_10S: usize = 215; // ~10s at 44100 Hz with upsample_factor=2048

#[test]
#[ignore] // Heavy benchmark (~240s); run explicitly with --ignored
fn audio_kernel_perf() {
    let context = <Metal as Backend>::Context::new().unwrap();
    let dt = DataType::F16;
    let batch_size: usize = 1;

    let mut results: Vec<(String, f64)> = Vec::new();

    println!();
    println!("  FishAudio s1-mini vocoder kernel benchmark (10s audio, 44.1kHz, F16)");
    println!("  =====================================================================");
    println!();
    println!("  {:<65} {:>8}", "Kernel", "Mean (ms)");
    println!("  {:-<65} {:-<8}", "", "");

    // -----------------------------------------------------------------------
    // 1. AudioQuantizerDecodeKernel
    //    tokens: [B, K, T], output: [B, T, input_dim]
    // -----------------------------------------------------------------------
    {
        let frames = SEMANTIC_FRAMES_10S;

        let kernel = <<Metal as Backend>::Kernels as Kernels>::AudioQuantizerDecodeKernel::new(&context, dt).unwrap();

        let tokens =
            context.create_array_zeros(&[batch_size * TOTAL_CODEBOOKS * frames], DataType::U32, "perf_quant_tokens");
        let lengths = make_lengths(&context, batch_size, frames as i32);
        let semantic_codebook = context.create_array_zeros(&[SEMANTIC_CARDINALITY * CODEBOOK_DIM], dt, "perf_sem_cb");
        let semantic_out_proj = context.create_array_zeros(&[INPUT_DIM * CODEBOOK_DIM], dt, "perf_sem_proj");
        let semantic_out_bias = context.create_array_zeros(&[INPUT_DIM], dt, "perf_sem_bias");
        let residual_codebooks = context.create_array_zeros(
            &[RESIDUAL_QUANTIZERS * RESIDUAL_CARDINALITY * CODEBOOK_DIM],
            dt,
            "perf_res_cbs",
        );
        let residual_out_proj =
            context.create_array_zeros(&[RESIDUAL_QUANTIZERS * INPUT_DIM * CODEBOOK_DIM], dt, "perf_res_proj");
        let residual_out_bias = context.create_array_zeros(&[RESIDUAL_QUANTIZERS * INPUT_DIM], dt, "perf_res_bias");
        let mut output =
            context.create_array_zeros(&[batch_size * frames * INPUT_DIM], dt, "perf_quant_out").into_allocation();

        let label = format!(
            "QuantizerDecode [B={}, T={}, K={}, dim={}, cdim={}]",
            batch_size, frames, TOTAL_CODEBOOKS, INPUT_DIM, CODEBOOK_DIM
        );
        let ms = measure(&label, || {
            let mut encoder = Encoder::<Metal>::new(context.as_ref()).unwrap();
            {
                let tok = tokens.allocation();
                let len = lengths.allocation();
                let scb = semantic_codebook.allocation();
                let sp = semantic_out_proj.allocation();
                let sb = semantic_out_bias.allocation();
                let rcb = residual_codebooks.allocation();
                let rp = residual_out_proj.allocation();
                let rb = residual_out_bias.allocation();
                let out = &mut output;
                kernel.encode(
                    tok,
                    len,
                    scb,
                    sp,
                    sb,
                    rcb,
                    rp,
                    rb,
                    out,
                    batch_size as i32,
                    TOTAL_CODEBOOKS as i32,
                    frames as i32,
                    INPUT_DIM as i32,
                    CODEBOOK_DIM as i32,
                    RESIDUAL_QUANTIZERS as i32,
                    SEMANTIC_CARDINALITY as i32,
                    RESIDUAL_CARDINALITY as i32,
                    &mut encoder,
                );
            }
            encoder.end_encoding().submit().wait_until_completed().unwrap();
        });
        results.push(("QuantizerDecode".into(), ms));
    }

    // -----------------------------------------------------------------------
    // 2. ConvTranspose1dCausalPad — all 6 stages where it appears
    //    Upsample: 1024->512 s=2, 512->256 s=2
    //    Decoder:  1536->768 s=8, 768->384 s=8, 384->192 s=4, 192->96 s=2
    // -----------------------------------------------------------------------
    {
        let kernel =
            <<Metal as Backend>::Kernels as Kernels>::AudioCausalConvTranspose1dCausalPadKernel::new(&context, dt)
                .unwrap();

        // Upsample blocks (NSC input for first, NCS for rest)
        let up_configs: [(usize, usize, usize, i32, &str); 2] = [
            (UP_CHANNELS[0], UP_CHANNELS[1], UP_STRIDES[0], 1, "up0 NSC"), // 1024->512, s=2
            (UP_CHANNELS[1], UP_CHANNELS[2], UP_STRIDES[1], 0, "up1 NCS"), // 512->256, s=2
        ];
        let mut frames = SEMANTIC_FRAMES_10S;
        for (cin, cout, stride, layout, tag) in up_configs {
            let seq_in = frames;
            let seq_out = seq_in * stride;
            let ksize = stride * 2;
            let input = context.create_array_zeros(&[batch_size * cin * seq_in], dt, "tconv_in");
            let weight = context.create_array_zeros(&[cin * cout * ksize], dt, "tconv_w");
            let bias = context.create_array_zeros(&[cout], dt, "tconv_b");
            let mut output =
                context.create_array_zeros(&[batch_size * cout * seq_out], dt, "tconv_out").into_allocation();
            let lengths = make_lengths(&context, batch_size, seq_out as i32);

            let label = format!("TransConv {tag} [{cin}->{cout}, s={stride}, k={ksize}, {seq_in}->{seq_out}]");
            let ms = measure(&label, || {
                let mut encoder = Encoder::<Metal>::new(context.as_ref()).unwrap();
                {
                    let i = input.allocation();
                    let w = weight.allocation();
                    let b = bias.allocation();
                    let o = &mut output;
                    let l = lengths.allocation();
                    kernel.encode(
                        i,
                        w,
                        b,
                        o,
                        l,
                        cin as i32,
                        cout as i32,
                        seq_in as i32,
                        seq_out as i32,
                        ksize as i32,
                        stride as i32,
                        1_i32,
                        layout,
                        batch_size as i32,
                        &mut encoder,
                    );
                }
                encoder.end_encoding().submit().wait_until_completed().unwrap();
            });
            results.push((format!("TransConv {tag}"), ms));
            frames = seq_out;
        }

        // Decoder blocks (all NCS input)
        frames = SEMANTIC_FRAMES_10S * UP_STRIDES[0] * UP_STRIDES[1]; // 860
        for block_idx in 0..4 {
            let cin = DEC_CHANNELS[block_idx];
            let cout = DEC_CHANNELS[block_idx + 1];
            let stride = DEC_STRIDES[block_idx];
            let seq_in = frames;
            let seq_out = seq_in * stride;
            let ksize = stride * 2;
            let input = context.create_array_zeros(&[batch_size * cin * seq_in], dt, "tconv_in");
            let weight = context.create_array_zeros(&[cin * cout * ksize], dt, "tconv_w");
            let bias = context.create_array_zeros(&[cout], dt, "tconv_b");
            let mut output =
                context.create_array_zeros(&[batch_size * cout * seq_out], dt, "tconv_out").into_allocation();
            let lengths = make_lengths(&context, batch_size, seq_out as i32);

            let label = format!("TransConv dec{block_idx} [{cin}->{cout}, s={stride}, k={ksize}, {seq_in}->{seq_out}]");
            let ms = measure(&label, || {
                let mut encoder = Encoder::<Metal>::new(context.as_ref()).unwrap();
                {
                    let i = input.allocation();
                    let w = weight.allocation();
                    let b = bias.allocation();
                    let o = &mut output;
                    let l = lengths.allocation();
                    kernel.encode(
                        i,
                        w,
                        b,
                        o,
                        l,
                        cin as i32,
                        cout as i32,
                        seq_in as i32,
                        seq_out as i32,
                        ksize as i32,
                        stride as i32,
                        1_i32,
                        0_i32,
                        batch_size as i32,
                        &mut encoder,
                    );
                }
                encoder.end_encoding().submit().wait_until_completed().unwrap();
            });
            results.push((format!("TransConv dec{block_idx}"), ms));
            frames = seq_out;
        }
    }

    // -----------------------------------------------------------------------
    // 3. CausalConv1d (groups=1) — first_conv, final_conv, and residual conv1
    //    first_conv: 256->1536, k=7
    //    residual conv1: same-channel, k=7, dilations [1,3,9]
    //    final_conv: 96->1, k=7
    // -----------------------------------------------------------------------
    {
        let kernel = <<Metal as Backend>::Kernels as Kernels>::AudioCausalConv1dKernel::new(&context, dt).unwrap();

        // first_conv after upsampler: 256->1536, k=7
        let fc_seq = SEMANTIC_FRAMES_10S * UP_STRIDES[0] * UP_STRIDES[1]; // 860
        {
            let cin = UP_CHANNELS[2]; // 256
            let cout = DECODER_DIM; // 1536
            let ksize = 7;
            let input = context.create_array_zeros(&[batch_size * cin * fc_seq], dt, "fc_in");
            let weight = context.create_array_zeros(&[cout * cin * ksize], dt, "fc_w");
            let bias = context.create_array_zeros(&[cout], dt, "fc_b");
            let mut output = context.create_array_zeros(&[batch_size * cout * fc_seq], dt, "fc_out").into_allocation();
            let lengths = make_lengths(&context, batch_size, fc_seq as i32);

            let label = format!("CausalConv1d first_conv [{cin}->{cout}, k={ksize}, T={fc_seq}]");
            let ms = measure(&label, || {
                let mut encoder = Encoder::<Metal>::new(context.as_ref()).unwrap();
                {
                    let i = input.allocation();
                    let w = weight.allocation();
                    let b = bias.allocation();
                    let o = &mut output;
                    let l = lengths.allocation();
                    kernel.encode(
                        i,
                        w,
                        b,
                        o,
                        l,
                        cin as i32,
                        cout as i32,
                        fc_seq as i32,
                        ksize as i32,
                        1_i32,
                        0_i32,
                        batch_size as i32,
                        &mut encoder,
                    );
                }
                encoder.end_encoding().submit().wait_until_completed().unwrap();
            });
            results.push(("CausalConv1d first_conv".into(), ms));
        }

        // Residual conv1 at each decoder block resolution and dilation
        let mut dec_frames = fc_seq;
        for block_idx in 0..4 {
            dec_frames *= DEC_STRIDES[block_idx];
            let ch = DEC_CHANNELS[block_idx + 1];
            let ksize = 7;

            for &dilation in &[1, 3, 9] {
                let label =
                    format!("CausalConv1d res dec{block_idx} [{ch}->{ch}, k={ksize}, d={dilation}, T={dec_frames}]");
                let input = context.create_array_zeros(&[batch_size * ch * dec_frames], dt, "res_in");
                let weight = context.create_array_zeros(&[ch * ch * ksize], dt, "res_w");
                let bias = context.create_array_zeros(&[ch], dt, "res_b");
                let mut output =
                    context.create_array_zeros(&[batch_size * ch * dec_frames], dt, "res_out").into_allocation();
                let lengths = make_lengths(&context, batch_size, dec_frames as i32);

                let ms = measure(&label, || {
                    let mut encoder = Encoder::<Metal>::new(context.as_ref()).unwrap();
                    {
                        let i = input.allocation();
                        let w = weight.allocation();
                        let b = bias.allocation();
                        let o = &mut output;
                        let l = lengths.allocation();
                        kernel.encode(
                            i,
                            w,
                            b,
                            o,
                            l,
                            ch as i32,
                            ch as i32,
                            dec_frames as i32,
                            ksize as i32,
                            dilation as i32,
                            0_i32,
                            batch_size as i32,
                            &mut encoder,
                        );
                    }
                    encoder.end_encoding().submit().wait_until_completed().unwrap();
                });
                results.push((format!("CausalConv1d res dec{block_idx} d={dilation}"), ms));
            }
        }

        // final_conv: 96->1, k=7
        {
            let cin = DEC_CHANNELS[4]; // 96
            let cout = 1;
            let ksize = 7;
            let final_frames = SEMANTIC_FRAMES_10S * 2 * 2 * 8 * 8 * 4 * 2; // full resolution
            let input = context.create_array_zeros(&[batch_size * cin * final_frames], dt, "fc_in");
            let weight = context.create_array_zeros(&[cout * cin * ksize], dt, "fc_w");
            let bias = context.create_array_zeros(&[cout], dt, "fc_b");
            let mut output =
                context.create_array_zeros(&[batch_size * cout * final_frames], dt, "fc_out").into_allocation();
            let lengths = make_lengths(&context, batch_size, final_frames as i32);

            let label = format!("CausalConv1d final_conv [{cin}->{cout}, k={ksize}, T={final_frames}]");
            let ms = measure(&label, || {
                let mut encoder = Encoder::<Metal>::new(context.as_ref()).unwrap();
                {
                    let i = input.allocation();
                    let w = weight.allocation();
                    let b = bias.allocation();
                    let o = &mut output;
                    let l = lengths.allocation();
                    kernel.encode(
                        i,
                        w,
                        b,
                        o,
                        l,
                        cin as i32,
                        cout as i32,
                        final_frames as i32,
                        ksize as i32,
                        1_i32,
                        0_i32,
                        batch_size as i32,
                        &mut encoder,
                    );
                }
                encoder.end_encoding().submit().wait_until_completed().unwrap();
            });
            results.push(("CausalConv1d final_conv".into(), ms));
        }
    }

    // -----------------------------------------------------------------------
    // 4. CausalConv1dGroupedResidual — residual conv2 at each decoder stage
    //    Same dimensions as conv1 but with residual add, dilation=1
    // -----------------------------------------------------------------------
    {
        let kernel =
            <<Metal as Backend>::Kernels as Kernels>::AudioCausalConv1dGroupedResidualKernel::new(&context, dt)
                .unwrap();

        let mut dec_frames = SEMANTIC_FRAMES_10S * UP_STRIDES[0] * UP_STRIDES[1];
        for block_idx in 0..4 {
            dec_frames *= DEC_STRIDES[block_idx];
            let ch = DEC_CHANNELS[block_idx + 1];
            let ksize = 7;

            let label = format!("CausalConv1dResidual dec{block_idx} [{ch}->{ch}, k={ksize}, T={dec_frames}]");
            let input = context.create_array_zeros(&[batch_size * ch * dec_frames], dt, "res_in");
            let residual = context.create_array_zeros(&[batch_size * ch * dec_frames], dt, "res_res");
            let weight = context.create_array_zeros(&[ch * ch * ksize], dt, "res_w");
            let bias = context.create_array_zeros(&[ch], dt, "res_b");
            let mut output =
                context.create_array_zeros(&[batch_size * ch * dec_frames], dt, "res_out").into_allocation();
            let lengths = make_lengths(&context, batch_size, dec_frames as i32);

            let ms = measure(&label, || {
                let mut encoder = Encoder::<Metal>::new(context.as_ref()).unwrap();
                {
                    let i = input.allocation();
                    let r = residual.allocation();
                    let w = weight.allocation();
                    let b = bias.allocation();
                    let o = &mut output;
                    let l = lengths.allocation();
                    kernel.encode(
                        i,
                        r,
                        w,
                        b,
                        o,
                        l,
                        ch as i32,
                        ch as i32,
                        dec_frames as i32,
                        ksize as i32,
                        1_i32,
                        1_i32,
                        batch_size as i32,
                        &mut encoder,
                    );
                }
                encoder.end_encoding().submit().wait_until_completed().unwrap();
            });
            results.push((format!("CausalConv1dResidual dec{block_idx}"), ms));
        }
    }

    // -----------------------------------------------------------------------
    // 5. CausalConv1dGrouped (depthwise) — ConvNeXt depthwise at upsampler resolutions
    // -----------------------------------------------------------------------
    {
        let kernel =
            <<Metal as Backend>::Kernels as Kernels>::AudioCausalConv1dGroupedKernel::new(&context, dt).unwrap();

        let mut frames = SEMANTIC_FRAMES_10S;
        for block_idx in 0..2 {
            frames *= UP_STRIDES[block_idx];
            let ch = UP_CHANNELS[block_idx + 1];
            let ksize = 7;

            let label = format!("CausalConv1dGrouped dw up{block_idx} [{ch}, g={ch}, k={ksize}, T={frames}]");
            let input = context.create_array_zeros(&[batch_size * ch * frames], dt, "dw_in");
            let weight = context.create_array_zeros(&[ch * 1 * ksize], dt, "dw_w");
            let bias = context.create_array_zeros(&[ch], dt, "dw_b");
            let mut output = context.create_array_zeros(&[batch_size * ch * frames], dt, "dw_out").into_allocation();
            let lengths = make_lengths(&context, batch_size, frames as i32);

            let ms = measure(&label, || {
                let mut encoder = Encoder::<Metal>::new(context.as_ref()).unwrap();
                {
                    let i = input.allocation();
                    let w = weight.allocation();
                    let b = bias.allocation();
                    let o = &mut output;
                    let l = lengths.allocation();
                    kernel.encode(
                        i,
                        w,
                        b,
                        o,
                        l,
                        ch as i32,
                        ch as i32,
                        frames as i32,
                        ksize as i32,
                        1_i32,
                        ch as i32,
                        0_i32,
                        batch_size as i32,
                        &mut encoder,
                    );
                }
                encoder.end_encoding().submit().wait_until_completed().unwrap();
            });
            results.push((format!("CausalConv1dGrouped dw up{block_idx}"), ms));
        }
    }

    // -----------------------------------------------------------------------
    // 6. HalfSnake — at each decoder stage resolution
    // -----------------------------------------------------------------------
    {
        let kernel = <<Metal as Backend>::Kernels as Kernels>::AudioHalfSnakeKernel::new(&context, dt).unwrap();

        // Before decoder blocks: snake on decoder_dim channels at upsampled frames
        let fc_frames = SEMANTIC_FRAMES_10S * UP_STRIDES[0] * UP_STRIDES[1];
        let mut dec_frames = fc_frames;
        for block_idx in 0..4 {
            // Snake before trans_conv (at input resolution)
            let ch_in = DEC_CHANNELS[block_idx];
            {
                let label = format!("HalfSnake pre-tconv dec{block_idx} [{ch_in}ch, T={dec_frames}]");
                let input = context.create_array_zeros(&[batch_size * ch_in * dec_frames], dt, "sn_in");
                let alpha = context.create_array_zeros(&[ch_in], dt, "sn_a");
                let mut output =
                    context.create_array_zeros(&[batch_size * ch_in * dec_frames], dt, "sn_out").into_allocation();

                let ms = measure(&label, || {
                    let mut encoder = Encoder::<Metal>::new(context.as_ref()).unwrap();
                    {
                        let i = input.allocation();
                        let a = alpha.allocation();
                        let o = &mut output;
                        kernel.encode(
                            i,
                            a,
                            o,
                            ch_in as i32,
                            dec_frames as i32,
                            ch_in as i32,
                            0.0_f32,
                            1e-9_f32,
                            batch_size as i32,
                            &mut encoder,
                        );
                    }
                    encoder.end_encoding().submit().wait_until_completed().unwrap();
                });
                results.push((format!("HalfSnake pre dec{block_idx}"), ms));
            }

            dec_frames *= DEC_STRIDES[block_idx];
            // Snake in residual units (at output resolution)
            let ch_out = DEC_CHANNELS[block_idx + 1];
            {
                let label = format!("HalfSnake res dec{block_idx} [{ch_out}ch, T={dec_frames}]");
                let input = context.create_array_zeros(&[batch_size * ch_out * dec_frames], dt, "sn_in");
                let alpha = context.create_array_zeros(&[ch_out], dt, "sn_a");
                let mut output =
                    context.create_array_zeros(&[batch_size * ch_out * dec_frames], dt, "sn_out").into_allocation();

                let ms = measure(&label, || {
                    let mut encoder = Encoder::<Metal>::new(context.as_ref()).unwrap();
                    {
                        let i = input.allocation();
                        let a = alpha.allocation();
                        let o = &mut output;
                        kernel.encode(
                            i,
                            a,
                            o,
                            ch_out as i32,
                            dec_frames as i32,
                            ch_out as i32,
                            0.0_f32,
                            1e-9_f32,
                            batch_size as i32,
                            &mut encoder,
                        );
                    }
                    encoder.end_encoding().submit().wait_until_completed().unwrap();
                });
                results.push((format!("HalfSnake res dec{block_idx}"), ms));
            }
        }
    }

    // -----------------------------------------------------------------------
    // 7. NormNcs and Conv1d pointwise — ConvNeXt blocks in upsampler
    // -----------------------------------------------------------------------
    {
        let k_norm = <<Metal as Backend>::Kernels as Kernels>::AudioNormNcsKernel::new(&context, dt).unwrap();
        let k_conv1d = <<Metal as Backend>::Kernels as Kernels>::AudioConv1dKernel::new(&context, dt).unwrap();
        let k_act = <<Metal as Backend>::Kernels as Kernels>::ActivationKernel::new(&context, dt, false).unwrap();

        let mut frames = SEMANTIC_FRAMES_10S;
        for block_idx in 0..2 {
            frames *= UP_STRIDES[block_idx];
            let ch = UP_CHANNELS[block_idx + 1];

            // Norm
            {
                let label = format!("NormNcs up{block_idx} [{ch}ch, T={frames}]");
                let input = context.create_array_zeros(&[batch_size * ch * frames], dt, "norm_in");
                let scales = context.create_array_zeros(&[ch], dt, "norm_s");
                let bias = context.create_array_zeros(&[ch], dt, "norm_b");
                let mut output =
                    context.create_array_zeros(&[batch_size * ch * frames], dt, "norm_out").into_allocation();
                let lengths = make_lengths(&context, batch_size, frames as i32);

                let ms = measure(&label, || {
                    let mut encoder = Encoder::<Metal>::new(context.as_ref()).unwrap();
                    {
                        let i = input.allocation();
                        let s = scales.allocation();
                        let b = bias.allocation();
                        let o = &mut output;
                        let l = lengths.allocation();
                        k_norm.encode(
                            i,
                            s,
                            b,
                            o,
                            l,
                            ch as i32,
                            frames as i32,
                            1e-6_f32,
                            1_i32,
                            batch_size as i32,
                            &mut encoder,
                        );
                    }
                    encoder.end_encoding().submit().wait_until_completed().unwrap();
                });
                results.push((format!("NormNcs up{block_idx}"), ms));
            }

            // Pointwise conv1 (ch -> ch)
            {
                let label = format!("Conv1d pw up{block_idx} [{ch}->{ch}, k=1, T={frames}]");
                let input = context.create_array_zeros(&[batch_size * ch * frames], dt, "pw_in");
                let weight = context.create_array_zeros(&[ch * ch], dt, "pw_w");
                let bias = context.create_array_zeros(&[ch], dt, "pw_b");
                let mut output =
                    context.create_array_zeros(&[batch_size * ch * frames], dt, "pw_out").into_allocation();
                let lengths = make_lengths(&context, batch_size, frames as i32);

                let ms = measure(&label, || {
                    let mut encoder = Encoder::<Metal>::new(context.as_ref()).unwrap();
                    {
                        let i = input.allocation();
                        let w = weight.allocation();
                        let b = bias.allocation();
                        let o = &mut output;
                        let l = lengths.allocation();
                        k_conv1d.encode(
                            i,
                            w,
                            b,
                            o,
                            l,
                            ch as i32,
                            ch as i32,
                            frames as i32,
                            frames as i32,
                            1_i32,
                            1_i32,
                            1_i32,
                            0_i32,
                            0_i32,
                            batch_size as i32,
                            &mut encoder,
                        );
                    }
                    encoder.end_encoding().submit().wait_until_completed().unwrap();
                });
                results.push((format!("Conv1d pw up{block_idx}"), ms));
            }

            // GELU
            {
                let n = ch * frames;
                let label = format!("GELU up{block_idx} [n={n}]");
                let input = context.create_array_zeros(&[n], dt, "gelu_in");
                let mut output = context.create_array_zeros(&[n], dt, "gelu_out").into_allocation();

                let ms = measure(&label, || {
                    let mut encoder = Encoder::<Metal>::new(context.as_ref()).unwrap();
                    {
                        let i = input.allocation();
                        let o = &mut output;
                        k_act.encode(Some(i), o, n as u32, ActivationType::GELU, &mut encoder);
                    }
                    encoder.end_encoding().submit().wait_until_completed().unwrap();
                });
                results.push((format!("GELU up{block_idx}"), ms));
            }
        }
    }

    // =======================================================================
    // Full vocoder pipeline simulation — single command buffer
    //
    // Matches submit_decode_padded flow:
    //   quantizer -> 2 upsample blocks -> first_conv -> 4 decoder blocks -> final snake+conv+tanh
    // =======================================================================
    println!();
    println!("  {:-<65} {:-<8}", "", "");
    println!("  {:<65} {:>8}", "Full Pipeline Simulation (single command buffer)", "Mean (ms)");
    println!("  {:-<65} {:-<8}", "", "");

    // Build all kernels
    let k_quant = <<Metal as Backend>::Kernels as Kernels>::AudioQuantizerDecodeKernel::new(&context, dt).unwrap();
    let k_tconv =
        <<Metal as Backend>::Kernels as Kernels>::AudioCausalConvTranspose1dCausalPadKernel::new(&context, dt).unwrap();
    let k_norm = <<Metal as Backend>::Kernels as Kernels>::AudioNormNcsKernel::new(&context, dt).unwrap();
    let k_conv1d = <<Metal as Backend>::Kernels as Kernels>::AudioConv1dKernel::new(&context, dt).unwrap();
    let k_act = <<Metal as Backend>::Kernels as Kernels>::ActivationKernel::new(&context, dt, false).unwrap();
    let k_add = <<Metal as Backend>::Kernels as Kernels>::AudioAddKernel::new(&context, dt).unwrap();
    let k_snake = <<Metal as Backend>::Kernels as Kernels>::AudioHalfSnakeKernel::new(&context, dt).unwrap();
    let k_conv_grp =
        <<Metal as Backend>::Kernels as Kernels>::AudioCausalConv1dGroupedKernel::new(&context, dt).unwrap();
    let k_conv1d_causal = <<Metal as Backend>::Kernels as Kernels>::AudioCausalConv1dKernel::new(&context, dt).unwrap();
    let k_conv_res =
        <<Metal as Backend>::Kernels as Kernels>::AudioCausalConv1dGroupedResidualKernel::new(&context, dt).unwrap();

    let frames = SEMANTIC_FRAMES_10S;

    // Quantizer arrays
    let p_q_tok = context.create_array_zeros(&[batch_size * TOTAL_CODEBOOKS * frames], DataType::U32, "pipe_q_tok");
    let p_q_len = make_lengths(&context, batch_size, frames as i32);
    let p_q_sem_cb = context.create_array_zeros(&[SEMANTIC_CARDINALITY * CODEBOOK_DIM], dt, "p_sem_cb");
    let p_q_sem_proj = context.create_array_zeros(&[INPUT_DIM * CODEBOOK_DIM], dt, "p_sem_proj");
    let p_q_sem_bias = context.create_array_zeros(&[INPUT_DIM], dt, "p_sem_bias");
    let p_q_res_cbs =
        context.create_array_zeros(&[RESIDUAL_QUANTIZERS * RESIDUAL_CARDINALITY * CODEBOOK_DIM], dt, "p_res_cbs");
    let p_q_res_proj = context.create_array_zeros(&[RESIDUAL_QUANTIZERS * INPUT_DIM * CODEBOOK_DIM], dt, "p_res_proj");
    let p_q_res_bias = context.create_array_zeros(&[RESIDUAL_QUANTIZERS * INPUT_DIM], dt, "p_res_bias");

    // Compute frame counts at each stage
    let f_up0 = frames * UP_STRIDES[0]; // 430
    let f_up1 = f_up0 * UP_STRIDES[1]; // 860
    let f_dec0 = f_up1 * DEC_STRIDES[0]; // 6880
    let f_dec1 = f_dec0 * DEC_STRIDES[1]; // 55040
    let f_dec2 = f_dec1 * DEC_STRIDES[2]; // 220160
    let f_dec3 = f_dec2 * DEC_STRIDES[3]; // 440320

    // Pipeline scratch: large enough for the biggest intermediate
    // Biggest is decoder_dim * f_up1 or ch * f_dec stages
    let max_elems: usize = batch_size * DECODER_DIM * f_dec3.max(f_up1);
    let mut scratch_a = context.create_array_zeros(&[max_elems], dt, "pipe_sa").into_allocation();
    let mut scratch_b = context.create_array_zeros(&[max_elems], dt, "pipe_sb").into_allocation();
    let scratch_r = context.create_array_zeros(&[max_elems], dt, "pipe_sr").into_allocation();

    // Upsample block 0: 1024->512, stride=2
    let u0_w = context.create_array_zeros(&[UP_CHANNELS[0] * UP_CHANNELS[1] * (UP_STRIDES[0] * 2)], dt, "u0_w");
    let u0_b = context.create_array_zeros(&[UP_CHANNELS[1]], dt, "u0_b");
    // ConvNeXt 0 (ch=512)
    let c0_ch = UP_CHANNELS[1];
    let c0_dw_w = context.create_array_zeros(&[c0_ch * 1 * 7], dt, "c0_dw_w");
    let c0_dw_b = context.create_array_zeros(&[c0_ch], dt, "c0_dw_b");
    let c0_ns = context.create_array_zeros(&[c0_ch], dt, "c0_ns");
    let c0_nb = context.create_array_zeros(&[c0_ch], dt, "c0_nb");
    let c0_p1_w = context.create_array_zeros(&[c0_ch * c0_ch], dt, "c0_p1_w");
    let c0_p1_b = context.create_array_zeros(&[c0_ch], dt, "c0_p1_b");
    let c0_p2_w = context.create_array_zeros(&[c0_ch * c0_ch], dt, "c0_p2_w");
    let c0_p2_b = context.create_array_zeros(&[c0_ch], dt, "c0_p2_b");

    // Upsample block 1: 512->256, stride=2
    let u1_w = context.create_array_zeros(&[UP_CHANNELS[1] * UP_CHANNELS[2] * (UP_STRIDES[1] * 2)], dt, "u1_w");
    let u1_b = context.create_array_zeros(&[UP_CHANNELS[2]], dt, "u1_b");
    // ConvNeXt 1 (ch=256)
    let c1_ch = UP_CHANNELS[2];
    let c1_dw_w = context.create_array_zeros(&[c1_ch * 1 * 7], dt, "c1_dw_w");
    let c1_dw_b = context.create_array_zeros(&[c1_ch], dt, "c1_dw_b");
    let c1_ns = context.create_array_zeros(&[c1_ch], dt, "c1_ns");
    let c1_nb = context.create_array_zeros(&[c1_ch], dt, "c1_nb");
    let c1_p1_w = context.create_array_zeros(&[c1_ch * c1_ch], dt, "c1_p1_w");
    let c1_p1_b = context.create_array_zeros(&[c1_ch], dt, "c1_p1_b");
    let c1_p2_w = context.create_array_zeros(&[c1_ch * c1_ch], dt, "c1_p2_w");
    let c1_p2_b = context.create_array_zeros(&[c1_ch], dt, "c1_p2_b");

    // first_conv: 256->1536, k=7
    let fc_w = context.create_array_zeros(&[DECODER_DIM * UP_CHANNELS[2] * 7], dt, "fc_w");
    let fc_b = context.create_array_zeros(&[DECODER_DIM], dt, "fc_b");

    // Decoder block weights for all 4 blocks
    struct DecBlockWeights {
        snake_a: Array<Metal>,
        tconv_w: Array<Metal>,
        tconv_b: Array<Metal>,
        // 3 residual units, each with snake1, conv1, snake2, conv2
        ru_s1: [Array<Metal>; 3],
        ru_c1_w: [Array<Metal>; 3],
        ru_c1_b: [Array<Metal>; 3],
        ru_s2: [Array<Metal>; 3],
        ru_c2_w: [Array<Metal>; 3],
        ru_c2_b: [Array<Metal>; 3],
    }

    let dec_blocks: Vec<DecBlockWeights> = (0..4)
        .map(|i| {
            let cin = DEC_CHANNELS[i];
            let cout = DEC_CHANNELS[i + 1];
            let stride = DEC_STRIDES[i];
            let ksize = stride * 2;
            DecBlockWeights {
                snake_a: context.create_array_zeros(&[cin], dt, &format!("d{i}_sa")),
                tconv_w: context.create_array_zeros(&[cin * cout * ksize], dt, &format!("d{i}_tw")),
                tconv_b: context.create_array_zeros(&[cout], dt, &format!("d{i}_tb")),
                ru_s1: std::array::from_fn(|j| context.create_array_zeros(&[cout], dt, &format!("d{i}_r{j}_s1"))),
                ru_c1_w: std::array::from_fn(|j| {
                    context.create_array_zeros(&[cout * cout * 7], dt, &format!("d{i}_r{j}_c1w"))
                }),
                ru_c1_b: std::array::from_fn(|j| context.create_array_zeros(&[cout], dt, &format!("d{i}_r{j}_c1b"))),
                ru_s2: std::array::from_fn(|j| context.create_array_zeros(&[cout], dt, &format!("d{i}_r{j}_s2"))),
                ru_c2_w: std::array::from_fn(|j| {
                    context.create_array_zeros(&[cout * cout * 7], dt, &format!("d{i}_r{j}_c2w"))
                }),
                ru_c2_b: std::array::from_fn(|j| context.create_array_zeros(&[cout], dt, &format!("d{i}_r{j}_c2b"))),
            }
        })
        .collect();

    // final snake + conv + tanh
    let final_snake_a = context.create_array_zeros(&[DEC_CHANNELS[4]], dt, "final_sa");
    let final_conv_w = context.create_array_zeros(&[1 * DEC_CHANNELS[4] * 7], dt, "final_cw");
    let final_conv_b = context.create_array_zeros(&[1], dt, "final_cb");

    // Length arrays
    let len_up0 = make_lengths(&context, batch_size, f_up0 as i32);
    let len_up1 = make_lengths(&context, batch_size, f_up1 as i32);
    let len_dec0 = make_lengths(&context, batch_size, f_dec0 as i32);
    let len_dec1 = make_lengths(&context, batch_size, f_dec1 as i32);
    let len_dec2 = make_lengths(&context, batch_size, f_dec2 as i32);
    let len_dec3 = make_lengths(&context, batch_size, f_dec3 as i32);
    let dec_lens = [&len_dec0, &len_dec1, &len_dec2, &len_dec3];
    let dec_frame_sizes = [f_dec0, f_dec1, f_dec2, f_dec3];
    let dilations = [1_i32, 3, 9];

    let ms = measure("Full pipeline (quant -> 2 up -> first_conv -> 4 dec -> final)", || {
        let mut encoder = Encoder::<Metal>::new(context.as_ref()).unwrap();

        // --- Quantizer: output [B, T, input_dim] (NSC) ---
        {
            let tok = p_q_tok.allocation();
            let len = p_q_len.allocation();
            let scb = p_q_sem_cb.allocation();
            let sp = p_q_sem_proj.allocation();
            let sbi = p_q_sem_bias.allocation();
            let rcb = p_q_res_cbs.allocation();
            let rp = p_q_res_proj.allocation();
            let rb = p_q_res_bias.allocation();
            k_quant.encode(
                tok,
                len,
                scb,
                sp,
                sbi,
                rcb,
                rp,
                rb,
                &mut scratch_a,
                batch_size as i32,
                TOTAL_CODEBOOKS as i32,
                frames as i32,
                INPUT_DIM as i32,
                CODEBOOK_DIM as i32,
                RESIDUAL_QUANTIZERS as i32,
                SEMANTIC_CARDINALITY as i32,
                RESIDUAL_CARDINALITY as i32,
                &mut encoder,
            );
        }

        // --- Upsample block 0: 1024->512, stride=2, NSC input ---
        {
            let w = u0_w.allocation();
            let b = u0_b.allocation();
            let l = len_up0.allocation();
            k_tconv.encode(
                &scratch_a,
                w,
                b,
                &mut scratch_b,
                l,
                UP_CHANNELS[0] as i32,
                UP_CHANNELS[1] as i32,
                frames as i32,
                f_up0 as i32,
                (UP_STRIDES[0] * 2) as i32,
                UP_STRIDES[0] as i32,
                1_i32,
                1_i32,
                batch_size as i32,
                &mut encoder,
            );
        }
        // ConvNeXt 0
        {
            let w = c0_dw_w.allocation();
            let b = c0_dw_b.allocation();
            let l = len_up0.allocation();
            k_conv_grp.encode(
                &scratch_b,
                w,
                b,
                &mut scratch_a,
                l,
                c0_ch as i32,
                c0_ch as i32,
                f_up0 as i32,
                7_i32,
                1_i32,
                c0_ch as i32,
                0_i32,
                batch_size as i32,
                &mut encoder,
            );
        }
        {
            let s = c0_ns.allocation();
            let b = c0_nb.allocation();
            let l = len_up0.allocation();
            k_norm.encode(
                &scratch_a,
                s,
                b,
                &mut scratch_b,
                l,
                c0_ch as i32,
                f_up0 as i32,
                1e-6_f32,
                1_i32,
                batch_size as i32,
                &mut encoder,
            );
        }
        {
            let w = c0_p1_w.allocation();
            let b = c0_p1_b.allocation();
            let l = len_up0.allocation();
            k_conv1d.encode(
                &scratch_b,
                w,
                b,
                &mut scratch_a,
                l,
                c0_ch as i32,
                c0_ch as i32,
                f_up0 as i32,
                f_up0 as i32,
                1_i32,
                1_i32,
                1_i32,
                0_i32,
                0_i32,
                batch_size as i32,
                &mut encoder,
            );
        }
        {
            k_act.encode(Some(&scratch_a), &mut scratch_b, (c0_ch * f_up0) as u32, ActivationType::GELU, &mut encoder);
        }
        {
            let w = c0_p2_w.allocation();
            let b = c0_p2_b.allocation();
            let l = len_up0.allocation();
            k_conv1d.encode(
                &scratch_b,
                w,
                b,
                &mut scratch_a,
                l,
                c0_ch as i32,
                c0_ch as i32,
                f_up0 as i32,
                f_up0 as i32,
                1_i32,
                1_i32,
                1_i32,
                0_i32,
                0_i32,
                batch_size as i32,
                &mut encoder,
            );
        }
        {
            k_add.encode(&scratch_a, &scratch_a, &mut scratch_b, (c0_ch * f_up0) as i32, &mut encoder);
        }

        // --- Upsample block 1: 512->256, stride=2, NCS input ---
        {
            let w = u1_w.allocation();
            let b = u1_b.allocation();
            let l = len_up1.allocation();
            k_tconv.encode(
                &scratch_b,
                w,
                b,
                &mut scratch_a,
                l,
                UP_CHANNELS[1] as i32,
                UP_CHANNELS[2] as i32,
                f_up0 as i32,
                f_up1 as i32,
                (UP_STRIDES[1] * 2) as i32,
                UP_STRIDES[1] as i32,
                1_i32,
                0_i32,
                batch_size as i32,
                &mut encoder,
            );
        }
        // ConvNeXt 1
        {
            let w = c1_dw_w.allocation();
            let b = c1_dw_b.allocation();
            let l = len_up1.allocation();
            k_conv_grp.encode(
                &scratch_a,
                w,
                b,
                &mut scratch_b,
                l,
                c1_ch as i32,
                c1_ch as i32,
                f_up1 as i32,
                7_i32,
                1_i32,
                c1_ch as i32,
                0_i32,
                batch_size as i32,
                &mut encoder,
            );
        }
        {
            let s = c1_ns.allocation();
            let b = c1_nb.allocation();
            let l = len_up1.allocation();
            k_norm.encode(
                &scratch_b,
                s,
                b,
                &mut scratch_a,
                l,
                c1_ch as i32,
                f_up1 as i32,
                1e-6_f32,
                1_i32,
                batch_size as i32,
                &mut encoder,
            );
        }
        {
            let w = c1_p1_w.allocation();
            let b = c1_p1_b.allocation();
            let l = len_up1.allocation();
            k_conv1d.encode(
                &scratch_a,
                w,
                b,
                &mut scratch_b,
                l,
                c1_ch as i32,
                c1_ch as i32,
                f_up1 as i32,
                f_up1 as i32,
                1_i32,
                1_i32,
                1_i32,
                0_i32,
                0_i32,
                batch_size as i32,
                &mut encoder,
            );
        }
        {
            k_act.encode(Some(&scratch_b), &mut scratch_a, (c1_ch * f_up1) as u32, ActivationType::GELU, &mut encoder);
        }
        {
            let w = c1_p2_w.allocation();
            let b = c1_p2_b.allocation();
            let l = len_up1.allocation();
            k_conv1d.encode(
                &scratch_a,
                w,
                b,
                &mut scratch_b,
                l,
                c1_ch as i32,
                c1_ch as i32,
                f_up1 as i32,
                f_up1 as i32,
                1_i32,
                1_i32,
                1_i32,
                0_i32,
                0_i32,
                batch_size as i32,
                &mut encoder,
            );
        }
        {
            k_add.encode(&scratch_b, &scratch_b, &mut scratch_a, (c1_ch * f_up1) as i32, &mut encoder);
        }

        // --- first_conv: 256->1536, k=7 ---
        {
            let w = fc_w.allocation();
            let b = fc_b.allocation();
            let l = len_up1.allocation();
            k_conv1d_causal.encode(
                &scratch_a,
                w,
                b,
                &mut scratch_b,
                l,
                UP_CHANNELS[2] as i32,
                DECODER_DIM as i32,
                f_up1 as i32,
                7_i32,
                1_i32,
                0_i32,
                batch_size as i32,
                &mut encoder,
            );
        }

        // --- 4 Decoder blocks ---
        // State in sb after first_conv
        let mut cur_in_sa = false; // data is in sb
        let mut cur_frames = f_up1;

        for block_idx in 0..4 {
            let blk = &dec_blocks[block_idx];
            let cin = DEC_CHANNELS[block_idx];
            let cout = DEC_CHANNELS[block_idx + 1];
            let stride = DEC_STRIDES[block_idx];
            let ksize = stride * 2;
            let next_frames = dec_frame_sizes[block_idx];
            let dec_len = dec_lens[block_idx];

            // snake (in-place swap)
            {
                let (i, o) = if cur_in_sa {
                    (&scratch_a, &mut scratch_b)
                } else {
                    (&scratch_b, &mut scratch_a)
                };
                let a = blk.snake_a.allocation();
                k_snake.encode(
                    i,
                    a,
                    o,
                    cin as i32,
                    cur_frames as i32,
                    cin as i32,
                    0.0_f32,
                    1e-9_f32,
                    batch_size as i32,
                    &mut encoder,
                );
            }
            cur_in_sa = !cur_in_sa;

            // trans_conv
            {
                let (i, o) = if cur_in_sa {
                    (&scratch_a, &mut scratch_b)
                } else {
                    (&scratch_b, &mut scratch_a)
                };
                let w = blk.tconv_w.allocation();
                let b = blk.tconv_b.allocation();
                let l = dec_len.allocation();
                k_tconv.encode(
                    i,
                    w,
                    b,
                    o,
                    l,
                    cin as i32,
                    cout as i32,
                    cur_frames as i32,
                    next_frames as i32,
                    ksize as i32,
                    stride as i32,
                    1_i32,
                    0_i32,
                    batch_size as i32,
                    &mut encoder,
                );
            }
            cur_in_sa = !cur_in_sa;
            cur_frames = next_frames;

            // 3 residual units
            for ru_idx in 0..3 {
                // snake1
                {
                    let (i, o) = if cur_in_sa {
                        (&scratch_a, &mut scratch_b)
                    } else {
                        (&scratch_b, &mut scratch_a)
                    };
                    let a = blk.ru_s1[ru_idx].allocation();
                    k_snake.encode(
                        i,
                        a,
                        o,
                        cout as i32,
                        cur_frames as i32,
                        cout as i32,
                        0.0_f32,
                        1e-9_f32,
                        batch_size as i32,
                        &mut encoder,
                    );
                }
                cur_in_sa = !cur_in_sa;

                // conv1 (dilation varies: 1, 3, 9)
                {
                    let (i, o) = if cur_in_sa {
                        (&scratch_a, &mut scratch_b)
                    } else {
                        (&scratch_b, &mut scratch_a)
                    };
                    let w = blk.ru_c1_w[ru_idx].allocation();
                    let b = blk.ru_c1_b[ru_idx].allocation();
                    let l = dec_len.allocation();
                    k_conv1d_causal.encode(
                        i,
                        w,
                        b,
                        o,
                        l,
                        cout as i32,
                        cout as i32,
                        cur_frames as i32,
                        7_i32,
                        dilations[ru_idx],
                        0_i32,
                        batch_size as i32,
                        &mut encoder,
                    );
                }
                cur_in_sa = !cur_in_sa;

                // snake2
                {
                    let (i, o) = if cur_in_sa {
                        (&scratch_a, &mut scratch_b)
                    } else {
                        (&scratch_b, &mut scratch_a)
                    };
                    let a = blk.ru_s2[ru_idx].allocation();
                    k_snake.encode(
                        i,
                        a,
                        o,
                        cout as i32,
                        cur_frames as i32,
                        cout as i32,
                        0.0_f32,
                        1e-9_f32,
                        batch_size as i32,
                        &mut encoder,
                    );
                }
                cur_in_sa = !cur_in_sa;

                // conv2 + residual add
                {
                    let (i, o) = if cur_in_sa {
                        (&scratch_a, &mut scratch_b)
                    } else {
                        (&scratch_b, &mut scratch_a)
                    };
                    let r = &scratch_r;
                    let w = blk.ru_c2_w[ru_idx].allocation();
                    let b = blk.ru_c2_b[ru_idx].allocation();
                    let l = dec_len.allocation();
                    k_conv_res.encode(
                        i,
                        r,
                        w,
                        b,
                        o,
                        l,
                        cout as i32,
                        cout as i32,
                        cur_frames as i32,
                        7_i32,
                        1_i32,
                        1_i32,
                        batch_size as i32,
                        &mut encoder,
                    );
                }
                cur_in_sa = !cur_in_sa;
            }
        }

        // --- final snake + conv + tanh ---
        {
            let (i, o) = if cur_in_sa {
                (&scratch_a, &mut scratch_b)
            } else {
                (&scratch_b, &mut scratch_a)
            };
            let a = final_snake_a.allocation();
            k_snake.encode(
                i,
                a,
                o,
                DEC_CHANNELS[4] as i32,
                f_dec3 as i32,
                DEC_CHANNELS[4] as i32,
                0.0_f32,
                1e-9_f32,
                batch_size as i32,
                &mut encoder,
            );
        }
        cur_in_sa = !cur_in_sa;
        {
            let (i, o) = if cur_in_sa {
                (&scratch_a, &mut scratch_b)
            } else {
                (&scratch_b, &mut scratch_a)
            };
            let w = final_conv_w.allocation();
            let b = final_conv_b.allocation();
            let l = len_dec3.allocation();
            k_conv1d_causal.encode(
                i,
                w,
                b,
                o,
                l,
                DEC_CHANNELS[4] as i32,
                1_i32,
                f_dec3 as i32,
                7_i32,
                1_i32,
                0_i32,
                batch_size as i32,
                &mut encoder,
            );
        }
        cur_in_sa = !cur_in_sa;
        // tanh
        {
            let (i, o) = if cur_in_sa {
                (&scratch_a, &mut scratch_b)
            } else {
                (&scratch_b, &mut scratch_a)
            };
            k_act.encode(Some(i), o, f_dec3 as u32, ActivationType::TANH, &mut encoder);
        }

        encoder.end_encoding().submit().wait_until_completed().unwrap();
    });
    results.push(("Full pipeline".into(), ms));

    // =======================================================================
    // Summary
    // =======================================================================
    println!();
    println!("  === SUMMARY ===");
    println!("  {:<65} {:>8}", "Kernel", "Mean (ms)");
    println!("  {:-<65} {:-<8}", "", "");
    let mut total_individual = 0.0;
    for (name, ms) in &results {
        println!("  {name:<65} {ms:>8.3}");
        if name != "Full pipeline" {
            total_individual += ms;
        }
    }
    println!("  {:-<65} {:-<8}", "", "");
    println!("  {:<65} {:>8.3}", "Sum of individual kernels", total_individual);
    if let Some((_, pipeline_ms)) = results.iter().find(|(n, _)| n == "Full pipeline") {
        println!("  {:<65} {:>8.3}", "Full pipeline (single CB)", pipeline_ms);
    }
    println!();
}
