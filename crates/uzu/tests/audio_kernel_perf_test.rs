#![cfg(all(feature = "audio-runtime", feature = "metal", target_os = "macos"))]

use std::time::Instant;

use uzu::{
    DataType,
    array::ArrayContextExt,
    backends::{
        common::{
            Backend, CommandBufferEncoding, CommandBufferExecutable,
            CommandBufferInitial, CommandBufferPending, Context, Kernels,
            kernel::{
                ActivationKernel, AudioAddKernel, AudioCausalConv1dGroupedKernel,
                AudioCausalConv1dGroupedResidualKernel, AudioCausalConv1dKernel,
                AudioCausalConvTranspose1dCausalPadKernel, AudioConv1dKernel,
                AudioHalfSnakeKernel, AudioNormNcsKernel, AudioQuantizerDecodeKernel,
            },
        },
        metal::Metal,
    },
};

type Ctx = <Metal as Backend>::Context;

const WARMUP: usize = 5;
const ITERATIONS: usize = 20;

fn measure<F: FnMut()>(label: &str, mut f: F) -> f64 {
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

fn make_lengths(context: &Ctx, batch_size: usize, value: i32) -> uzu::array::Array<Metal> {
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
    println!(
        "  {:<65} {:>8}",
        "Kernel", "Mean (ms)"
    );
    println!("  {:-<65} {:-<8}", "", "");

    // -----------------------------------------------------------------------
    // 1. AudioQuantizerDecodeKernel
    //    tokens: [B, K, T], output: [B, T, input_dim]
    // -----------------------------------------------------------------------
    {
        let frames = SEMANTIC_FRAMES_10S;

        let kernel =
            <<Metal as Backend>::Kernels as Kernels>::AudioQuantizerDecodeKernel::new(&context, dt)
                .unwrap();

        let tokens = context.create_array(
            &[batch_size * TOTAL_CODEBOOKS * frames],
            DataType::U32,
            "perf_quant_tokens",
        );
        let lengths = make_lengths(&context, batch_size, frames as i32);
        let semantic_codebook =
            context.create_array(&[SEMANTIC_CARDINALITY * CODEBOOK_DIM], dt, "perf_sem_cb");
        let semantic_out_proj =
            context.create_array(&[INPUT_DIM * CODEBOOK_DIM], dt, "perf_sem_proj");
        let semantic_out_bias = context.create_array(&[INPUT_DIM], dt, "perf_sem_bias");
        let residual_codebooks = context.create_array(
            &[RESIDUAL_QUANTIZERS * RESIDUAL_CARDINALITY * CODEBOOK_DIM],
            dt,
            "perf_res_cbs",
        );
        let residual_out_proj = context.create_array(
            &[RESIDUAL_QUANTIZERS * INPUT_DIM * CODEBOOK_DIM],
            dt,
            "perf_res_proj",
        );
        let residual_out_bias =
            context.create_array(&[RESIDUAL_QUANTIZERS * INPUT_DIM], dt, "perf_res_bias");
        let output =
            context.create_array(&[batch_size * frames * INPUT_DIM], dt, "perf_quant_out");

        let label = format!(
            "QuantizerDecode [B={}, T={}, K={}, dim={}, cdim={}]",
            batch_size, frames, TOTAL_CODEBOOKS, INPUT_DIM, CODEBOOK_DIM
        );
        let ms = measure(&label, || {
            let mut cb = context.create_command_buffer().unwrap().start_encoding();
            {
                let tok = tokens.buffer(); let tok = tok.borrow();
                let len = lengths.buffer(); let len = len.borrow();
                let scb = semantic_codebook.buffer(); let scb = scb.borrow();
                let sp = semantic_out_proj.buffer(); let sp = sp.borrow();
                let sb = semantic_out_bias.buffer(); let sb = sb.borrow();
                let rcb = residual_codebooks.buffer(); let rcb = rcb.borrow();
                let rp = residual_out_proj.buffer(); let rp = rp.borrow();
                let rb = residual_out_bias.buffer(); let rb = rb.borrow();
                let out = output.buffer(); let mut out = out.borrow_mut();
                kernel.encode(
                    &*tok, &*len, &*scb, &*sp, &*sb, &*rcb, &*rp, &*rb,
                    &mut *out,
                    batch_size as i32, TOTAL_CODEBOOKS as i32,
                    frames as i32, INPUT_DIM as i32, CODEBOOK_DIM as i32,
                    RESIDUAL_QUANTIZERS as i32,
                    SEMANTIC_CARDINALITY as i32, RESIDUAL_CARDINALITY as i32,
                    &mut cb,
                );
            }
            cb.end_encoding().submit().wait_until_completed().unwrap();
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
            <<Metal as Backend>::Kernels as Kernels>::AudioCausalConvTranspose1dCausalPadKernel::new(
                &context, dt,
            )
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
            let input = context.create_array(&[batch_size * cin * seq_in], dt, "tconv_in");
            let weight = context.create_array(&[cin * cout * ksize], dt, "tconv_w");
            let bias = context.create_array(&[cout], dt, "tconv_b");
            let output = context.create_array(&[batch_size * cout * seq_out], dt, "tconv_out");
            let lengths = make_lengths(&context, batch_size, seq_out as i32);

            let label = format!(
                "TransConv {tag} [{cin}->{cout}, s={stride}, k={ksize}, {seq_in}->{seq_out}]"
            );
            let ms = measure(&label, || {
                let mut cb = context.create_command_buffer().unwrap().start_encoding();
                {
                    let i = input.buffer(); let i = i.borrow();
                    let w = weight.buffer(); let w = w.borrow();
                    let b = bias.buffer(); let b = b.borrow();
                    let o = output.buffer(); let mut o = o.borrow_mut();
                    let l = lengths.buffer(); let l = l.borrow();
                    kernel.encode(
                        &*i, &*w, &*b, &mut *o, &*l,
                        cin as i32, cout as i32,
                        seq_in as i32, seq_out as i32,
                        ksize as i32, stride as i32,
                        1_i32, layout, batch_size as i32,
                        &mut cb,
                    );
                }
                cb.end_encoding().submit().wait_until_completed().unwrap();
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
            let input = context.create_array(&[batch_size * cin * seq_in], dt, "tconv_in");
            let weight = context.create_array(&[cin * cout * ksize], dt, "tconv_w");
            let bias = context.create_array(&[cout], dt, "tconv_b");
            let output = context.create_array(&[batch_size * cout * seq_out], dt, "tconv_out");
            let lengths = make_lengths(&context, batch_size, seq_out as i32);

            let label = format!(
                "TransConv dec{block_idx} [{cin}->{cout}, s={stride}, k={ksize}, {seq_in}->{seq_out}]"
            );
            let ms = measure(&label, || {
                let mut cb = context.create_command_buffer().unwrap().start_encoding();
                {
                    let i = input.buffer(); let i = i.borrow();
                    let w = weight.buffer(); let w = w.borrow();
                    let b = bias.buffer(); let b = b.borrow();
                    let o = output.buffer(); let mut o = o.borrow_mut();
                    let l = lengths.buffer(); let l = l.borrow();
                    kernel.encode(
                        &*i, &*w, &*b, &mut *o, &*l,
                        cin as i32, cout as i32,
                        seq_in as i32, seq_out as i32,
                        ksize as i32, stride as i32,
                        1_i32, 0_i32, batch_size as i32,
                        &mut cb,
                    );
                }
                cb.end_encoding().submit().wait_until_completed().unwrap();
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
        let kernel =
            <<Metal as Backend>::Kernels as Kernels>::AudioCausalConv1dKernel::new(&context, dt)
                .unwrap();

        // first_conv after upsampler: 256->1536, k=7
        let fc_seq = SEMANTIC_FRAMES_10S * UP_STRIDES[0] * UP_STRIDES[1]; // 860
        {
            let cin = UP_CHANNELS[2]; // 256
            let cout = DECODER_DIM;   // 1536
            let ksize = 7;
            let input = context.create_array(&[batch_size * cin * fc_seq], dt, "fc_in");
            let weight = context.create_array(&[cout * cin * ksize], dt, "fc_w");
            let bias = context.create_array(&[cout], dt, "fc_b");
            let output = context.create_array(&[batch_size * cout * fc_seq], dt, "fc_out");
            let lengths = make_lengths(&context, batch_size, fc_seq as i32);

            let label = format!("CausalConv1d first_conv [{cin}->{cout}, k={ksize}, T={fc_seq}]");
            let ms = measure(&label, || {
                let mut cb = context.create_command_buffer().unwrap().start_encoding();
                {
                    let i = input.buffer(); let i = i.borrow();
                    let w = weight.buffer(); let w = w.borrow();
                    let b = bias.buffer(); let b = b.borrow();
                    let o = output.buffer(); let mut o = o.borrow_mut();
                    let l = lengths.buffer(); let l = l.borrow();
                    kernel.encode(
                        &*i, &*w, &*b, &mut *o, &*l,
                        cin as i32, cout as i32, fc_seq as i32,
                        ksize as i32, 1_i32, 0_i32, batch_size as i32,
                        &mut cb,
                    );
                }
                cb.end_encoding().submit().wait_until_completed().unwrap();
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
                let label = format!(
                    "CausalConv1d res dec{block_idx} [{ch}->{ch}, k={ksize}, d={dilation}, T={dec_frames}]"
                );
                let input = context.create_array(&[batch_size * ch * dec_frames], dt, "res_in");
                let weight = context.create_array(&[ch * ch * ksize], dt, "res_w");
                let bias = context.create_array(&[ch], dt, "res_b");
                let output = context.create_array(&[batch_size * ch * dec_frames], dt, "res_out");
                let lengths = make_lengths(&context, batch_size, dec_frames as i32);

                let ms = measure(&label, || {
                    let mut cb = context.create_command_buffer().unwrap().start_encoding();
                    {
                        let i = input.buffer(); let i = i.borrow();
                        let w = weight.buffer(); let w = w.borrow();
                        let b = bias.buffer(); let b = b.borrow();
                        let o = output.buffer(); let mut o = o.borrow_mut();
                        let l = lengths.buffer(); let l = l.borrow();
                        kernel.encode(
                            &*i, &*w, &*b, &mut *o, &*l,
                            ch as i32, ch as i32, dec_frames as i32,
                            ksize as i32, dilation as i32, 0_i32, batch_size as i32,
                            &mut cb,
                        );
                    }
                    cb.end_encoding().submit().wait_until_completed().unwrap();
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
            let input = context.create_array(&[batch_size * cin * final_frames], dt, "fc_in");
            let weight = context.create_array(&[cout * cin * ksize], dt, "fc_w");
            let bias = context.create_array(&[cout], dt, "fc_b");
            let output = context.create_array(&[batch_size * cout * final_frames], dt, "fc_out");
            let lengths = make_lengths(&context, batch_size, final_frames as i32);

            let label = format!("CausalConv1d final_conv [{cin}->{cout}, k={ksize}, T={final_frames}]");
            let ms = measure(&label, || {
                let mut cb = context.create_command_buffer().unwrap().start_encoding();
                {
                    let i = input.buffer(); let i = i.borrow();
                    let w = weight.buffer(); let w = w.borrow();
                    let b = bias.buffer(); let b = b.borrow();
                    let o = output.buffer(); let mut o = o.borrow_mut();
                    let l = lengths.buffer(); let l = l.borrow();
                    kernel.encode(
                        &*i, &*w, &*b, &mut *o, &*l,
                        cin as i32, cout as i32, final_frames as i32,
                        ksize as i32, 1_i32, 0_i32, batch_size as i32,
                        &mut cb,
                    );
                }
                cb.end_encoding().submit().wait_until_completed().unwrap();
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
            <<Metal as Backend>::Kernels as Kernels>::AudioCausalConv1dGroupedResidualKernel::new(
                &context, dt,
            )
            .unwrap();

        let mut dec_frames = SEMANTIC_FRAMES_10S * UP_STRIDES[0] * UP_STRIDES[1];
        for block_idx in 0..4 {
            dec_frames *= DEC_STRIDES[block_idx];
            let ch = DEC_CHANNELS[block_idx + 1];
            let ksize = 7;

            let label = format!(
                "CausalConv1dResidual dec{block_idx} [{ch}->{ch}, k={ksize}, T={dec_frames}]"
            );
            let input = context.create_array(&[batch_size * ch * dec_frames], dt, "res_in");
            let residual = context.create_array(&[batch_size * ch * dec_frames], dt, "res_res");
            let weight = context.create_array(&[ch * ch * ksize], dt, "res_w");
            let bias = context.create_array(&[ch], dt, "res_b");
            let output = context.create_array(&[batch_size * ch * dec_frames], dt, "res_out");
            let lengths = make_lengths(&context, batch_size, dec_frames as i32);

            let ms = measure(&label, || {
                let mut cb = context.create_command_buffer().unwrap().start_encoding();
                {
                    let i = input.buffer(); let i = i.borrow();
                    let r = residual.buffer(); let r = r.borrow();
                    let w = weight.buffer(); let w = w.borrow();
                    let b = bias.buffer(); let b = b.borrow();
                    let o = output.buffer(); let mut o = o.borrow_mut();
                    let l = lengths.buffer(); let l = l.borrow();
                    kernel.encode(
                        &*i, &*r, &*w, &*b, &mut *o, &*l,
                        ch as i32, ch as i32, dec_frames as i32,
                        ksize as i32, 1_i32, 1_i32,
                        batch_size as i32,
                        &mut cb,
                    );
                }
                cb.end_encoding().submit().wait_until_completed().unwrap();
            });
            results.push((format!("CausalConv1dResidual dec{block_idx}"), ms));
        }
    }

    // -----------------------------------------------------------------------
    // 5. CausalConv1dGrouped (depthwise) — ConvNeXt depthwise at upsampler resolutions
    // -----------------------------------------------------------------------
    {
        let kernel =
            <<Metal as Backend>::Kernels as Kernels>::AudioCausalConv1dGroupedKernel::new(
                &context, dt,
            )
            .unwrap();

        let mut frames = SEMANTIC_FRAMES_10S;
        for block_idx in 0..2 {
            frames *= UP_STRIDES[block_idx];
            let ch = UP_CHANNELS[block_idx + 1];
            let ksize = 7;

            let label = format!(
                "CausalConv1dGrouped dw up{block_idx} [{ch}, g={ch}, k={ksize}, T={frames}]"
            );
            let input = context.create_array(&[batch_size * ch * frames], dt, "dw_in");
            let weight = context.create_array(&[ch * 1 * ksize], dt, "dw_w");
            let bias = context.create_array(&[ch], dt, "dw_b");
            let output = context.create_array(&[batch_size * ch * frames], dt, "dw_out");
            let lengths = make_lengths(&context, batch_size, frames as i32);

            let ms = measure(&label, || {
                let mut cb = context.create_command_buffer().unwrap().start_encoding();
                {
                    let i = input.buffer(); let i = i.borrow();
                    let w = weight.buffer(); let w = w.borrow();
                    let b = bias.buffer(); let b = b.borrow();
                    let o = output.buffer(); let mut o = o.borrow_mut();
                    let l = lengths.buffer(); let l = l.borrow();
                    kernel.encode(
                        &*i, &*w, &*b, &mut *o, &*l,
                        ch as i32, ch as i32, frames as i32,
                        ksize as i32, 1_i32, ch as i32,
                        0_i32, batch_size as i32,
                        &mut cb,
                    );
                }
                cb.end_encoding().submit().wait_until_completed().unwrap();
            });
            results.push((format!("CausalConv1dGrouped dw up{block_idx}"), ms));
        }
    }

    // -----------------------------------------------------------------------
    // 6. HalfSnake — at each decoder stage resolution
    // -----------------------------------------------------------------------
    {
        let kernel =
            <<Metal as Backend>::Kernels as Kernels>::AudioHalfSnakeKernel::new(&context, dt)
                .unwrap();

        // Before decoder blocks: snake on decoder_dim channels at upsampled frames
        let fc_frames = SEMANTIC_FRAMES_10S * UP_STRIDES[0] * UP_STRIDES[1];
        let mut dec_frames = fc_frames;
        for block_idx in 0..4 {
            // Snake before trans_conv (at input resolution)
            let ch_in = DEC_CHANNELS[block_idx];
            {
                let label = format!(
                    "HalfSnake pre-tconv dec{block_idx} [{ch_in}ch, T={dec_frames}]"
                );
                let input = context.create_array(&[batch_size * ch_in * dec_frames], dt, "sn_in");
                let alpha = context.create_array(&[ch_in], dt, "sn_a");
                let output = context.create_array(&[batch_size * ch_in * dec_frames], dt, "sn_out");

                let ms = measure(&label, || {
                    let mut cb = context.create_command_buffer().unwrap().start_encoding();
                    {
                        let i = input.buffer(); let i = i.borrow();
                        let a = alpha.buffer(); let a = a.borrow();
                        let o = output.buffer(); let mut o = o.borrow_mut();
                        kernel.encode(
                            &*i, &*a, &mut *o,
                            ch_in as i32, dec_frames as i32, ch_in as i32,
                            0.0_f32, 1e-9_f32, batch_size as i32,
                            &mut cb,
                        );
                    }
                    cb.end_encoding().submit().wait_until_completed().unwrap();
                });
                results.push((format!("HalfSnake pre dec{block_idx}"), ms));
            }

            dec_frames *= DEC_STRIDES[block_idx];
            // Snake in residual units (at output resolution)
            let ch_out = DEC_CHANNELS[block_idx + 1];
            {
                let label = format!(
                    "HalfSnake res dec{block_idx} [{ch_out}ch, T={dec_frames}]"
                );
                let input = context.create_array(&[batch_size * ch_out * dec_frames], dt, "sn_in");
                let alpha = context.create_array(&[ch_out], dt, "sn_a");
                let output = context.create_array(&[batch_size * ch_out * dec_frames], dt, "sn_out");

                let ms = measure(&label, || {
                    let mut cb = context.create_command_buffer().unwrap().start_encoding();
                    {
                        let i = input.buffer(); let i = i.borrow();
                        let a = alpha.buffer(); let a = a.borrow();
                        let o = output.buffer(); let mut o = o.borrow_mut();
                        kernel.encode(
                            &*i, &*a, &mut *o,
                            ch_out as i32, dec_frames as i32, ch_out as i32,
                            0.0_f32, 1e-9_f32, batch_size as i32,
                            &mut cb,
                        );
                    }
                    cb.end_encoding().submit().wait_until_completed().unwrap();
                });
                results.push((format!("HalfSnake res dec{block_idx}"), ms));
            }
        }
    }

    // -----------------------------------------------------------------------
    // 7. NormNcs and Conv1d pointwise — ConvNeXt blocks in upsampler
    // -----------------------------------------------------------------------
    {
        let k_norm =
            <<Metal as Backend>::Kernels as Kernels>::AudioNormNcsKernel::new(&context, dt).unwrap();
        let k_conv1d =
            <<Metal as Backend>::Kernels as Kernels>::AudioConv1dKernel::new(&context, dt).unwrap();
        let k_act =
            <<Metal as Backend>::Kernels as Kernels>::ActivationKernel::new(&context, dt, false).unwrap();

        let mut frames = SEMANTIC_FRAMES_10S;
        for block_idx in 0..2 {
            frames *= UP_STRIDES[block_idx];
            let ch = UP_CHANNELS[block_idx + 1];

            // Norm
            {
                let label = format!("NormNcs up{block_idx} [{ch}ch, T={frames}]");
                let input = context.create_array(&[batch_size * ch * frames], dt, "norm_in");
                let scales = context.create_array(&[ch], dt, "norm_s");
                let bias = context.create_array(&[ch], dt, "norm_b");
                let output = context.create_array(&[batch_size * ch * frames], dt, "norm_out");
                let lengths = make_lengths(&context, batch_size, frames as i32);

                let ms = measure(&label, || {
                    let mut cb = context.create_command_buffer().unwrap().start_encoding();
                    {
                        let i = input.buffer(); let i = i.borrow();
                        let s = scales.buffer(); let s = s.borrow();
                        let b = bias.buffer(); let b = b.borrow();
                        let o = output.buffer(); let mut o = o.borrow_mut();
                        let l = lengths.buffer(); let l = l.borrow();
                        k_norm.encode(
                            &*i, &*s, &*b, &mut *o, &*l,
                            ch as i32, frames as i32, 1e-6_f32, 1_i32, batch_size as i32,
                            &mut cb,
                        );
                    }
                    cb.end_encoding().submit().wait_until_completed().unwrap();
                });
                results.push((format!("NormNcs up{block_idx}"), ms));
            }

            // Pointwise conv1 (ch -> ch)
            {
                let label = format!("Conv1d pw up{block_idx} [{ch}->{ch}, k=1, T={frames}]");
                let input = context.create_array(&[batch_size * ch * frames], dt, "pw_in");
                let weight = context.create_array(&[ch * ch], dt, "pw_w");
                let bias = context.create_array(&[ch], dt, "pw_b");
                let output = context.create_array(&[batch_size * ch * frames], dt, "pw_out");
                let lengths = make_lengths(&context, batch_size, frames as i32);

                let ms = measure(&label, || {
                    let mut cb = context.create_command_buffer().unwrap().start_encoding();
                    {
                        let i = input.buffer(); let i = i.borrow();
                        let w = weight.buffer(); let w = w.borrow();
                        let b = bias.buffer(); let b = b.borrow();
                        let o = output.buffer(); let mut o = o.borrow_mut();
                        let l = lengths.buffer(); let l = l.borrow();
                        k_conv1d.encode(
                            &*i, &*w, &*b, &mut *o, &*l,
                            ch as i32, ch as i32,
                            frames as i32, frames as i32,
                            1_i32, 1_i32, 1_i32, 0_i32, 0_i32,
                            batch_size as i32, &mut cb,
                        );
                    }
                    cb.end_encoding().submit().wait_until_completed().unwrap();
                });
                results.push((format!("Conv1d pw up{block_idx}"), ms));
            }

            // GELU
            {
                let n = ch * frames;
                let label = format!("GELU up{block_idx} [n={n}]");
                let input = context.create_array(&[n], dt, "gelu_in");
                let output = context.create_array(&[n], dt, "gelu_out");

                let ms = measure(&label, || {
                    let mut cb = context.create_command_buffer().unwrap().start_encoding();
                    {
                        let i = input.buffer(); let i = i.borrow();
                        let o = output.buffer(); let mut o = o.borrow_mut();
                        k_act.encode(Some(&*i), &mut *o, n as u32, 1_u32, &mut cb);
                    }
                    cb.end_encoding().submit().wait_until_completed().unwrap();
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
    println!(
        "  {:<65} {:>8}",
        "Full Pipeline Simulation (single command buffer)", "Mean (ms)"
    );
    println!("  {:-<65} {:-<8}", "", "");

    // Build all kernels
    let k_quant =
        <<Metal as Backend>::Kernels as Kernels>::AudioQuantizerDecodeKernel::new(&context, dt).unwrap();
    let k_tconv =
        <<Metal as Backend>::Kernels as Kernels>::AudioCausalConvTranspose1dCausalPadKernel::new(&context, dt).unwrap();
    let k_norm =
        <<Metal as Backend>::Kernels as Kernels>::AudioNormNcsKernel::new(&context, dt).unwrap();
    let k_conv1d =
        <<Metal as Backend>::Kernels as Kernels>::AudioConv1dKernel::new(&context, dt).unwrap();
    let k_act =
        <<Metal as Backend>::Kernels as Kernels>::ActivationKernel::new(&context, dt, false).unwrap();
    let k_add =
        <<Metal as Backend>::Kernels as Kernels>::AudioAddKernel::new(&context, dt).unwrap();
    let k_snake =
        <<Metal as Backend>::Kernels as Kernels>::AudioHalfSnakeKernel::new(&context, dt).unwrap();
    let k_conv_grp =
        <<Metal as Backend>::Kernels as Kernels>::AudioCausalConv1dGroupedKernel::new(&context, dt).unwrap();
    let k_conv1d_causal =
        <<Metal as Backend>::Kernels as Kernels>::AudioCausalConv1dKernel::new(&context, dt).unwrap();
    let k_conv_res =
        <<Metal as Backend>::Kernels as Kernels>::AudioCausalConv1dGroupedResidualKernel::new(&context, dt).unwrap();

    let frames = SEMANTIC_FRAMES_10S;

    // Quantizer arrays
    let p_q_tok = context.create_array(
        &[batch_size * TOTAL_CODEBOOKS * frames],
        DataType::U32, "pipe_q_tok",
    );
    let p_q_len = make_lengths(&context, batch_size, frames as i32);
    let p_q_sem_cb = context.create_array(&[SEMANTIC_CARDINALITY * CODEBOOK_DIM], dt, "p_sem_cb");
    let p_q_sem_proj = context.create_array(&[INPUT_DIM * CODEBOOK_DIM], dt, "p_sem_proj");
    let p_q_sem_bias = context.create_array(&[INPUT_DIM], dt, "p_sem_bias");
    let p_q_res_cbs = context.create_array(
        &[RESIDUAL_QUANTIZERS * RESIDUAL_CARDINALITY * CODEBOOK_DIM], dt, "p_res_cbs",
    );
    let p_q_res_proj = context.create_array(
        &[RESIDUAL_QUANTIZERS * INPUT_DIM * CODEBOOK_DIM], dt, "p_res_proj",
    );
    let p_q_res_bias = context.create_array(&[RESIDUAL_QUANTIZERS * INPUT_DIM], dt, "p_res_bias");

    // Compute frame counts at each stage
    let f_up0 = frames * UP_STRIDES[0];                       // 430
    let f_up1 = f_up0 * UP_STRIDES[1];                        // 860
    let f_dec0 = f_up1 * DEC_STRIDES[0];                      // 6880
    let f_dec1 = f_dec0 * DEC_STRIDES[1];                     // 55040
    let f_dec2 = f_dec1 * DEC_STRIDES[2];                     // 220160
    let f_dec3 = f_dec2 * DEC_STRIDES[3];                     // 440320

    // Pipeline scratch: large enough for the biggest intermediate
    // Biggest is decoder_dim * f_up1 or ch * f_dec stages
    let max_elems: usize = batch_size * DECODER_DIM * f_dec3.max(f_up1);
    let scratch_a = context.create_array(&[max_elems], dt, "pipe_sa");
    let scratch_b = context.create_array(&[max_elems], dt, "pipe_sb");
    // Third scratch buffer for residual connections (avoids RefCell borrow conflict)
    let scratch_r = context.create_array(&[max_elems], dt, "pipe_sr");

    // Upsample block 0: 1024->512, stride=2
    let u0_w = context.create_array(&[UP_CHANNELS[0] * UP_CHANNELS[1] * (UP_STRIDES[0] * 2)], dt, "u0_w");
    let u0_b = context.create_array(&[UP_CHANNELS[1]], dt, "u0_b");
    // ConvNeXt 0 (ch=512)
    let c0_ch = UP_CHANNELS[1];
    let c0_dw_w = context.create_array(&[c0_ch * 1 * 7], dt, "c0_dw_w");
    let c0_dw_b = context.create_array(&[c0_ch], dt, "c0_dw_b");
    let c0_ns = context.create_array(&[c0_ch], dt, "c0_ns");
    let c0_nb = context.create_array(&[c0_ch], dt, "c0_nb");
    let c0_p1_w = context.create_array(&[c0_ch * c0_ch], dt, "c0_p1_w");
    let c0_p1_b = context.create_array(&[c0_ch], dt, "c0_p1_b");
    let c0_p2_w = context.create_array(&[c0_ch * c0_ch], dt, "c0_p2_w");
    let c0_p2_b = context.create_array(&[c0_ch], dt, "c0_p2_b");

    // Upsample block 1: 512->256, stride=2
    let u1_w = context.create_array(&[UP_CHANNELS[1] * UP_CHANNELS[2] * (UP_STRIDES[1] * 2)], dt, "u1_w");
    let u1_b = context.create_array(&[UP_CHANNELS[2]], dt, "u1_b");
    // ConvNeXt 1 (ch=256)
    let c1_ch = UP_CHANNELS[2];
    let c1_dw_w = context.create_array(&[c1_ch * 1 * 7], dt, "c1_dw_w");
    let c1_dw_b = context.create_array(&[c1_ch], dt, "c1_dw_b");
    let c1_ns = context.create_array(&[c1_ch], dt, "c1_ns");
    let c1_nb = context.create_array(&[c1_ch], dt, "c1_nb");
    let c1_p1_w = context.create_array(&[c1_ch * c1_ch], dt, "c1_p1_w");
    let c1_p1_b = context.create_array(&[c1_ch], dt, "c1_p1_b");
    let c1_p2_w = context.create_array(&[c1_ch * c1_ch], dt, "c1_p2_w");
    let c1_p2_b = context.create_array(&[c1_ch], dt, "c1_p2_b");

    // first_conv: 256->1536, k=7
    let fc_w = context.create_array(&[DECODER_DIM * UP_CHANNELS[2] * 7], dt, "fc_w");
    let fc_b = context.create_array(&[DECODER_DIM], dt, "fc_b");

    // Decoder block weights for all 4 blocks
    struct DecBlockWeights {
        snake_a: uzu::array::Array<Metal>,
        tconv_w: uzu::array::Array<Metal>,
        tconv_b: uzu::array::Array<Metal>,
        // 3 residual units, each with snake1, conv1, snake2, conv2
        ru_s1: [uzu::array::Array<Metal>; 3],
        ru_c1_w: [uzu::array::Array<Metal>; 3],
        ru_c1_b: [uzu::array::Array<Metal>; 3],
        ru_s2: [uzu::array::Array<Metal>; 3],
        ru_c2_w: [uzu::array::Array<Metal>; 3],
        ru_c2_b: [uzu::array::Array<Metal>; 3],
    }

    let dec_blocks: Vec<DecBlockWeights> = (0..4).map(|i| {
        let cin = DEC_CHANNELS[i];
        let cout = DEC_CHANNELS[i + 1];
        let stride = DEC_STRIDES[i];
        let ksize = stride * 2;
        DecBlockWeights {
            snake_a: context.create_array(&[cin], dt, &format!("d{i}_sa")),
            tconv_w: context.create_array(&[cin * cout * ksize], dt, &format!("d{i}_tw")),
            tconv_b: context.create_array(&[cout], dt, &format!("d{i}_tb")),
            ru_s1: std::array::from_fn(|j| context.create_array(&[cout], dt, &format!("d{i}_r{j}_s1"))),
            ru_c1_w: std::array::from_fn(|j| context.create_array(&[cout * cout * 7], dt, &format!("d{i}_r{j}_c1w"))),
            ru_c1_b: std::array::from_fn(|j| context.create_array(&[cout], dt, &format!("d{i}_r{j}_c1b"))),
            ru_s2: std::array::from_fn(|j| context.create_array(&[cout], dt, &format!("d{i}_r{j}_s2"))),
            ru_c2_w: std::array::from_fn(|j| context.create_array(&[cout * cout * 7], dt, &format!("d{i}_r{j}_c2w"))),
            ru_c2_b: std::array::from_fn(|j| context.create_array(&[cout], dt, &format!("d{i}_r{j}_c2b"))),
        }
    }).collect();

    // final snake + conv + tanh
    let final_snake_a = context.create_array(&[DEC_CHANNELS[4]], dt, "final_sa");
    let final_conv_w = context.create_array(&[1 * DEC_CHANNELS[4] * 7], dt, "final_cw");
    let final_conv_b = context.create_array(&[1], dt, "final_cb");

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

    let ms = measure(
        "Full pipeline (quant -> 2 up -> first_conv -> 4 dec -> final)",
        || {
            let mut cb = context.create_command_buffer().unwrap().start_encoding();
            let sa = scratch_a.buffer();
            let sb = scratch_b.buffer();

            // --- Quantizer: output [B, T, input_dim] (NSC) ---
            {
                let tok = p_q_tok.buffer(); let tok = tok.borrow();
                let len = p_q_len.buffer(); let len = len.borrow();
                let scb = p_q_sem_cb.buffer(); let scb = scb.borrow();
                let sp = p_q_sem_proj.buffer(); let sp = sp.borrow();
                let sbi = p_q_sem_bias.buffer(); let sbi = sbi.borrow();
                let rcb = p_q_res_cbs.buffer(); let rcb = rcb.borrow();
                let rp = p_q_res_proj.buffer(); let rp = rp.borrow();
                let rb = p_q_res_bias.buffer(); let rb = rb.borrow();
                let mut out = sa.borrow_mut();
                k_quant.encode(
                    &*tok, &*len, &*scb, &*sp, &*sbi, &*rcb, &*rp, &*rb,
                    &mut *out,
                    batch_size as i32, TOTAL_CODEBOOKS as i32,
                    frames as i32, INPUT_DIM as i32, CODEBOOK_DIM as i32,
                    RESIDUAL_QUANTIZERS as i32,
                    SEMANTIC_CARDINALITY as i32, RESIDUAL_CARDINALITY as i32,
                    &mut cb,
                );
            }

            // --- Upsample block 0: 1024->512, stride=2, NSC input ---
            {
                let i = sa.borrow();
                let w = u0_w.buffer(); let w = w.borrow();
                let b = u0_b.buffer(); let b = b.borrow();
                let mut o = sb.borrow_mut();
                let l = len_up0.buffer(); let l = l.borrow();
                k_tconv.encode(
                    &*i, &*w, &*b, &mut *o, &*l,
                    UP_CHANNELS[0] as i32, UP_CHANNELS[1] as i32,
                    frames as i32, f_up0 as i32,
                    (UP_STRIDES[0] * 2) as i32, UP_STRIDES[0] as i32,
                    1_i32, 1_i32, batch_size as i32, &mut cb,
                );
            }
            // ConvNeXt 0
            {
                let i = sb.borrow();
                let w = c0_dw_w.buffer(); let w = w.borrow();
                let b = c0_dw_b.buffer(); let b = b.borrow();
                let mut o = sa.borrow_mut();
                let l = len_up0.buffer(); let l = l.borrow();
                k_conv_grp.encode(
                    &*i, &*w, &*b, &mut *o, &*l,
                    c0_ch as i32, c0_ch as i32, f_up0 as i32,
                    7_i32, 1_i32, c0_ch as i32, 0_i32, batch_size as i32,
                    &mut cb,
                );
            }
            {
                let i = sa.borrow();
                let s = c0_ns.buffer(); let s = s.borrow();
                let b = c0_nb.buffer(); let b = b.borrow();
                let mut o = sb.borrow_mut();
                let l = len_up0.buffer(); let l = l.borrow();
                k_norm.encode(
                    &*i, &*s, &*b, &mut *o, &*l,
                    c0_ch as i32, f_up0 as i32, 1e-6_f32, 1_i32, batch_size as i32,
                    &mut cb,
                );
            }
            {
                let i = sb.borrow();
                let w = c0_p1_w.buffer(); let w = w.borrow();
                let b = c0_p1_b.buffer(); let b = b.borrow();
                let mut o = sa.borrow_mut();
                let l = len_up0.buffer(); let l = l.borrow();
                k_conv1d.encode(
                    &*i, &*w, &*b, &mut *o, &*l,
                    c0_ch as i32, c0_ch as i32,
                    f_up0 as i32, f_up0 as i32,
                    1_i32, 1_i32, 1_i32, 0_i32, 0_i32,
                    batch_size as i32, &mut cb,
                );
            }
            {
                let i = sa.borrow();
                let mut o = sb.borrow_mut();
                k_act.encode(Some(&*i), &mut *o, (c0_ch * f_up0) as u32, 1_u32, &mut cb);
            }
            {
                let i = sb.borrow();
                let w = c0_p2_w.buffer(); let w = w.borrow();
                let b = c0_p2_b.buffer(); let b = b.borrow();
                let mut o = sa.borrow_mut();
                let l = len_up0.buffer(); let l = l.borrow();
                k_conv1d.encode(
                    &*i, &*w, &*b, &mut *o, &*l,
                    c0_ch as i32, c0_ch as i32,
                    f_up0 as i32, f_up0 as i32,
                    1_i32, 1_i32, 1_i32, 0_i32, 0_i32,
                    batch_size as i32, &mut cb,
                );
            }
            {
                let a = sa.borrow();
                let mut o = sb.borrow_mut();
                k_add.encode(&*a, &*a, &mut *o, (c0_ch * f_up0) as i32, &mut cb);
            }

            // --- Upsample block 1: 512->256, stride=2, NCS input ---
            {
                let i = sb.borrow();
                let w = u1_w.buffer(); let w = w.borrow();
                let b = u1_b.buffer(); let b = b.borrow();
                let mut o = sa.borrow_mut();
                let l = len_up1.buffer(); let l = l.borrow();
                k_tconv.encode(
                    &*i, &*w, &*b, &mut *o, &*l,
                    UP_CHANNELS[1] as i32, UP_CHANNELS[2] as i32,
                    f_up0 as i32, f_up1 as i32,
                    (UP_STRIDES[1] * 2) as i32, UP_STRIDES[1] as i32,
                    1_i32, 0_i32, batch_size as i32,
                    &mut cb,
                );
            }
            // ConvNeXt 1
            {
                let i = sa.borrow();
                let w = c1_dw_w.buffer(); let w = w.borrow();
                let b = c1_dw_b.buffer(); let b = b.borrow();
                let mut o = sb.borrow_mut();
                let l = len_up1.buffer(); let l = l.borrow();
                k_conv_grp.encode(
                    &*i, &*w, &*b, &mut *o, &*l,
                    c1_ch as i32, c1_ch as i32, f_up1 as i32,
                    7_i32, 1_i32, c1_ch as i32, 0_i32, batch_size as i32,
                    &mut cb,
                );
            }
            {
                let i = sb.borrow();
                let s = c1_ns.buffer(); let s = s.borrow();
                let b = c1_nb.buffer(); let b = b.borrow();
                let mut o = sa.borrow_mut();
                let l = len_up1.buffer(); let l = l.borrow();
                k_norm.encode(
                    &*i, &*s, &*b, &mut *o, &*l,
                    c1_ch as i32, f_up1 as i32, 1e-6_f32, 1_i32, batch_size as i32,
                    &mut cb,
                );
            }
            {
                let i = sa.borrow();
                let w = c1_p1_w.buffer(); let w = w.borrow();
                let b = c1_p1_b.buffer(); let b = b.borrow();
                let mut o = sb.borrow_mut();
                let l = len_up1.buffer(); let l = l.borrow();
                k_conv1d.encode(
                    &*i, &*w, &*b, &mut *o, &*l,
                    c1_ch as i32, c1_ch as i32,
                    f_up1 as i32, f_up1 as i32,
                    1_i32, 1_i32, 1_i32, 0_i32, 0_i32,
                    batch_size as i32, &mut cb,
                );
            }
            {
                let i = sb.borrow();
                let mut o = sa.borrow_mut();
                k_act.encode(Some(&*i), &mut *o, (c1_ch * f_up1) as u32, 1_u32, &mut cb);
            }
            {
                let i = sa.borrow();
                let w = c1_p2_w.buffer(); let w = w.borrow();
                let b = c1_p2_b.buffer(); let b = b.borrow();
                let mut o = sb.borrow_mut();
                let l = len_up1.buffer(); let l = l.borrow();
                k_conv1d.encode(
                    &*i, &*w, &*b, &mut *o, &*l,
                    c1_ch as i32, c1_ch as i32,
                    f_up1 as i32, f_up1 as i32,
                    1_i32, 1_i32, 1_i32, 0_i32, 0_i32,
                    batch_size as i32, &mut cb,
                );
            }
            {
                let a = sb.borrow();
                let mut o = sa.borrow_mut();
                k_add.encode(&*a, &*a, &mut *o, (c1_ch * f_up1) as i32, &mut cb);
            }

            // --- first_conv: 256->1536, k=7 ---
            {
                let i = sa.borrow();
                let w = fc_w.buffer(); let w = w.borrow();
                let b = fc_b.buffer(); let b = b.borrow();
                let mut o = sb.borrow_mut();
                let l = len_up1.buffer(); let l = l.borrow();
                k_conv1d_causal.encode(
                    &*i, &*w, &*b, &mut *o, &*l,
                    UP_CHANNELS[2] as i32, DECODER_DIM as i32, f_up1 as i32,
                    7_i32, 1_i32, 0_i32, batch_size as i32,
                    &mut cb,
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
                    let (i_buf, o_buf) = if cur_in_sa { (&sa, &sb) } else { (&sb, &sa) };
                    let i = i_buf.borrow();
                    let a = blk.snake_a.buffer(); let a = a.borrow();
                    let mut o = o_buf.borrow_mut();
                    k_snake.encode(
                        &*i, &*a, &mut *o,
                        cin as i32, cur_frames as i32, cin as i32,
                        0.0_f32, 1e-9_f32, batch_size as i32,
                        &mut cb,
                    );
                }
                cur_in_sa = !cur_in_sa;

                // trans_conv
                {
                    let (i_buf, o_buf) = if cur_in_sa { (&sa, &sb) } else { (&sb, &sa) };
                    let i = i_buf.borrow();
                    let w = blk.tconv_w.buffer(); let w = w.borrow();
                    let b = blk.tconv_b.buffer(); let b = b.borrow();
                    let mut o = o_buf.borrow_mut();
                    let l = dec_len.buffer(); let l = l.borrow();
                    k_tconv.encode(
                        &*i, &*w, &*b, &mut *o, &*l,
                        cin as i32, cout as i32,
                        cur_frames as i32, next_frames as i32,
                        ksize as i32, stride as i32,
                        1_i32, 0_i32, batch_size as i32,
                        &mut cb,
                    );
                }
                cur_in_sa = !cur_in_sa;
                cur_frames = next_frames;

                // 3 residual units
                let sr = scratch_r.buffer();
                for ru_idx in 0..3 {
                    // snake1
                    {
                        let (i_buf, o_buf) = if cur_in_sa { (&sa, &sb) } else { (&sb, &sa) };
                        let i = i_buf.borrow();
                        let a = blk.ru_s1[ru_idx].buffer(); let a = a.borrow();
                        let mut o = o_buf.borrow_mut();
                        k_snake.encode(
                            &*i, &*a, &mut *o,
                            cout as i32, cur_frames as i32, cout as i32,
                            0.0_f32, 1e-9_f32, batch_size as i32,
                            &mut cb,
                        );
                    }
                    cur_in_sa = !cur_in_sa;

                    // conv1 (dilation varies: 1, 3, 9)
                    {
                        let (i_buf, o_buf) = if cur_in_sa { (&sa, &sb) } else { (&sb, &sa) };
                        let i = i_buf.borrow();
                        let w = blk.ru_c1_w[ru_idx].buffer(); let w = w.borrow();
                        let b = blk.ru_c1_b[ru_idx].buffer(); let b = b.borrow();
                        let mut o = o_buf.borrow_mut();
                        let l = dec_len.buffer(); let l = l.borrow();
                        k_conv1d_causal.encode(
                            &*i, &*w, &*b, &mut *o, &*l,
                            cout as i32, cout as i32, cur_frames as i32,
                            7_i32, dilations[ru_idx], 0_i32, batch_size as i32,
                            &mut cb,
                        );
                    }
                    cur_in_sa = !cur_in_sa;

                    // snake2
                    {
                        let (i_buf, o_buf) = if cur_in_sa { (&sa, &sb) } else { (&sb, &sa) };
                        let i = i_buf.borrow();
                        let a = blk.ru_s2[ru_idx].buffer(); let a = a.borrow();
                        let mut o = o_buf.borrow_mut();
                        k_snake.encode(
                            &*i, &*a, &mut *o,
                            cout as i32, cur_frames as i32, cout as i32,
                            0.0_f32, 1e-9_f32, batch_size as i32,
                            &mut cb,
                        );
                    }
                    cur_in_sa = !cur_in_sa;

                    // conv2 + residual add
                    // Use scratch_r as the residual source (separate buffer to
                    // avoid RefCell borrow conflict with the output destination).
                    {
                        let (i_buf, o_buf) = if cur_in_sa { (&sa, &sb) } else { (&sb, &sa) };
                        let i = i_buf.borrow();
                        let r = sr.borrow();
                        let w = blk.ru_c2_w[ru_idx].buffer(); let w = w.borrow();
                        let b = blk.ru_c2_b[ru_idx].buffer(); let b = b.borrow();
                        let mut o = o_buf.borrow_mut();
                        let l = dec_len.buffer(); let l = l.borrow();
                        k_conv_res.encode(
                            &*i, &*r, &*w, &*b, &mut *o, &*l,
                            cout as i32, cout as i32, cur_frames as i32,
                            7_i32, 1_i32, 1_i32, batch_size as i32,
                            &mut cb,
                        );
                    }
                    cur_in_sa = !cur_in_sa;
                }
            }

            // --- final snake + conv + tanh ---
            {
                let (i_buf, o_buf) = if cur_in_sa { (&sa, &sb) } else { (&sb, &sa) };
                let i = i_buf.borrow();
                let a = final_snake_a.buffer(); let a = a.borrow();
                let mut o = o_buf.borrow_mut();
                k_snake.encode(
                    &*i, &*a, &mut *o,
                    DEC_CHANNELS[4] as i32, f_dec3 as i32, DEC_CHANNELS[4] as i32,
                    0.0_f32, 1e-9_f32, batch_size as i32,
                    &mut cb,
                );
            }
            cur_in_sa = !cur_in_sa;
            {
                let (i_buf, o_buf) = if cur_in_sa { (&sa, &sb) } else { (&sb, &sa) };
                let i = i_buf.borrow();
                let w = final_conv_w.buffer(); let w = w.borrow();
                let b = final_conv_b.buffer(); let b = b.borrow();
                let mut o = o_buf.borrow_mut();
                let l = len_dec3.buffer(); let l = l.borrow();
                k_conv1d_causal.encode(
                    &*i, &*w, &*b, &mut *o, &*l,
                    DEC_CHANNELS[4] as i32, 1_i32, f_dec3 as i32,
                    7_i32, 1_i32, 0_i32, batch_size as i32,
                    &mut cb,
                );
            }
            cur_in_sa = !cur_in_sa;
            // tanh (activation id=2)
            {
                let (i_buf, o_buf) = if cur_in_sa { (&sa, &sb) } else { (&sb, &sa) };
                let i = i_buf.borrow();
                let mut o = o_buf.borrow_mut();
                k_act.encode(Some(&*i), &mut *o, f_dec3 as u32, 2_u32, &mut cb);
            }

            cb.end_encoding().submit().wait_until_completed().unwrap();
        },
    );
    results.push(("Full pipeline".into(), ms));

    // =======================================================================
    // Summary
    // =======================================================================
    println!();
    println!("  === SUMMARY ===");
    println!(
        "  {:<65} {:>8}",
        "Kernel", "Mean (ms)"
    );
    println!("  {:-<65} {:-<8}", "", "");
    let mut total_individual = 0.0;
    for (name, ms) in &results {
        println!("  {name:<65} {ms:>8.3}");
        if name != "Full pipeline" {
            total_individual += ms;
        }
    }
    println!("  {:-<65} {:-<8}", "", "");
    println!(
        "  {:<65} {:>8.3}",
        "Sum of individual kernels", total_individual
    );
    if let Some((_, pipeline_ms)) = results.iter().find(|(n, _)| n == "Full pipeline") {
        println!(
            "  {:<65} {:>8.3}",
            "Full pipeline (single CB)", pipeline_ms
        );
    }
    println!();
}
