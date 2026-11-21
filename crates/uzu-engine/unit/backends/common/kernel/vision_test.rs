use half::bf16;
use uzu_engine_macros::uzu_test;

use crate::{
    backends::common::{
        Backend, Context, Kernels,
        kernel::{VisionPatchifyKernel, VisionPositionEmbeddingKernel},
    },
    data_type::DataType,
    tests::helpers::{
        buffer_to_vec, create_buffer, create_buffer_with_data, create_context, for_each_backend, submit_command_buffer,
    },
};

fn check_kernels<B: Backend>() {
    let ctx = create_context::<B>();
    let mut commands = ctx.create_command_buffer(None, None).unwrap();
    // The 8x4 image has two side-by-side merge blocks of four 2x2 patches each.
    let pixels: Vec<u8> = (0..96).collect();
    let mean = [0.5, 0.25, 0.0];
    let std = [0.25, 0.5, 1.0];
    let rgb = create_buffer_with_data::<B, _>(&ctx, &pixels);
    let means = create_buffer_with_data::<B, _>(&ctx, &mean);
    let stds = create_buffer_with_data::<B, _>(&ctx, &std);
    let mut patches = create_buffer::<B, bf16>(&ctx, 192);
    let patchify = <B::Kernels as Kernels>::VisionPatchifyKernel::new(&ctx, DataType::BF16).unwrap();
    patchify.encode(&rgb, &mut patches, &means, &stds, 8, 2, 2, 2, 192, &mut commands);

    // The kernels take four channels at a time, so the model dim is a multiple of 4.
    // The 3x3 table is linear in the position and the channel, so its bilinear resize is linear too.
    const MODEL_DIM: usize = 4;
    let table: Vec<_> =
        (0..9 * MODEL_DIM).map(|i| bf16::from_f32((i / MODEL_DIM) as f32 + 0.25 * (i % MODEL_DIM) as f32)).collect();
    let table = create_buffer_with_data::<B, _>(&ctx, &table);
    let mut stream = create_buffer_with_data::<B, _>(&ctx, &[bf16::ONE; 8 * MODEL_DIM]);
    let pos = <B::Kernels as Kernels>::VisionPositionEmbeddingKernel::new(&ctx, DataType::BF16).unwrap();
    pos.encode(&table, &mut stream, 3, 4, 2, 2, MODEL_DIM as u32, (8 * MODEL_DIM) as u32, &mut commands);
    submit_command_buffer(commands);

    let patch_order = [(0, 0), (1, 0), (0, 1), (1, 1), (2, 0), (3, 0), (2, 1), (3, 1)];
    let mut expected = Vec::new();
    for (px, py) in patch_order {
        for c in 0..3 {
            for _frame in 0..2 {
                for y in py * 2..py * 2 + 2 {
                    for x in px * 2..px * 2 + 2 {
                        let sample = f32::from(pixels[(y * 8 + x) * 3 + c]) / 255.0;
                        expected.push(bf16::from_f32((sample - mean[c]) / std[c]));
                    }
                }
            }
        }
    }
    assert_eq!(buffer_to_vec::<B, bf16>(&ctx, &patches), expected);
    // Resize the 3x3 table to 4x2: the four columns land on x = 0, 2/3, 4/3, 2 and the two rows on y = 0, 2.
    // The sum is rounded to BF16 once, when it is added to the stream.
    let mut expected_positions = Vec::new();
    for (px, py) in patch_order {
        let x = px as f32 * 2.0 / 3.0;
        let y = py as f32 * 2.0;
        for channel in 0..MODEL_DIM {
            expected_positions.push(bf16::from_f32(1.0 + 3.0 * y + x + 0.25 * channel as f32));
        }
    }
    assert_eq!(buffer_to_vec::<B, bf16>(&ctx, &stream), expected_positions);
}

#[uzu_test]
fn vision_kernels() {
    for_each_backend!(|B| check_kernels::<B>());
}
