use half::f16;
use metal::MTLResourceOptions;
use uzu::{
    DataType,
    backends::metal::{MTLContext, MatmulArguments, MatmulKernel},
};

fn make_context() -> MTLContext {
    let device = metal::Device::system_default().expect("metal device");
    let queue = device.new_command_queue();
    MTLContext::new(device, queue).expect("ctx")
}

#[test]
#[ignore = "Requires Apple Silicon + Metal runtime"]
fn gemv_single_token_runs() {
    let ctx = make_context();

    // One token (M=1), K=4, N=4
    let a_vals: [f16; 4] = [1.0, 2.0, 3.0, 4.0].map(f16::from_f32);
    let b_vals: [f16; 16] = (0..16)
        .map(|i| f16::from_f32((i as f32) * 0.1))
        .collect::<Vec<_>>()
        .try_into()
        .unwrap();

    let a_buf = ctx.device.new_buffer_with_data(
        a_vals.as_ptr() as *const _,
        (a_vals.len() * std::mem::size_of::<f16>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let b_buf = ctx.device.new_buffer_with_data(
        b_vals.as_ptr() as *const _,
        (b_vals.len() * std::mem::size_of::<f16>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let d_buf = ctx.device.new_buffer(
        (4 * std::mem::size_of::<f16>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let args = MatmulArguments {
        a: &a_buf,
        b: &b_buf,
        d: &d_buf,
        batch: 1,
        input_dim: 4,
        output_dim: 4,
        lda: 4,
        ldb: 4,
        ldd: 4,
        batch_count: 1,
    };

    let mut kernel =
        MatmulKernel::new(&ctx, DataType::F16, false, true).unwrap();
    let command_buffer = ctx.command_queue.new_command_buffer().to_owned();
    let encoder = command_buffer.new_compute_command_encoder();
    kernel.encode(&ctx, &encoder, args).unwrap();
    encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();

    // Read back output
    let out_ptr = d_buf.contents() as *const f16;
    let out = unsafe { std::slice::from_raw_parts(out_ptr, 4) };
    let cpu: f32 = a_vals
        .iter()
        .zip(b_vals.chunks_exact(4))
        .map(|(a, row)| a.to_f32() * row[0].to_f32())
        .sum();
    assert!((out[0].to_f32() - cpu).abs() < 1e-2);
}
