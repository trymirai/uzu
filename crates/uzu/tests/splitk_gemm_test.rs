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
fn splitk_triggers_for_large_k() {
    let ctx = make_context();

    // M=32, N=32, K large to trigger split-k heuristic.
    let m = 32usize;
    let n = 32usize;
    let k = 512usize;

    let a_vals: Vec<f16> =
        (0..(m * k)).map(|i| f16::from_f32((i % 17) as f32 * 0.01)).collect();
    let b_vals: Vec<f16> = (0..(k * n))
        .map(|i| f16::from_f32((i % 23) as f32 * 0.02 - 0.1))
        .collect();

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
        (m * n * std::mem::size_of::<f16>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let args = MatmulArguments {
        a: &a_buf,
        b: &b_buf,
        d: &d_buf,
        batch: m as i32,
        input_dim: k as i32,
        output_dim: n as i32,
        lda: k as i32,
        ldb: n as i32,
        ldd: n as i32,
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

    // Basic sanity: output buffer was written (check first element non-NaN)
    let out_ptr = d_buf.contents() as *const f16;
    let out = unsafe { std::slice::from_raw_parts(out_ptr, m * n) };
    assert!(!out[0].to_f32().is_nan());
}
