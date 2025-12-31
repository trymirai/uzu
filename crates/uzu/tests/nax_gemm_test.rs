use half::bf16;
use metal::MTLResourceOptions;
use uzu::{
    DataType,
    backends::metal::{
        MTLContext,
        kernel::{MatmulArguments, MatmulKernel},
    },
};

fn make_context() -> MTLContext {
    let device = metal::Device::system_default().expect("metal device");
    let queue = device.new_command_queue();
    MTLContext::new(device, queue).expect("ctx")
}

#[test]
#[ignore = "Requires Apple Silicon + Metal runtime"]
fn nax_path_runs_when_available() {
    let ctx = make_context();

    let m = 64usize;
    let n = 128usize;
    let k = 256usize;

    let a_vals: Vec<bf16> =
        (0..(m * k)).map(|i| bf16::from_f32((i % 11) as f32 * 0.01)).collect();
    let b_vals: Vec<bf16> = (0..(k * n))
        .map(|i| bf16::from_f32((i % 13) as f32 * 0.02 - 0.05))
        .collect();

    let a_buf = ctx.device.new_buffer_with_data(
        a_vals.as_ptr() as *const _,
        (a_vals.len() * std::mem::size_of::<bf16>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let b_buf = ctx.device.new_buffer_with_data(
        b_vals.as_ptr() as *const _,
        (b_vals.len() * std::mem::size_of::<bf16>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let d_buf = ctx.device.new_buffer(
        (m * n * std::mem::size_of::<bf16>()) as u64,
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
        MatmulKernel::new(&ctx, DataType::BF16, false, true).unwrap();
    let command_buffer = ctx.command_queue.new_command_buffer().to_owned();
    let encoder = command_buffer.new_compute_command_encoder();
    kernel.encode(&ctx, &encoder, args).unwrap();
    encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();

    let out_ptr = d_buf.contents() as *const bf16;
    let out = unsafe { std::slice::from_raw_parts(out_ptr, m * n) };
    assert!(!out[0].to_f32().is_nan());
}
