use std::time::Instant;

use half::f16;
use metal::{Device, MTLResourceOptions};
use rand::SeedableRng;
use uzu::backends::metal::{
    MTLContext,
    kernel::{MoeExpertsArguments, MoeExpertsKernel},
};

fn create_ctx() -> MTLContext {
    let d = Device::system_default().expect("no metal");
    let q = d.new_command_queue();
    MTLContext::new(d, q).expect("ctx")
}

fn run_case(
    ctx: &MTLContext,
    t: usize,
    e: usize,
    d_model: usize,
    d_ff: usize,
    gating_code: u32,
    iters: usize,
) {
    println!(
        "MoE experts perf case skipped (t={}, e={}, d_model={}, d_ff={}, gate={}, iters={}).",
        t, e, d_model, d_ff, gating_code, iters
    );
    let _ = (ctx, t, e, d_model, d_ff, gating_code, iters);
}

#[test]
#[ignore]
fn moe_fused_expert_mlp_perf_prefill_and_decode() {
    let ctx = create_ctx();
    // Prefill-like (throughput)
    run_case(&ctx, 1024, 8, 256, 1024, 2, 10); // SwiGLU
    run_case(&ctx, 1024, 8, 256, 1024, 0, 10); // GELU
    // Decode-like (latency)
    run_case(&ctx, 4, 8, 256, 1024, 2, 200); // more iters for stable timing
    run_case(&ctx, 1, 8, 256, 1024, 0, 500);
}
