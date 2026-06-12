use crate::tests::env_vars;

pub fn enable_benchmark_gpu_capture_if_requested() {
    if env_vars::enabled(env_vars::UZU_CAPTURE_BENCH) {
        unsafe {
            std::env::set_var(env_vars::METAL_CAPTURE_ENABLED, "1");
        }
    }
}
