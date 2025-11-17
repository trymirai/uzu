use tokio::runtime::Runtime;

use crate::server::main::run_server;

pub fn handle_serve(
    model_path: String,
    prefill_step_size: Option<usize>,
) {
    let runtime = Runtime::new().unwrap();
    runtime.block_on(run_server(model_path, prefill_step_size));
}
