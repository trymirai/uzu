use tokio::runtime::Runtime;

use crate::{server::main::run_server, speculator_args::SpeculatorArgs};

pub fn handle_serve(
    model_path: String,
    prefill_step_size: Option<usize>,
    speculator_args: SpeculatorArgs,
) {
    let runtime = Runtime::new().unwrap();
    runtime.block_on(run_server(model_path, prefill_step_size, speculator_args));
}
