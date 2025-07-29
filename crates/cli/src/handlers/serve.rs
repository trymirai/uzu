use tokio::runtime::Runtime;

use crate::server::main::run_server;

pub fn handle_serve(model_path: String) {
    let runtime = Runtime::new().unwrap();
    runtime.block_on(run_server(model_path));
}
