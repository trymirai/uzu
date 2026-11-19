use std::{
    ffi::{CStr, c_char, c_int},
    fs,
};

use anyhow::Error;
use clap::Parser;
use tokio::runtime::Runtime;

use crate::{
    args::{Args, ModelConfig},
    common::{StdoutRedirect, run_loop},
    engine::UzuEngine,
};

pub async fn run(args: Args) -> anyhow::Result<()> {
    let config: ModelConfig = serde_json::from_slice(&fs::read(&args.model)?)?;
    let mut redirect = StdoutRedirect::to_stderr()?;
    let mut engine = UzuEngine::new(&config).await?;
    redirect.restore()?;
    run_loop(&mut engine).await
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn benchmarks_uzu_main(
    argc: c_int,
    argv: *const *const c_char,
) -> c_int {
    // SAFETY: callers pass argc valid NUL-terminated strings, as for a C `main`.
    let arguments =
        (0..argc as usize).map(|index| unsafe { CStr::from_ptr(*argv.add(index)) }.to_string_lossy().into_owned());
    let args = match Args::try_parse_from(arguments) {
        Ok(args) => args,
        Err(error) => {
            let _ = error.print();
            return error.exit_code();
        },
    };
    let result = Runtime::new().map_err(Error::from).and_then(|runtime| runtime.block_on(run(args)));
    match result {
        Ok(()) => 0,
        Err(error) => {
            eprintln!("Error: {error:?}");
            1
        },
    }
}
