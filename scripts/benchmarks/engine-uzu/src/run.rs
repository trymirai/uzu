use std::ffi::{CStr, c_char, c_int};

use clap::Parser;

use crate::{
    args::Args,
    common::{StdoutRedirect, run_loop},
    engine::UzuEngine,
};

pub async fn run(args: Args) -> anyhow::Result<()> {
    let mut redirect = StdoutRedirect::to_stderr()?;
    let mut engine = UzuEngine::new(&args.model).await?;
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
    let result =
        tokio::runtime::Runtime::new().map_err(anyhow::Error::from).and_then(|runtime| runtime.block_on(run(args)));
    match result {
        Ok(()) => 0,
        Err(error) => {
            eprintln!("Error: {error:?}");
            1
        },
    }
}
