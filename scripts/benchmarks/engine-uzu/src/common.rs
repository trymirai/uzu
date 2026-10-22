use std::{
    io::{self, BufRead, Write},
    os::fd::{AsFd, AsRawFd, OwnedFd, RawFd},
};

use crate::{
    bench::{BenchRequest, BenchResponse},
    engine::UzuEngine,
};

pub trait InferenceEngine {
    async fn execute(
        &mut self,
        request: &BenchRequest,
    ) -> anyhow::Result<Vec<BenchResponse>>;
}

async fn execute(
    engine: &mut UzuEngine,
    req_str: &str,
) -> anyhow::Result<String> {
    let request = serde_json::from_str::<BenchRequest>(req_str)?;
    let mut redirect = StdoutRedirect::to_stderr()?;
    let responses = engine.execute(&request).await;
    redirect.restore()?;
    Ok(serde_json::to_string(&responses?)?)
}

pub async fn run_loop(engine: &mut UzuEngine) -> anyhow::Result<()> {
    for line in io::stdin().lock().lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }

        match execute(engine, &line).await {
            Ok(response_json) => {
                let mut output = io::stdout().lock();
                writeln!(output, "{response_json}")?;
                output.flush()?;
            },
            Err(error) => {
                let mut error_output = io::stderr().lock();
                writeln!(error_output, "Failed to process request: {error}")?;
                error_output.flush()?;
            },
        }
    }

    Ok(())
}

// Redirect the file descriptor so engine output cannot corrupt the JSON protocol.
pub(crate) struct StdoutRedirect {
    original: Option<OwnedFd>,
}

impl StdoutRedirect {
    pub(crate) fn to_stderr() -> io::Result<Self> {
        io::stdout().flush()?;
        let original = io::stdout().as_fd().try_clone_to_owned()?;
        Self::redirect_to(io::stderr().as_raw_fd())?;
        Ok(Self {
            original: Some(original),
        })
    }

    fn redirect_to(fd: RawFd) -> io::Result<()> {
        loop {
            // SAFETY: dup2 only operates on file descriptors; the caller keeps fd open.
            if unsafe { libc::dup2(fd, libc::STDOUT_FILENO) } >= 0 {
                return Ok(());
            }
            let error = io::Error::last_os_error();
            if error.kind() != io::ErrorKind::Interrupted {
                return Err(error);
            }
        }
    }

    pub(crate) fn restore(&mut self) -> io::Result<()> {
        if let Some(original) = &self.original {
            // Restore stdout even if flushing engine output fails.
            let flushed = io::stdout().flush();
            Self::redirect_to(original.as_raw_fd())?;
            self.original = None;
            flushed?;
        }
        Ok(())
    }
}

impl Drop for StdoutRedirect {
    fn drop(&mut self) {
        let _ = self.restore();
    }
}
