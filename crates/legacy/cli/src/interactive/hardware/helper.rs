use std::{
    io::BufReader,
    net::Shutdown,
    os::unix::net::UnixStream,
    path::Path,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::Duration,
};

use tokio::signal::unix::{SignalKind, signal};

use super::{HardwareControls, HardwareError, PerformanceMode, peer_identity, read_message, write_message};

pub async fn run_helper(
    socket: &Path,
    client_uid: u32,
    client_pid: i32,
) -> Result<(), HardwareError> {
    if unsafe { libc::geteuid() } != 0 {
        return Err(HardwareError::Protocol("helper requires administrator privileges"));
    }
    let stream = UnixStream::connect(socket)?;
    if peer_identity(&stream)? != (client_uid, client_pid) {
        return Err(HardwareError::Protocol("unexpected client identity"));
    }
    let mut interrupt = signal(SignalKind::interrupt())?;
    let mut terminate = signal(SignalKind::terminate())?;
    let mut hangup = signal(SignalKind::hangup())?;
    let shutdown = stream.try_clone()?;
    let stopped = Arc::new(AtomicBool::new(false));
    let worker_stopped = stopped.clone();
    let mut worker = tokio::task::spawn_blocking(move || serve(stream, HardwareControls::default(), &worker_stopped));
    tokio::select! {
        result = &mut worker => return result?,
        _ = interrupt.recv() => {},
        _ = terminate.recv() => {},
        _ = hangup.recv() => {},
    }
    // Wake an idle read, then let any in-flight write finish before restoration.
    stopped.store(true, Ordering::Release);
    let interrupted = shutdown.shutdown(Shutdown::Read);
    let result = worker.await?;
    interrupted?;
    result
}

fn serve(
    mut stream: UnixStream,
    mut controls: HardwareControls,
    stopped: &AtomicBool,
) -> Result<(), HardwareError> {
    let operation = (|| {
        // A client that stops reading must not block restoration indefinitely.
        stream.set_write_timeout(Some(Duration::from_secs(5)))?;
        let mut reader = BufReader::new(stream.try_clone()?);
        loop {
            if stopped.load(Ordering::Acquire) {
                return Ok(false);
            }
            let request = read_message::<PerformanceMode>(&mut reader)?;
            if stopped.load(Ordering::Acquire) {
                return Ok(false);
            }
            let Some(mode) = request else {
                return Ok(true);
            };
            let result = controls.apply(mode).map_err(|error| error.to_string());
            if stopped.load(Ordering::Acquire) {
                return Ok(false);
            }
            write_message(&mut stream, &result)?;
        }
    })();
    let restoration = controls.restore();
    let operation = operation.and_then(|eof| {
        // Signal cancellation must not masquerade as a successful pending command.
        if !eof {
            return Ok(());
        }
        let acknowledgement = restoration.as_ref().copied().map_err(|error| error.to_string());
        write_message(&mut stream, &acknowledgement)
    });
    match (operation, restoration) {
        (Ok(()), Ok(())) => Ok(()),
        (Err(operation), Err(restore)) => Err(HardwareError::Rollback {
            operation: Box::new(operation),
            restore: Box::new(restore),
        }),
        (Err(error), _) | (_, Err(error)) => Err(error),
    }
}
