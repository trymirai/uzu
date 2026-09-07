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
    let mut worker = tokio::task::spawn_blocking(move || {
        serve(stream, HardwareControls::default(), &worker_stopped, HardwareControls::apply, HardwareControls::restore)
    });
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

fn serve<C>(
    mut stream: UnixStream,
    mut controls: C,
    stopped: &AtomicBool,
    mut apply: impl FnMut(&mut C, PerformanceMode) -> Result<(), HardwareError>,
    mut restore: impl FnMut(&mut C) -> Result<(), HardwareError>,
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
            let result = apply(&mut controls, mode).map_err(|error| error.to_string());
            if stopped.load(Ordering::Acquire) {
                return Ok(false);
            }
            write_message(&mut stream, &result)?;
        }
    })();
    let restoration = restore(&mut controls);
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

#[cfg(test)]
mod tests {
    use std::{io::Write, sync::Mutex, thread};

    use super::*;

    #[test]
    fn clean_eof_restores_before_final_acknowledgement() {
        let (mut client, server) = UnixStream::pair().unwrap();
        let calls = Arc::new(Mutex::new(Vec::new()));
        let state = calls.clone();
        let worker = thread::spawn(move || {
            serve(
                server,
                state,
                &AtomicBool::new(false),
                |calls, _| {
                    calls.lock().unwrap().push("apply");
                    Ok(())
                },
                |calls| {
                    calls.lock().unwrap().push("restore");
                    Ok(())
                },
            )
        });
        write_message(&mut client, &PerformanceMode::Fast).unwrap();
        client.shutdown(Shutdown::Write).unwrap();
        let mut reader = BufReader::new(client);
        for _ in 0..2 {
            assert_eq!(read_message::<Result<(), String>>(&mut reader).unwrap(), Some(Ok(())));
        }
        worker.join().unwrap().unwrap();
        assert_eq!(*calls.lock().unwrap(), ["apply", "restore"]);
    }

    #[test]
    fn malformed_request_still_restores() {
        let (mut client, server) = UnixStream::pair().unwrap();
        client.write_all(b"invalid\n").unwrap();
        let mut restored = false;
        let result = serve(
            server,
            &mut restored,
            &AtomicBool::new(false),
            |_, _| panic!("invalid request must not be applied"),
            |restored| {
                **restored = true;
                Ok(())
            },
        );
        assert!(matches!(result, Err(HardwareError::Json(_))));
        assert!(restored);
    }

    #[test]
    fn restoration_failure_is_sent_in_final_acknowledgement() {
        let (client, server) = UnixStream::pair().unwrap();
        let worker = thread::spawn(move || {
            serve(
                server,
                (),
                &AtomicBool::new(false),
                |_, _| panic!("no request was sent"),
                |_| Err(HardwareError::Protocol("restore failed")),
            )
        });
        client.shutdown(Shutdown::Write).unwrap();
        let response = read_message::<Result<(), String>>(&mut BufReader::new(client)).unwrap();
        assert_eq!(response, Some(Err("Hardware helper: restore failed".to_owned())));
        assert!(matches!(worker.join().unwrap(), Err(HardwareError::Protocol("restore failed"))));
    }

    #[test]
    fn cancellation_during_apply_suppresses_success_and_queued_requests() {
        let (mut client, server) = UnixStream::pair().unwrap();
        for _ in 0..2 {
            write_message(&mut client, &PerformanceMode::Fast).unwrap();
        }
        client.shutdown(Shutdown::Write).unwrap();
        let stopped = AtomicBool::new(false);
        let mut calls = Vec::new();
        serve(
            server,
            &mut calls,
            &stopped,
            |calls, _| {
                calls.push("apply");
                stopped.store(true, Ordering::Release);
                Ok(())
            },
            |calls| {
                calls.push("restore");
                Ok(())
            },
        )
        .unwrap();
        let mut reader = BufReader::new(client);
        assert!(read_message::<Result<(), String>>(&mut reader).unwrap().is_none());
        assert_eq!(calls, ["apply", "restore"]);
    }
}
