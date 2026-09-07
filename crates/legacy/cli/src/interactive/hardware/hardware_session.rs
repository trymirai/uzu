use std::{
    fs::{self, DirBuilder},
    io::{BufReader, Read, Write},
    net::Shutdown,
    os::unix::{
        fs::DirBuilderExt,
        net::{UnixListener, UnixStream},
    },
    path::PathBuf,
    process::{Child, Command, Stdio},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    thread,
    time::{Duration, Instant},
};

use super::{HardwareError, PerformanceMode, peer_identity, read_message, write_message};

const AUTHORIZE: &str = r#"on run argv
    set helperCommand to "exec " & quoted form of (item 1 of argv) & " hardware-helper " & quoted form of (item 2 of argv) & " " & quoted form of (item 3 of argv) & " " & quoted form of (item 4 of argv)
    do shell script helperCommand with administrator privileges
end run"#;

/// Unprivileged client of the session's authorized hardware process.
pub struct HardwareSession {
    closed: Arc<AtomicBool>,
    connection: Option<BufReader<UnixStream>>,
    authorization: Option<Child>,
    directory: Option<PathBuf>,
}

impl HardwareSession {
    pub fn new(closed: Arc<AtomicBool>) -> Self {
        Self {
            closed,
            connection: None,
            authorization: None,
            directory: None,
        }
    }

    pub fn apply(
        &mut self,
        mode: PerformanceMode,
    ) -> Result<(), HardwareError> {
        if self.closed.load(Ordering::Acquire) {
            return Err(HardwareError::Closed);
        }
        if self.connection.is_none() {
            if mode == PerformanceMode::Auto {
                return Ok(());
            }
            if let Err(error) = self.connect() {
                self.remove_socket();
                // No request has been sent. A late helper can only see EOF or a missing socket.
                if let Some(mut child) = self.authorization.take() {
                    let _ = child.kill();
                    let _ = child.wait();
                }
                return Err(error);
            }
        }
        if self.closed.load(Ordering::Acquire) {
            return Err(HardwareError::Closed);
        }
        let result = self.exchange(mode);
        if result.is_err() && !matches!(&result, Err(HardwareError::Remote(_))) {
            // Lost framing or acknowledgement makes another request unsafe. EOF restores the helper.
            self.closed.store(true, Ordering::Release);
            if let Some(connection) = self.connection.take() {
                let _ = connection.get_ref().shutdown(Shutdown::Both);
            }
        }
        result
    }

    fn connect(&mut self) -> Result<(), HardwareError> {
        // A short, private path also fits macOS's 104-byte Unix socket address.
        let directory = PathBuf::from(format!("/tmp/mirai-hw-{}", uuid::Uuid::new_v4().simple()));
        DirBuilder::new().mode(0o700).create(&directory)?;
        self.directory = Some(directory.clone());
        let socket = directory.join("control");
        let listener = UnixListener::bind(&socket)?;
        listener.set_nonblocking(true)?;
        self.authorization = Some(
            Command::new("/usr/bin/osascript")
                .args(["-e", AUTHORIZE, "--"])
                .arg(std::env::current_exe()?)
                .arg(&socket)
                .arg(unsafe { libc::geteuid() }.to_string())
                .arg(std::process::id().to_string())
                .stdin(Stdio::null())
                .stdout(Stdio::null())
                .stderr(Stdio::piped())
                .spawn()?,
        );
        let authorization =
            self.authorization.as_mut().ok_or(HardwareError::Protocol("authorization did not start"))?;
        let stream = accept_helper(&listener, authorization, &self.closed)?;
        // macOS accept inherits O_NONBLOCK; RPC reads must wait for the helper's reply.
        stream.set_nonblocking(false)?;
        // Up to ten fans can each need readback settling, including both rollback attempts.
        stream.set_read_timeout(Some(Duration::from_secs(300)))?;
        stream.set_write_timeout(Some(Duration::from_secs(5)))?;
        self.connection = Some(BufReader::new(stream));
        self.remove_socket();
        Ok(())
    }

    fn exchange(
        &mut self,
        mode: PerformanceMode,
    ) -> Result<(), HardwareError> {
        let connection = self.connection.as_mut().ok_or(HardwareError::Protocol("connection is unavailable"))?;
        write_message(connection.get_mut(), &mode)?;
        read_message::<Result<(), String>>(connection)?
            .ok_or(HardwareError::Protocol("connection closed before acknowledging the setting"))?
            .map_err(HardwareError::Remote)
    }

    pub fn restore(&mut self) -> Result<(), HardwareError> {
        self.closed.store(true, Ordering::Release);
        let restoration = if let Some(mut connection) = self.connection.take() {
            // The helper restores on EOF and sends one final acknowledgement before exiting.
            (|| {
                connection.get_ref().shutdown(Shutdown::Write)?;
                read_message::<Result<(), String>>(&mut connection)?
                    .ok_or(HardwareError::Protocol("helper exited without confirming restoration"))?
                    .map_err(HardwareError::Remote)
            })()
        } else {
            Ok(())
        };
        self.remove_socket();
        let completion = self.wait_for_helper();
        match (restoration, completion) {
            (Err(operation), Err(restore)) => Err(HardwareError::Rollback {
                operation: Box::new(operation),
                restore: Box::new(restore),
            }),
            (Err(error), _) | (_, Err(error)) => Err(error),
            _ => Ok(()),
        }
    }

    fn wait_for_helper(&mut self) -> Result<(), HardwareError> {
        let Some(child) = self.authorization.as_mut() else {
            return Ok(());
        };
        let deadline = Instant::now() + Duration::from_secs(5);
        loop {
            if let Some(status) = child.try_wait()? {
                let result = if status.success() {
                    Ok(())
                } else {
                    Err(authorization_error(child))
                };
                self.authorization = None;
                return result;
            }
            if Instant::now() >= deadline {
                // Never kill a helper that may still be restoring hardware. Reap it when it exits.
                if let Some(mut child) = self.authorization.take() {
                    thread::spawn(move || {
                        let _ = child.wait();
                    });
                }
                return Err(HardwareError::Protocol("helper has not exited; restoration may still be running"));
            }
            thread::sleep(Duration::from_millis(25));
        }
    }

    fn remove_socket(&mut self) {
        if let Some(directory) = self.directory.take() {
            for result in [fs::remove_file(directory.join("control")), fs::remove_dir(&directory)] {
                if let Err(error) = result
                    && error.kind() != std::io::ErrorKind::NotFound
                {
                    tracing::warn!("Unable to remove hardware session socket: {error}");
                }
            }
        }
    }
}

impl Drop for HardwareSession {
    fn drop(&mut self) {
        if let Err(error) = self.restore() {
            let _ = writeln!(std::io::stderr(), "Unable to restore hardware settings: {error}");
        }
    }
}

fn accept_helper(
    listener: &UnixListener,
    authorization: &mut Child,
    closed: &AtomicBool,
) -> Result<UnixStream, HardwareError> {
    let deadline = Instant::now() + Duration::from_secs(120);
    loop {
        if closed.load(Ordering::Acquire) {
            return Err(HardwareError::Closed);
        }
        match listener.accept() {
            Ok((stream, _)) => {
                if peer_identity(&stream)?.0 != 0 {
                    return Err(HardwareError::Protocol("connection was not made by an authorized helper"));
                }
                return Ok(stream);
            },
            Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {},
            Err(error) => return Err(error.into()),
        }
        if authorization.try_wait()?.is_some() {
            return Err(authorization_error(authorization));
        }
        if Instant::now() >= deadline {
            return Err(HardwareError::Protocol("administrator authorization timed out; select Fast to try again"));
        }
        thread::sleep(Duration::from_millis(25));
    }
}

fn authorization_error(child: &mut Child) -> HardwareError {
    let mut message = String::new();
    if let Some(stderr) = child.stderr.take()
        && let Err(error) = stderr.take(16 * 1024).read_to_string(&mut message)
    {
        return error.into();
    }
    if message.contains("(-128)") {
        message = "cancelled; select Fast to try again".to_string();
    } else if message.trim().is_empty() {
        message = "helper exited before completing the request".to_string();
    }
    HardwareError::Authorization(message.trim().to_string())
}
