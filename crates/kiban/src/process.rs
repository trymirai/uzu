#[cfg(unix)]
pub async fn is_process_alive(process_id: u32) -> bool {
    // SAFETY: `kill(pid, 0)` is a probe with no signal delivery. Direct syscall is required
    // because sandboxed macOS apps can't exec `/bin/kill`.
    let result = unsafe { libc::kill(process_id as libc::pid_t, 0) };
    if result == 0 {
        return true;
    }
    // EPERM = exists but we can't signal it; ESRCH = gone.
    std::io::Error::last_os_error().raw_os_error() == Some(libc::EPERM)
}

#[cfg(windows)]
pub async fn is_process_alive(process_id: u32) -> bool {
    tokio::task::spawn_blocking(move || {
        std::process::Command::new("tasklist")
            .args(["/FI", &format!("PID eq {process_id}")])
            .output()
            .map(|output| String::from_utf8_lossy(&output.stdout).contains(&process_id.to_string()))
            .unwrap_or(false)
    })
    .await
    .unwrap_or(false)
}

#[cfg(target_family = "wasm")]
pub async fn is_process_alive(_process_id: u32) -> bool {
    false
}

pub fn id() -> u32 {
    if proc_supported() {
        std::process::id()
    } else {
        0
    }
}

pub fn proc_supported() -> bool {
    #[cfg(not(target_family = "wasm"))]
    return true;

    #[cfg(target_family = "wasm")]
    false
}
