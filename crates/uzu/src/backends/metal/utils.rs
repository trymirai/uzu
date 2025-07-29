#![allow(dead_code)]
use std::{
    ffi,
    sync::{Arc, Mutex},
};

bitflags::bitflags! {
    pub struct CaptureOptions: u8 {
        const STDOUT = 0b00000001;
        const STDERR = 0b00000010;
        const BOTH = Self::STDOUT.bits() | Self::STDERR.bits();
    }
}

/// Utility for capturing stdout/stderr output (for placement analysis, etc.)
pub struct StdoutCapture {
    capture_stdout: bool,
    capture_stderr: bool,
    _stdout_old: Option<*mut ffi::c_void>,
    _stderr_old: Option<*mut ffi::c_void>,
    stdout_pipe: Option<[i32; 2]>,
    stderr_pipe: Option<[i32; 2]>,
    stdout_thread: Option<std::thread::JoinHandle<()>>,
    stderr_thread: Option<std::thread::JoinHandle<()>>,
    stdout_buffer: Option<Arc<Mutex<String>>>,
    stderr_buffer: Option<Arc<Mutex<String>>>,
}

impl StdoutCapture {
    pub fn new(options: CaptureOptions) -> Self {
        Self {
            capture_stdout: options.contains(CaptureOptions::STDOUT),
            capture_stderr: options.contains(CaptureOptions::STDERR),
            _stdout_old: None,
            _stderr_old: None,
            stdout_pipe: None,
            stderr_pipe: None,
            stdout_thread: None,
            stderr_thread: None,
            stdout_buffer: None,
            stderr_buffer: None,
        }
    }

    pub fn start(&mut self) {
        unsafe {
            if self.capture_stdout {
                let mut pipes = [0; 2];
                libc::pipe(pipes.as_mut_ptr());
                self.stdout_pipe = Some(pipes);
                let stdout_old = libc::dup(libc::STDOUT_FILENO);
                self._stdout_old = Some(stdout_old as *mut ffi::c_void);
                libc::dup2(self.stdout_pipe.unwrap()[1], libc::STDOUT_FILENO);
                libc::close(self.stdout_pipe.unwrap()[1]);
                let read_fd = self.stdout_pipe.unwrap()[0];
                let buffer = Arc::new(Mutex::new(String::new()));
                let buffer_clone = Arc::clone(&buffer);
                self.stdout_buffer = Some(buffer);
                self.stdout_thread = Some(std::thread::spawn(move || {
                    let mut buf = [0u8; 4096];
                    loop {
                        let bytes_read = libc::read(
                            read_fd,
                            buf.as_mut_ptr() as *mut libc::c_void,
                            buf.len(),
                        );
                        if bytes_read <= 0 {
                            break;
                        }
                        if let Ok(s) =
                            std::str::from_utf8(&buf[0..bytes_read as usize])
                        {
                            let mut b = buffer_clone.lock().unwrap();
                            b.push_str(s);
                        }
                    }
                    libc::close(read_fd);
                }));
            }
            if self.capture_stderr {
                let mut pipes = [0; 2];
                libc::pipe(pipes.as_mut_ptr());
                self.stderr_pipe = Some(pipes);
                let stderr_old = libc::dup(libc::STDERR_FILENO);
                self._stderr_old = Some(stderr_old as *mut ffi::c_void);
                libc::dup2(self.stderr_pipe.unwrap()[1], libc::STDERR_FILENO);
                libc::close(self.stderr_pipe.unwrap()[1]);
                let read_fd = self.stderr_pipe.unwrap()[0];
                let buffer = Arc::new(Mutex::new(String::new()));
                let buffer_clone = Arc::clone(&buffer);
                self.stderr_buffer = Some(buffer);
                self.stderr_thread = Some(std::thread::spawn(move || {
                    let mut buf = [0u8; 4096];
                    loop {
                        let bytes_read = libc::read(
                            read_fd,
                            buf.as_mut_ptr() as *mut libc::c_void,
                            buf.len(),
                        );
                        if bytes_read <= 0 {
                            break;
                        }
                        if let Ok(s) =
                            std::str::from_utf8(&buf[0..bytes_read as usize])
                        {
                            let mut b = buffer_clone.lock().unwrap();
                            b.push_str(s);
                        }
                    }
                    libc::close(read_fd);
                }));
            }
        }
    }

    pub fn stop(&mut self) -> String {
        unsafe {
            libc::fflush(std::ptr::null_mut());
            if let Some(stdout_old) = self._stdout_old {
                libc::dup2(stdout_old as i32, libc::STDOUT_FILENO);
                libc::close(stdout_old as i32);
                self._stdout_old = None;
            }
            if let Some(stderr_old) = self._stderr_old {
                libc::dup2(stderr_old as i32, libc::STDERR_FILENO);
                libc::close(stderr_old as i32);
                self._stderr_old = None;
            }
        }
        if let Some(thread) = self.stdout_thread.take() {
            let _ = thread.join();
        }
        if let Some(thread) = self.stderr_thread.take() {
            let _ = thread.join();
        }
        let mut captured = String::new();
        if let Some(buffer) = &self.stdout_buffer {
            let b = buffer.lock().unwrap();
            captured.push_str(&b);
        }
        if let Some(buffer) = &self.stderr_buffer {
            let b = buffer.lock().unwrap();
            captured.push_str(&b);
        }
        self.stdout_buffer = None;
        self.stderr_buffer = None;
        captured
    }
}
