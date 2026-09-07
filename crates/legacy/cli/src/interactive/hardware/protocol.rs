use std::{
    io::{self, BufRead, Read, Write},
    os::{fd::AsRawFd, unix::net::UnixStream},
};

use serde::{Serialize, de::DeserializeOwned};

use super::HardwareError;

const MAX_MESSAGE: usize = 16 * 1024;

pub fn peer_identity(stream: &UnixStream) -> io::Result<(u32, i32)> {
    let mut uid = 0;
    let mut gid = 0;
    let mut pid: libc::pid_t = 0;
    let mut size = size_of::<libc::pid_t>() as libc::socklen_t;
    // Peer credentials come from the connected kernel socket, never from a message.
    if unsafe { libc::getpeereid(stream.as_raw_fd(), &mut uid, &mut gid) } != 0 {
        return Err(io::Error::last_os_error());
    }
    if unsafe {
        libc::getsockopt(
            stream.as_raw_fd(),
            libc::SOL_LOCAL,
            libc::LOCAL_PEERPID,
            (&mut pid as *mut libc::pid_t).cast(),
            &mut size,
        )
    } != 0
    {
        return Err(io::Error::last_os_error());
    }
    if size as usize != size_of::<libc::pid_t>() || pid <= 0 {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "invalid peer process identity"));
    }
    Ok((uid, pid))
}

pub fn read_message<T: DeserializeOwned>(reader: &mut impl BufRead) -> Result<Option<T>, HardwareError> {
    let mut bytes = Vec::new();
    let count = reader.take(MAX_MESSAGE as u64 + 1).read_until(b'\n', &mut bytes)?;
    if count == 0 {
        return Ok(None);
    }
    if count > MAX_MESSAGE {
        return Err(HardwareError::Protocol("message exceeds 16 KiB"));
    }
    if bytes.last() != Some(&b'\n') {
        return Err(HardwareError::Protocol("message ended before newline"));
    }
    Ok(Some(serde_json::from_slice(&bytes[..count - 1])?))
}

pub fn write_message<T: Serialize>(
    writer: &mut impl Write,
    message: &T,
) -> Result<(), HardwareError> {
    let mut bytes = serde_json::to_vec(message)?;
    if bytes.len() >= MAX_MESSAGE {
        return Err(HardwareError::Protocol("message exceeds 16 KiB"));
    }
    bytes.push(b'\n');
    writer.write_all(&bytes)?;
    writer.flush()?;
    Ok(())
}
