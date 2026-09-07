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

#[cfg(test)]
mod tests {
    use std::{io::Cursor, os::unix::net::UnixListener, time::SystemTime};

    use super::*;

    #[test]
    fn connected_peer_identity_comes_from_the_kernel() {
        let nonce = SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).unwrap().as_nanos();
        let path = std::env::temp_dir().join(format!("uzu-hardware-peer-{}-{nonce}.sock", std::process::id()));
        let listener = UnixListener::bind(&path).unwrap();
        let client = UnixStream::connect(&path).unwrap();
        let (server, _) = listener.accept().unwrap();
        std::fs::remove_file(path).unwrap();
        let expected = (unsafe { libc::geteuid() }, std::process::id() as i32);
        assert_eq!(peer_identity(&client).unwrap(), expected);
        assert_eq!(peer_identity(&server).unwrap(), expected);
    }

    #[test]
    fn messages_allow_exact_size_limit_and_reject_larger_before_writing() {
        let message = "x".repeat(MAX_MESSAGE - 3);
        let mut bytes = Vec::new();
        write_message(&mut bytes, &message).unwrap();
        assert_eq!(bytes.len(), MAX_MESSAGE);
        assert_eq!(read_message::<String>(&mut Cursor::new(bytes)).unwrap(), Some(message));
        let mut bytes = Vec::new();
        assert!(write_message(&mut bytes, &"x".repeat(MAX_MESSAGE - 2)).is_err());
        assert!(bytes.is_empty());
    }

    #[test]
    fn only_empty_eof_is_clean() {
        assert!(read_message::<String>(&mut Cursor::new([])).unwrap().is_none());
        assert!(matches!(read_message::<String>(&mut Cursor::new(b"\"Fast\"")), Err(HardwareError::Protocol(_))));
        assert!(matches!(read_message::<String>(&mut Cursor::new(b"invalid\n")), Err(HardwareError::Json(_))));
    }

    #[test]
    fn oversized_input_is_bounded() {
        let mut input = Cursor::new(vec![b'x'; MAX_MESSAGE * 2]);
        assert!(matches!(read_message::<String>(&mut input), Err(HardwareError::Protocol(_))));
        assert_eq!(input.position(), MAX_MESSAGE as u64 + 1);
    }
}
