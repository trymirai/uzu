use std::{
    fs::File,
    io::{Error as IoError, Read},
    path::Path,
};

use base64::Engine;

const CHUNK_SIZE: usize = 8 * 1024 * 1024;

pub fn calculate_and_verify_crc(
    file_path: &Path,
    expected_crc32c_base64: &str,
) -> Result<bool, IoError> {
    let expected_bytes = match base64::engine::general_purpose::STANDARD.decode(expected_crc32c_base64) {
        Ok(bytes) => bytes,
        Err(_) => return Ok(false),
    };

    if expected_bytes.len() != 4 {
        return Ok(false);
    }

    let expected_crc = u32::from_be_bytes([expected_bytes[0], expected_bytes[1], expected_bytes[2], expected_bytes[3]]);

    let mut file = File::open(file_path)?;
    let mut buffer = vec![0u8; CHUNK_SIZE];
    let mut crc: u32 = 0;

    loop {
        let bytes_read = file.read(&mut buffer)?;
        if bytes_read == 0 {
            break;
        }
        crc = crc32c::crc32c_append(crc, &buffer[..bytes_read]);
    }

    Ok(crc == expected_crc)
}

pub fn save_crc_file(
    file_path: &Path,
    crc_value: &str,
) -> Result<(), IoError> {
    let crc_path_str = format!("{}.crc", file_path.display());
    std::fs::write(&crc_path_str, crc_value)?;
    Ok(())
}
