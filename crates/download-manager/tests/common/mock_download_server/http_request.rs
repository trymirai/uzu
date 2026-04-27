use std::{collections::HashMap, io::Result as IoResult};

use tokio::{io::AsyncReadExt, net::TcpStream};

use crate::common::mock_download_server::utils::parse_range;

#[derive(Debug)]
pub(super) struct HttpRequest {
    pub method: String,
    pub path: String,
    pub headers: HashMap<String, String>,
    pub range: Option<(u64, Option<u64>)>,
}

pub(super) async fn read_request(stream: &mut TcpStream) -> IoResult<Option<HttpRequest>> {
    let mut buffer = Vec::new();
    loop {
        let mut chunk = [0u8; 4096];
        let bytes_read = stream.read(&mut chunk).await?;
        if bytes_read == 0 {
            return Ok(None);
        }
        buffer.extend_from_slice(&chunk[..bytes_read]);
        if buffer.windows(4).any(|window| window == b"\r\n\r\n") {
            break;
        }
    }

    let header_text = String::from_utf8_lossy(&buffer);
    let mut lines = header_text.split("\r\n");
    let Some(request_line) = lines.next() else {
        return Ok(None);
    };
    let mut request_parts = request_line.split_whitespace();
    let method = request_parts.next().unwrap_or_default().to_string();
    let raw_path = request_parts.next().unwrap_or("/");
    let path = raw_path.split('?').next().unwrap_or(raw_path).to_string();
    let headers = lines
        .take_while(|line| !line.is_empty())
        .filter_map(|line| {
            let (name, value) = line.split_once(':')?;
            Some((name.trim().to_ascii_lowercase(), value.trim().to_string()))
        })
        .collect::<HashMap<_, _>>();
    let range = headers.get("range").and_then(|value| parse_range(value));

    Ok(Some(HttpRequest {
        method,
        path,
        headers,
        range,
    }))
}
