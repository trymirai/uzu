use http::{
    Response, StatusCode,
    header::{
        ACCEPT_RANGES, CONNECTION, CONTENT_LENGTH, CONTENT_RANGE, CONTENT_TYPE, HeaderValue,
        LAST_MODIFIED as LAST_MODIFIED_HEADER,
    },
};
use tokio::{io::AsyncWriteExt, net::TcpStream};

use crate::Result;

const LAST_MODIFIED: &str = "Sun, 26 Apr 2026 12:00:00 GMT";

pub(crate) fn file_response(
    status: StatusCode,
    body_start: u64,
    body_end_exclusive: u64,
    file_size: u64,
) -> Result<Response<()>> {
    let content_length = body_end_exclusive.saturating_sub(body_start);
    let mut response = Response::builder()
        .status(status)
        .header(ACCEPT_RANGES, "bytes")
        .header(LAST_MODIFIED_HEADER, LAST_MODIFIED)
        .header(CONTENT_LENGTH, content_length.to_string())
        .header(CONTENT_TYPE, "application/octet-stream")
        .header(CONNECTION, "close");

    if status == StatusCode::PARTIAL_CONTENT {
        response = response.header(
            CONTENT_RANGE,
            format!("bytes {}-{}/{}", body_start, body_end_exclusive.saturating_sub(1), file_size),
        );
    } else if status == StatusCode::RANGE_NOT_SATISFIABLE {
        response = response.header(CONTENT_RANGE, format!("bytes */{file_size}"));
    }

    Ok(response.body(())?)
}

pub(crate) fn empty_response(status: StatusCode) -> Result<Response<()>> {
    Ok(Response::builder().status(status).header(CONTENT_LENGTH, "0").header(CONNECTION, "close").body(())?)
}

pub(crate) async fn write_response_headers(
    stream: &mut TcpStream,
    response: &Response<()>,
) -> Result<()> {
    let reason = response.status().canonical_reason().unwrap_or("Unknown");
    let mut response_head = format!("HTTP/1.1 {} {reason}\r\n", response.status().as_u16());
    for (name, value) in response.headers() {
        response_head.push_str(name.as_str());
        response_head.push_str(": ");
        response_head.push_str(header_value(value)?);
        response_head.push_str("\r\n");
    }
    response_head.push_str("\r\n");
    stream.write_all(response_head.as_bytes()).await?;
    Ok(())
}

fn header_value(value: &HeaderValue) -> Result<&str> {
    Ok(value.to_str()?)
}
