use http::Request;
use tokio::{io::AsyncReadExt, net::TcpStream};

use crate::{Error, Result};

const REQUEST_HEADER_LIMIT: usize = 16 * 1024;

pub(crate) async fn read_request(stream: &mut TcpStream) -> Result<Request<()>> {
    let mut request_bytes = Vec::new();
    let mut buffer = [0_u8; 1024];
    while !request_bytes.windows(4).any(|window| window == b"\r\n\r\n") {
        let bytes_read = stream.read(&mut buffer).await?;
        if bytes_read == 0 {
            return Err(Error::RequestEndedBeforeHeaders);
        }
        request_bytes.extend_from_slice(&buffer[..bytes_read]);
        if request_bytes.len() > REQUEST_HEADER_LIMIT {
            return Err(Error::RequestHeadersTooLarge {
                limit: REQUEST_HEADER_LIMIT,
            });
        }
    }

    let mut headers = [httparse::EMPTY_HEADER; 32];
    let mut parsed_request = httparse::Request::new(&mut headers);
    parsed_request.parse(&request_bytes)?;

    let method = parsed_request.method.ok_or(Error::MissingRequestMethod)?;
    let path = parsed_request.path.ok_or(Error::MissingRequestPath)?;

    let mut request_builder = Request::builder().method(method).uri(path);
    for header in parsed_request.headers {
        request_builder = request_builder.header(header.name, header.value);
    }

    Ok(request_builder.body(())?)
}
