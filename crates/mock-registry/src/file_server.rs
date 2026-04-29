use std::{net::SocketAddr, sync::Arc, time::Duration};

use http::{Method, StatusCode, header::RANGE};
use stream_throttle::{ThrottlePool, ThrottleRate};
use tokio::{
    io::AsyncWriteExt,
    net::{TcpListener, TcpStream},
    task::JoinHandle as TokioJoinHandle,
};

use crate::{
    Behavior, Result, ServedFile,
    file_request::read_request,
    file_response::{empty_response, file_response, write_response_headers},
};

const THROTTLED_CHUNK_SIZE: usize = 64 * 1024;
const THROTTLED_CHUNK_INTERVAL: Duration = Duration::from_millis(5);

pub(crate) struct FileServer {
    listener: TcpListener,
    pub(crate) base_url: String,
}

impl FileServer {
    pub(crate) async fn bind() -> Result<Self> {
        let listener = TcpListener::bind(("127.0.0.1", 0)).await?;
        let base_url = file_server_base_url(&listener)?;
        Ok(Self {
            listener,
            base_url,
        })
    }

    pub(crate) fn serve(
        self,
        files: Box<[ServedFile]>,
        behavior: Behavior,
    ) -> FileServerTask {
        FileServerTask {
            task: tokio::spawn(serve_files(self.listener, files, behavior)),
        }
    }
}

pub(crate) struct FileServerTask {
    task: TokioJoinHandle<()>,
}

impl Drop for FileServerTask {
    fn drop(&mut self) {
        self.task.abort();
    }
}

async fn serve_files(
    listener: TcpListener,
    files: Box<[ServedFile]>,
    behavior: Behavior,
) {
    let files: Arc<[ServedFile]> = files.into();
    loop {
        let Ok((stream, _address)) = listener.accept().await else {
            break;
        };
        let files = files.clone();
        tokio::spawn(async move {
            if let Err(error) = handle_file_request(stream, files, behavior).await {
                tracing::debug!("mock registry file request failed: {}", error);
            }
        });
    }
}

async fn handle_file_request(
    mut stream: TcpStream,
    files: Arc<[ServedFile]>,
    behavior: Behavior,
) -> Result<()> {
    let request = read_request(&mut stream).await?;
    let Some(served_file) = files.iter().find(|served_file| request.uri().path() == file_path(served_file)) else {
        let response = empty_response(StatusCode::NOT_FOUND)?;
        write_response_headers(&mut stream, &response).await?;
        return Ok(());
    };
    if request.method() != Method::GET && request.method() != Method::HEAD {
        let response = empty_response(StatusCode::METHOD_NOT_ALLOWED)?;
        write_response_headers(&mut stream, &response).await?;
        return Ok(());
    }

    let bytes = if behavior.contains(Behavior::CORRUPT_BODY) {
        corrupt_bytes(&served_file.bytes)
    } else {
        served_file.bytes.clone()
    };
    let file_size = bytes.len() as u64;
    let range_header = request.headers().get(RANGE).and_then(|header| header.to_str().ok());
    let (status, body_start, body_end_exclusive) =
        parse_range(range_header, file_size).unwrap_or((StatusCode::OK, 0, file_size));
    let response = file_response(status, body_start, body_end_exclusive, file_size)?;

    write_response_headers(&mut stream, &response).await?;
    if request.method() == Method::GET && response.status() != StatusCode::RANGE_NOT_SATISFIABLE {
        write_file_body(&mut stream, &bytes[body_start as usize..body_end_exclusive as usize], behavior).await?;
    }
    Ok(())
}

async fn write_file_body(
    stream: &mut TcpStream,
    body: &[u8],
    behavior: Behavior,
) -> Result<()> {
    if !behavior.contains(Behavior::THROTTLED) {
        stream.write_all(body).await?;
        return Ok(());
    }

    let throttle = ThrottlePool::new(ThrottleRate::new(1, THROTTLED_CHUNK_INTERVAL));
    for chunk in body.chunks(THROTTLED_CHUNK_SIZE) {
        throttle.queue().await;
        stream.write_all(chunk).await?;
        stream.flush().await?;
    }
    Ok(())
}

fn corrupt_bytes(bytes: &Arc<[u8]>) -> Arc<[u8]> {
    let mut corrupted_bytes = bytes.to_vec();
    if let Some(first_byte) = corrupted_bytes.first_mut() {
        *first_byte = first_byte.wrapping_add(1);
    }
    corrupted_bytes.into()
}

fn file_server_base_url(listener: &TcpListener) -> Result<String> {
    let address = listener.local_addr()?;
    Ok(match address {
        SocketAddr::V4(address) => format!("http://{}", address),
        SocketAddr::V6(address) => format!("http://[{}]:{}", address.ip(), address.port()),
    })
}

fn file_path(served_file: &ServedFile) -> String {
    format!("/{}", served_file.file.name)
}

fn parse_range(
    header: Option<&str>,
    file_size: u64,
) -> Option<(StatusCode, u64, u64)> {
    let range = header?.strip_prefix("bytes=")?;
    let (start, end) = range.split_once('-')?;
    let start = start.parse::<u64>().ok()?;
    if start >= file_size {
        return Some((StatusCode::RANGE_NOT_SATISFIABLE, 0, 0));
    }
    let end_exclusive = end.parse::<u64>().ok().map_or(file_size, |end| (end + 1).min(file_size));
    Some((StatusCode::PARTIAL_CONTENT, start, end_exclusive))
}
