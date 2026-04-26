#![allow(dead_code)]

use std::{
    collections::HashMap,
    net::SocketAddr,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
    time::Duration,
};

use base64::Engine;
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::{TcpListener, TcpStream},
    sync::{Mutex, Notify, RwLock},
    task::JoinHandle,
};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MockFile {
    pub name: String,
    pub path: String,
    pub bytes: Arc<[u8]>,
    pub size: u64,
    pub crc32c: String,
    pub last_modified: String,
}

impl MockFile {
    pub fn config(path_prefix: &str) -> Self {
        Self::new(
            path_prefix,
            "config.json",
            br#"{"model_type":"mock","hidden_size":32,"num_attention_heads":4}"#.to_vec(),
        )
    }

    pub fn tokenizer(path_prefix: &str) -> Self {
        Self::new(path_prefix, "tokenizer.json", deterministic_bytes(256 * 1024))
    }

    pub fn large_tokenizer(path_prefix: &str) -> Self {
        Self::new(path_prefix, "tokenizer.json", deterministic_bytes(2 * 1024 * 1024))
    }

    pub fn tokenizer_config(path_prefix: &str) -> Self {
        Self::new(path_prefix, "tokenizer_config.json", br#"{"add_bos_token":true}"#.to_vec())
    }

    pub fn new(
        path_prefix: &str,
        name: &str,
        bytes: Vec<u8>,
    ) -> Self {
        let crc32c = crc32c::crc32c(&bytes);
        let crc32c = base64::engine::general_purpose::STANDARD.encode(crc32c.to_be_bytes());
        let path = format!("/{}/{}", path_prefix.trim_matches('/'), name);
        Self {
            name: name.to_string(),
            path,
            size: bytes.len() as u64,
            bytes: Arc::from(bytes),
            crc32c,
            last_modified: "Sun, 26 Apr 2026 12:00:00 GMT".to_string(),
        }
    }

    pub fn corrupt_variant(&self) -> Self {
        let mut bytes = self.bytes.to_vec();
        if let Some(first_byte) = bytes.first_mut() {
            *first_byte = first_byte.wrapping_add(1);
        }
        Self {
            name: self.name.clone(),
            path: self.path.clone(),
            bytes: Arc::from(bytes),
            size: self.size,
            crc32c: self.crc32c.clone(),
            last_modified: self.last_modified.clone(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MockFileSet {
    pub config: MockFile,
    pub tokenizer: MockFile,
    pub tokenizer_config: MockFile,
}

impl MockFileSet {
    pub fn qwen_like(model_path: &str) -> Self {
        Self {
            config: MockFile::config(model_path),
            tokenizer: MockFile::tokenizer(model_path),
            tokenizer_config: MockFile::tokenizer_config(model_path),
        }
    }

    pub fn all(&self) -> Box<[MockFile]> {
        Box::new([self.config.clone(), self.tokenizer.clone(), self.tokenizer_config.clone()])
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RouteBehavior {
    Normal,
    SlowChunks {
        chunk_size: usize,
        delay_ms: u64,
    },
    StallAt {
        byte_offset: u64,
    },
    DisconnectAt {
        byte_offset: u64,
    },
    RetryThenOk {
        failures: u64,
        status: u16,
    },
    RedirectTo {
        target: String,
    },
    CorruptBody,
    WrongContentLength,
    NoContentLength,
    NoRangeSupport,
    InvalidRangeResponse,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RequestRecord {
    pub order: u64,
    pub method: String,
    pub path: String,
    pub headers: HashMap<String, String>,
    pub range: Option<(u64, Option<u64>)>,
    pub status: u16,
    pub bytes_sent: u64,
}

#[derive(Clone, Debug)]
struct MockRoute {
    file: MockFile,
    behavior: RouteBehavior,
}

#[derive(Debug)]
struct ServerState {
    routes: RwLock<HashMap<String, MockRoute>>,
    records: Mutex<Vec<RequestRecord>>,
    bytes_sent_by_path: Mutex<HashMap<String, u64>>,
    request_counts_by_path: Mutex<HashMap<String, u64>>,
    stall_notifies_by_path: Mutex<HashMap<String, Arc<Notify>>>,
    next_order: AtomicU64,
    notify: Notify,
}

#[derive(Debug)]
pub struct MockDownloadServer {
    address: SocketAddr,
    state: Arc<ServerState>,
    server_task: JoinHandle<()>,
}

impl MockDownloadServer {
    pub async fn start() -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").await.expect("failed to bind mock server");
        let address = listener.local_addr().expect("failed to read mock server address");
        let state = Arc::new(ServerState {
            routes: RwLock::new(HashMap::new()),
            records: Mutex::new(Vec::new()),
            bytes_sent_by_path: Mutex::new(HashMap::new()),
            request_counts_by_path: Mutex::new(HashMap::new()),
            stall_notifies_by_path: Mutex::new(HashMap::new()),
            next_order: AtomicU64::new(1),
            notify: Notify::new(),
        });

        let server_state = state.clone();
        let server_task = tokio::spawn(async move {
            loop {
                let Ok((stream, _peer_address)) = listener.accept().await else {
                    break;
                };
                let connection_state = server_state.clone();
                tokio::spawn(async move {
                    let _ = handle_connection(stream, connection_state).await;
                });
            }
        });

        Self {
            address,
            state,
            server_task,
        }
    }

    pub fn base_url(&self) -> String {
        format!("http://{}", self.address)
    }

    pub fn url_for_path(
        &self,
        path: &str,
    ) -> String {
        format!("{}{}", self.base_url(), path)
    }

    pub fn url_for_file(
        &self,
        file: &MockFile,
    ) -> String {
        self.url_for_path(&file.path)
    }

    pub async fn serve_file(
        &self,
        file: MockFile,
        behavior: RouteBehavior,
    ) {
        self.state.routes.write().await.insert(
            file.path.clone(),
            MockRoute {
                file,
                behavior,
            },
        );
    }

    pub async fn serve_file_set(
        &self,
        file_set: &MockFileSet,
        behavior: RouteBehavior,
    ) {
        for file in file_set.all() {
            self.serve_file(file, behavior.clone()).await;
        }
    }

    pub async fn release_stall(
        &self,
        path: &str,
    ) {
        if let Some(notify) = self.state.stall_notifies_by_path.lock().await.get(path).cloned() {
            notify.notify_waiters();
        }
    }

    pub async fn wait_for_bytes(
        &self,
        path: &str,
        minimum_bytes: u64,
    ) {
        wait_until(Duration::from_secs(10), || async {
            let sent_bytes = self.state.bytes_sent_by_path.lock().await.get(path).copied().unwrap_or(0);
            sent_bytes >= minimum_bytes
        })
        .await;
    }

    pub async fn wait_for_request(
        &self,
        path: &str,
        method: &str,
    ) -> RequestRecord {
        wait_until_value(Duration::from_secs(10), || async {
            self.records_snapshot().await.into_iter().find(|record| record.path == path && record.method == method)
        })
        .await
    }

    pub async fn wait_for_range(
        &self,
        path: &str,
        start: u64,
    ) -> RequestRecord {
        wait_until_value(Duration::from_secs(10), || async {
            self.records_snapshot()
                .await
                .into_iter()
                .find(|record| record.path == path && record.range.is_some_and(|range| range.0 == start))
        })
        .await
    }

    pub async fn records_snapshot(&self) -> Vec<RequestRecord> {
        self.state.records.lock().await.clone()
    }
}

impl Drop for MockDownloadServer {
    fn drop(&mut self) {
        self.server_task.abort();
    }
}

async fn handle_connection(
    mut stream: TcpStream,
    state: Arc<ServerState>,
) -> std::io::Result<()> {
    let Some(request) = read_request(&mut stream).await? else {
        return Ok(());
    };

    let route = state.routes.read().await.get(&request.path).cloned();
    let Some(route) = route else {
        respond_empty(&mut stream, 404, "Not Found").await?;
        return Ok(());
    };

    let order = state.next_order.fetch_add(1, Ordering::SeqCst);
    state.records.lock().await.push(RequestRecord {
        order,
        method: request.method.clone(),
        path: request.path.clone(),
        headers: request.headers.clone(),
        range: request.range,
        status: 0,
        bytes_sent: 0,
    });
    state.notify.notify_waiters();

    let request_count = {
        let mut request_counts = state.request_counts_by_path.lock().await;
        let request_count = request_counts.entry(request.path.clone()).or_insert(0);
        *request_count += 1;
        *request_count
    };

    let bytes_sent = match &route.behavior {
        RouteBehavior::RetryThenOk {
            failures,
            status,
        } if request_count <= *failures => respond_empty(&mut stream, *status, "retryable failure").await?,
        RouteBehavior::RedirectTo {
            target,
        } => respond_redirect(&mut stream, target).await?,
        _ => respond_with_file(&mut stream, &state, &request, &route).await?,
    };

    update_record(&state, order, bytes_sent.0, bytes_sent.1).await;
    Ok(())
}

#[derive(Debug)]
struct HttpRequest {
    method: String,
    path: String,
    headers: HashMap<String, String>,
    range: Option<(u64, Option<u64>)>,
}

async fn read_request(stream: &mut TcpStream) -> std::io::Result<Option<HttpRequest>> {
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

async fn respond_with_file(
    stream: &mut TcpStream,
    state: &Arc<ServerState>,
    request: &HttpRequest,
    route: &MockRoute,
) -> std::io::Result<(u16, u64)> {
    let source_bytes = match route.behavior {
        RouteBehavior::CorruptBody => route.file.corrupt_variant().bytes,
        _ => route.file.bytes.clone(),
    };
    let range_supported = !matches!(route.behavior, RouteBehavior::NoRangeSupport);
    let file_size = source_bytes.len() as u64;

    let (status, body_start, body_end_exclusive) = if let Some((range_start, range_end)) = request.range {
        if !range_supported {
            (200, 0, file_size)
        } else if range_start >= file_size {
            let bytes_sent = respond_range_not_satisfiable(stream, file_size).await?;
            return Ok((416, bytes_sent));
        } else {
            let range_end_exclusive = range_end.map_or(file_size, |end| (end + 1).min(file_size));
            (206, range_start, range_end_exclusive)
        }
    } else {
        (200, 0, file_size)
    };

    let body = &source_bytes[body_start as usize..body_end_exclusive as usize];
    let advertised_length = match route.behavior {
        RouteBehavior::WrongContentLength => Some((body.len() as u64 / 2).max(1)),
        RouteBehavior::DisconnectAt {
            byte_offset,
        } => {
            let bytes_before_disconnect = byte_offset.saturating_sub(body_start).min(body.len() as u64);
            Some(bytes_before_disconnect)
        },
        RouteBehavior::NoContentLength => None,
        _ => Some(body.len() as u64),
    };
    let mut header_lines = vec![
        format!("HTTP/1.1 {} {}\r\n", status, reason_phrase(status)),
        "Connection: close\r\n".to_string(),
        "Content-Type: application/octet-stream\r\n".to_string(),
        format!("Last-Modified: {}\r\n", route.file.last_modified),
    ];
    if range_supported {
        header_lines.push("Accept-Ranges: bytes\r\n".to_string());
    }
    if let Some(content_length) = advertised_length {
        header_lines.push(format!("Content-Length: {}\r\n", content_length));
    }
    if status == 206 {
        let content_range = match route.behavior {
            RouteBehavior::InvalidRangeResponse => format!("Content-Range: bytes 0-0/{}\r\n", file_size),
            _ => format!(
                "Content-Range: bytes {}-{}/{}\r\n",
                body_start,
                body_end_exclusive.saturating_sub(1),
                file_size
            ),
        };
        header_lines.push(content_range);
    }
    header_lines.push("\r\n".to_string());
    stream.write_all(header_lines.concat().as_bytes()).await?;

    if request.method == "HEAD" {
        return Ok((status, 0));
    }

    let bytes_sent = write_body(stream, state, &route.file.path, body_start, body, &route.behavior).await?;
    Ok((status, bytes_sent))
}

async fn write_body(
    stream: &mut TcpStream,
    state: &Arc<ServerState>,
    path: &str,
    body_start: u64,
    body: &[u8],
    behavior: &RouteBehavior,
) -> std::io::Result<u64> {
    match behavior {
        RouteBehavior::SlowChunks {
            chunk_size,
            delay_ms,
        } => {
            let mut total_bytes_sent = 0;
            for chunk in body.chunks(*chunk_size) {
                stream.write_all(chunk).await?;
                total_bytes_sent += chunk.len() as u64;
                add_bytes_sent(state, path, chunk.len() as u64).await;
                tokio::time::sleep(Duration::from_millis(*delay_ms)).await;
            }
            Ok(total_bytes_sent)
        },
        RouteBehavior::StallAt {
            byte_offset,
        } => {
            let body_end = body_start + body.len() as u64;
            if *byte_offset <= body_start || *byte_offset >= body_end {
                stream.write_all(body).await?;
                add_bytes_sent(state, path, body.len() as u64).await;
                return Ok(body.len() as u64);
            }

            let bytes_before_stall = (*byte_offset - body_start) as usize;
            stream.write_all(&body[..bytes_before_stall]).await?;
            add_bytes_sent(state, path, bytes_before_stall as u64).await;

            let notify = {
                let mut notifies = state.stall_notifies_by_path.lock().await;
                notifies.entry(path.to_string()).or_insert_with(|| Arc::new(Notify::new())).clone()
            };
            notify.notified().await;

            let remaining = &body[bytes_before_stall..];
            stream.write_all(remaining).await?;
            add_bytes_sent(state, path, remaining.len() as u64).await;
            Ok(body.len() as u64)
        },
        RouteBehavior::DisconnectAt {
            byte_offset,
        } => {
            let body_end = body_start + body.len() as u64;
            let bytes_to_send = if *byte_offset <= body_start {
                0
            } else if *byte_offset >= body_end {
                body.len()
            } else {
                (*byte_offset - body_start) as usize
            };
            stream.write_all(&body[..bytes_to_send]).await?;
            add_bytes_sent(state, path, bytes_to_send as u64).await;
            Ok(bytes_to_send as u64)
        },
        RouteBehavior::WrongContentLength => {
            let bytes_to_send = (body.len() / 2).max(1);
            stream.write_all(&body[..bytes_to_send]).await?;
            add_bytes_sent(state, path, bytes_to_send as u64).await;
            Ok(bytes_to_send as u64)
        },
        _ => {
            stream.write_all(body).await?;
            add_bytes_sent(state, path, body.len() as u64).await;
            Ok(body.len() as u64)
        },
    }
}

async fn respond_empty(
    stream: &mut TcpStream,
    status: u16,
    message: &str,
) -> std::io::Result<(u16, u64)> {
    let body = message.as_bytes();
    let headers = format!(
        "HTTP/1.1 {} {}\r\nConnection: close\r\nContent-Length: {}\r\n\r\n",
        status,
        reason_phrase(status),
        body.len()
    );
    stream.write_all(headers.as_bytes()).await?;
    stream.write_all(body).await?;
    Ok((status, body.len() as u64))
}

async fn respond_redirect(
    stream: &mut TcpStream,
    target: &str,
) -> std::io::Result<(u16, u64)> {
    let headers =
        format!("HTTP/1.1 302 Found\r\nConnection: close\r\nLocation: {}\r\nContent-Length: 0\r\n\r\n", target);
    stream.write_all(headers.as_bytes()).await?;
    Ok((302, 0))
}

async fn respond_range_not_satisfiable(
    stream: &mut TcpStream,
    file_size: u64,
) -> std::io::Result<u64> {
    let headers = format!(
        "HTTP/1.1 416 Range Not Satisfiable\r\nConnection: close\r\nContent-Range: bytes */{}\r\nContent-Length: 0\r\n\r\n",
        file_size
    );
    stream.write_all(headers.as_bytes()).await?;
    Ok(0)
}

async fn add_bytes_sent(
    state: &Arc<ServerState>,
    path: &str,
    bytes_sent: u64,
) {
    let mut bytes_sent_by_path = state.bytes_sent_by_path.lock().await;
    *bytes_sent_by_path.entry(path.to_string()).or_insert(0) += bytes_sent;
    state.notify.notify_waiters();
}

async fn update_record(
    state: &Arc<ServerState>,
    order: u64,
    status: u16,
    bytes_sent: u64,
) {
    if let Some(record) = state.records.lock().await.iter_mut().find(|record| record.order == order) {
        record.status = status;
        record.bytes_sent = bytes_sent;
    }
    state.notify.notify_waiters();
}

fn parse_range(value: &str) -> Option<(u64, Option<u64>)> {
    let bytes_range = value.strip_prefix("bytes=")?;
    let (start, end) = bytes_range.split_once('-')?;
    let start = start.parse::<u64>().ok()?;
    let end = if end.is_empty() {
        None
    } else {
        Some(end.parse::<u64>().ok()?)
    };
    Some((start, end))
}

fn reason_phrase(status: u16) -> &'static str {
    match status {
        200 => "OK",
        206 => "Partial Content",
        302 => "Found",
        404 => "Not Found",
        416 => "Range Not Satisfiable",
        500 => "Internal Server Error",
        503 => "Service Unavailable",
        _ => "Mock Status",
    }
}

fn deterministic_bytes(size: usize) -> Vec<u8> {
    (0..size)
        .map(|byte_index| {
            let value = byte_index.wrapping_mul(31).wrapping_add(byte_index / 7);
            value as u8
        })
        .collect()
}

async fn wait_until<F, Fut>(
    timeout_duration: Duration,
    mut predicate: F,
) where
    F: FnMut() -> Fut,
    Fut: std::future::Future<Output = bool>,
{
    let result = tokio::time::timeout(timeout_duration, async {
        loop {
            if predicate().await {
                return;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await;
    assert!(result.is_ok(), "mock server wait timed out after {:?}", timeout_duration);
}

async fn wait_until_value<F, Fut, T>(
    timeout_duration: Duration,
    mut value: F,
) -> T
where
    F: FnMut() -> Fut,
    Fut: std::future::Future<Output = Option<T>>,
{
    tokio::time::timeout(timeout_duration, async {
        loop {
            if let Some(result) = value().await {
                return result;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .unwrap_or_else(|_| panic!("mock server wait timed out after {:?}", timeout_duration))
}
