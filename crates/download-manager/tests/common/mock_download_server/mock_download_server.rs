use std::{
    collections::HashMap,
    io::Result as IoResult,
    net::SocketAddr,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
    time::Duration,
};

use tokio::{
    io::AsyncWriteExt,
    net::{TcpListener, TcpStream},
    spawn as tokio_spawn,
    sync::{Mutex as TokioMutex, Notify, RwLock},
    task::JoinHandle,
    time::sleep as tokio_sleep,
};

use crate::common::mock_download_server::{
    FilePayload, RegistryFixture, RequestRecord, RouteBehavior,
    http_request::{HttpRequest, read_request},
    mock_route::MockRoute,
    server_state::ServerState,
    utils::{reason_phrase, wait_until, wait_until_value},
};

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
            json_routes: RwLock::new(HashMap::new()),
            records: TokioMutex::new(Vec::new()),
            bytes_sent_by_path: TokioMutex::new(HashMap::new()),
            request_counts_by_route: TokioMutex::new(HashMap::new()),
            stall_notifies_by_path: TokioMutex::new(HashMap::new()),
            next_order: AtomicU64::new(1),
        });

        let server_state = state.clone();
        let server_task = tokio_spawn(async move {
            loop {
                let Ok((stream, _peer_address)) = listener.accept().await else {
                    break;
                };
                let connection_state = server_state.clone();
                tokio_spawn(async move {
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

    pub async fn serve_file(
        &self,
        payload: FilePayload,
        behavior: RouteBehavior,
    ) {
        let path = payload.path();
        self.state.routes.write().await.insert(
            path,
            MockRoute {
                payload,
                behavior,
            },
        );
    }

    pub async fn serve_registry_fixture(
        &self,
        fixture: &RegistryFixture,
        behavior: RouteBehavior,
    ) {
        for payload in fixture.payloads() {
            self.serve_file(payload, behavior.clone()).await;
        }
    }

    pub async fn serve_json(
        &self,
        path: &str,
        json: String,
    ) {
        self.state.json_routes.write().await.insert(path.to_string(), json);
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
) -> IoResult<()> {
    let Some(request) = read_request(&mut stream).await? else {
        return Ok(());
    };

    if let Some(json) = state.json_routes.read().await.get(&request.path).cloned() {
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
        let bytes_sent = respond_json(&mut stream, &json).await?;
        update_record(&state, order, 200, bytes_sent).await;
        return Ok(());
    }

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
    let request_count = {
        let mut request_counts = state.request_counts_by_route.lock().await;
        let request_count = request_counts.entry((request.path.clone(), request.method.clone())).or_insert(0);
        *request_count += 1;
        *request_count
    };

    let bytes_sent = match &route.behavior {
        RouteBehavior::RetryThenOk {
            failures,
            status,
        } if request_count <= *failures => respond_empty(&mut stream, *status, "retryable failure").await?,
        _ => respond_with_file(&mut stream, &state, &request, &route).await?,
    };

    update_record(&state, order, bytes_sent.0, bytes_sent.1).await;
    Ok(())
}

async fn respond_json(
    stream: &mut TcpStream,
    json: &str,
) -> IoResult<u64> {
    let body = json.as_bytes();
    let headers = format!(
        "HTTP/1.1 200 OK\r\nConnection: close\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n",
        body.len()
    );
    stream.write_all(headers.as_bytes()).await?;
    stream.write_all(body).await?;
    Ok(body.len() as u64)
}

async fn respond_with_file(
    stream: &mut TcpStream,
    state: &Arc<ServerState>,
    request: &HttpRequest,
    route: &MockRoute,
) -> IoResult<(u16, u64)> {
    let source_bytes = match route.behavior {
        RouteBehavior::CorruptBody => route.payload.corrupt_bytes(),
        _ => route.payload.bytes.clone(),
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
        RouteBehavior::WrongContentLength => (body.len() as u64 / 2).max(1),
        _ => body.len() as u64,
    };
    let mut header_lines = vec![
        format!("HTTP/1.1 {} {}\r\n", status, reason_phrase(status)),
        "Connection: close\r\n".to_string(),
        "Content-Type: application/octet-stream\r\n".to_string(),
        format!("Last-Modified: {}\r\n", route.payload.last_modified),
    ];
    if range_supported {
        header_lines.push("Accept-Ranges: bytes\r\n".to_string());
    }
    header_lines.push(format!("Content-Length: {}\r\n", advertised_length));
    if status == 206 {
        let content_range =
            format!("Content-Range: bytes {}-{}/{}\r\n", body_start, body_end_exclusive.saturating_sub(1), file_size);
        header_lines.push(content_range);
    }
    header_lines.push("\r\n".to_string());
    stream.write_all(header_lines.concat().as_bytes()).await?;

    if request.method == "HEAD" {
        return Ok((status, 0));
    }

    let bytes_sent = write_body(stream, state, &route.payload.path(), body_start, body, &route.behavior).await?;
    Ok((status, bytes_sent))
}

async fn write_body(
    stream: &mut TcpStream,
    state: &Arc<ServerState>,
    path: &str,
    body_start: u64,
    body: &[u8],
    behavior: &RouteBehavior,
) -> IoResult<u64> {
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
                tokio_sleep(Duration::from_millis(*delay_ms)).await;
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
) -> IoResult<(u16, u64)> {
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

async fn respond_range_not_satisfiable(
    stream: &mut TcpStream,
    file_size: u64,
) -> IoResult<u64> {
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
}
