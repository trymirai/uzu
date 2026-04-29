use std::{path::Path, time::Duration};

use download_manager_v2::{
    FileCheck, FileDownloadManager, FileDownloadPhase, backends::universal::UniversalDownloadManager,
};
use tempfile::tempdir;
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::TcpListener,
    time::timeout,
};

#[tokio::test]
async fn test_universal_manager_downloads_file_to_destination() {
    let body = b"hello from v2";
    let source_url = spawn_one_response_http_server(body).await;
    let temporary_directory = tempdir().unwrap();
    let destination = temporary_directory.path().join("download.bin");
    let manager = UniversalDownloadManager::new("test-manager".to_string());
    let task = manager
        .file_download_task(&source_url, Path::new(&destination), FileCheck::None, Some(body.len() as u64))
        .await
        .unwrap();

    task.download().await.unwrap();
    timeout(Duration::from_secs(5), task.wait()).await.unwrap();

    let state = task.state().await;
    assert_eq!(state.phase, FileDownloadPhase::Downloaded);
    assert_eq!(tokio::fs::read(destination).await.unwrap(), body);
}

async fn spawn_one_response_http_server(body: &'static [u8]) -> String {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();

    tokio::spawn(async move {
        for _ in 0..4 {
            let (mut stream, _) = listener.accept().await.unwrap();
            let mut request_buffer = [0_u8; 1024];
            let bytes_read = stream.read(&mut request_buffer).await.unwrap();
            let request = String::from_utf8_lossy(&request_buffer[..bytes_read]);
            let is_head = request.starts_with("HEAD ");
            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nAccept-Ranges: bytes\r\nConnection: close\r\n\r\n",
                body.len()
            );
            stream.write_all(response.as_bytes()).await.unwrap();
            if !is_head {
                stream.write_all(body).await.unwrap();
            }
            stream.shutdown().await.unwrap();
        }
    });

    format!("http://{address}/file")
}
