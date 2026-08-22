use std::{
    collections::HashMap,
    path::Path,
    sync::{Arc, Mutex},
};

use shoji::types::model::ModelReference;
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::TcpListener,
    task::JoinHandle,
};

use super::*;

const COMMIT: &str = "0123456789abcdef0123456789abcdef01234567";
const GIT_SHA1: &str = "1111111111111111111111111111111111111111";
const LFS_SHA256: &str = "2222222222222222222222222222222222222222222222222222222222222222";

#[test]
fn default_resolver_redacts_its_token() {
    let cache = tempfile::tempdir().unwrap();
    let resolver = HuggingFaceResolver::new(cache.path().to_path_buf(), Some("top-secret".to_owned())).unwrap();

    assert!(!format!("{resolver:?}").contains("top-secret"));
}

#[test]
fn repository_ids_match_the_official_one_or_two_component_shape() {
    assert!(validate_repository_id("model").is_ok());
    assert!(validate_repository_id("organization/model").is_ok());

    for invalid in ["organization/team/model", ".model", "model.git", "model--variant", "model name"] {
        assert!(validate_repository_id(invalid).is_err(), "{invalid} must be rejected");
    }
}

#[tokio::test]
async fn cache_write_keeps_an_existing_valid_immutable_tree() {
    let cache = tempfile::tempdir().unwrap();
    let resolver = HuggingFaceResolver::new(cache.path().to_path_buf(), None).unwrap();
    let first_tree = cached_tree("config.json");
    resolver.write_cache(&first_tree).await.unwrap();
    let path = resolver.cache_path("acme/model", COMMIT);
    let first_bytes = tokio::fs::read(&path).await.unwrap();

    resolver.write_cache(&cached_tree("replacement.json")).await.unwrap();

    assert_eq!(tokio::fs::read(path).await.unwrap(), first_bytes);
}

#[tokio::test]
async fn cache_write_replaces_an_invalid_entry_with_a_complete_tree() {
    let cache = tempfile::tempdir().unwrap();
    let resolver = HuggingFaceResolver::new(cache.path().to_path_buf(), None).unwrap();
    let path = resolver.cache_path("acme/model", COMMIT);
    tokio::fs::create_dir_all(path.parent().unwrap()).await.unwrap();
    tokio::fs::write(&path, b"incomplete json").await.unwrap();

    resolver.write_cache(&cached_tree("config.json")).await.unwrap();

    let cached = resolver.read_cache("acme/model", COMMIT).await.unwrap().unwrap();
    assert_eq!(cached.files[0].relative_path, "config.json");
}

#[tokio::test]
async fn concurrent_cache_writers_publish_one_complete_tree_without_temporary_files() {
    let cache = tempfile::tempdir().unwrap();
    let resolver = Arc::new(HuggingFaceResolver::new(cache.path().to_path_buf(), None).unwrap());
    let barrier = Arc::new(tokio::sync::Barrier::new(3));
    let mut tasks = Vec::new();
    for file_name in ["first.json", "second.json"] {
        let resolver = Arc::clone(&resolver);
        let barrier = Arc::clone(&barrier);
        tasks.push(tokio::spawn(async move {
            barrier.wait().await;
            resolver.write_cache(&cached_tree(file_name)).await
        }));
    }

    barrier.wait().await;
    for task in tasks {
        task.await.unwrap().unwrap();
    }

    let cached = resolver.read_cache("acme/model", COMMIT).await.unwrap().unwrap();
    assert!(matches!(cached.files[0].relative_path.as_str(), "first.json" | "second.json"));
    let cache_directory = resolver.cache_path("acme/model", COMMIT).parent().unwrap().to_path_buf();
    let mut entries = tokio::fs::read_dir(cache_directory).await.unwrap();
    let mut names = Vec::new();
    while let Some(entry) = entries.next_entry().await.unwrap() {
        names.push(entry.file_name());
    }
    assert_eq!(names, [std::ffi::OsString::from(format!("{COMMIT}.json"))]);
}

#[cfg(unix)]
#[tokio::test]
async fn cache_write_rejects_a_symlinked_repository_directory() {
    use std::os::unix::fs::symlink;

    let temporary = tempfile::tempdir().unwrap();
    let cache_root = temporary.path().join("cache");
    let outside = temporary.path().join("outside");
    tokio::fs::create_dir_all(&cache_root).await.unwrap();
    tokio::fs::create_dir_all(&outside).await.unwrap();
    let resolver = HuggingFaceResolver::new(cache_root, None).unwrap();
    let repository_cache = resolver.cache_path("acme/model", COMMIT).parent().unwrap().to_path_buf();
    symlink(&outside, &repository_cache).unwrap();

    let error = resolver.write_cache(&cached_tree("config.json")).await.unwrap_err();

    assert!(matches!(error, HuggingFaceResolverError::CacheIo { .. }));
    assert!(std::fs::read_dir(outside).unwrap().next().is_none());
}

#[tokio::test]
async fn resolves_revision_pagination_digests_and_cache_without_leaking_token() {
    let server = TestServer::start().await;
    server.respond("/api/models/acme/model/revision/main", TestResponse::json(format!(r#"{{"sha":"{COMMIT}"}}"#)));
    server.respond(
        &format!("/api/models/acme/model/tree/{COMMIT}?recursive=true"),
        TestResponse::json(format!(
            r#"[
                {{"type":"directory","path":"nested","size":0}},
                {{"type":"file","path":"config.json","size":12,"oid":"{GIT_SHA1}"}}
            ]"#
        ))
        .with_header("Link", format!("<{}page-2>; rel=\"next\"", server.base_url())),
    );
    server.respond(
        "/page-2",
        TestResponse::json(format!(
            r#"[{{"type":"file","path":"nested/model.safetensors","size":128,"oid":"unused","lfs":{{"sha256":"{LFS_SHA256}","size":4096,"pointer_size":134}}}}]"#
        )),
    );

    let cache = tempfile::tempdir().unwrap();
    let resolver = HuggingFaceResolver::with_base_url(
        Client::new(),
        Url::parse(&server.base_url()).unwrap(),
        cache.path().to_path_buf(),
        Some("top-secret".to_owned()),
    )
    .unwrap();
    let repository = Repository {
        identifier: "acme/model".to_owned(),
        commit_hash: Some("main".to_owned()),
        paths: None,
    };

    let resolved = resolver.resolve_repository(&repository).await.unwrap();

    assert_eq!(resolved.commit, COMMIT);
    assert_eq!(resolved.files.len(), 2);
    assert_eq!(resolved.files[0].relative_path, PathBuf::from("config.json"));
    assert_eq!(resolved.files[0].digest, HuggingFaceDigest::GitBlobSha1(GIT_SHA1.to_owned()));
    assert_eq!(resolved.files[1].relative_path, PathBuf::from("nested/model.safetensors"));
    assert_eq!(resolved.files[1].size, 4096);
    assert_eq!(resolved.files[1].digest, HuggingFaceDigest::Sha256(LFS_SHA256.to_owned()));
    assert!(resolved.files[1].source_url.ends_with(&format!("/acme/model/resolve/{COMMIT}/nested/model.safetensors")));
    assert_eq!(resolved.authorization.as_ref().unwrap().to_str().unwrap(), "Bearer top-secret");
    assert!(!format!("{resolver:?}{resolved:?}").contains("top-secret"));

    let requests = server.requests();
    assert_eq!(requests.len(), 3);
    assert!(requests.iter().all(|request| request.authorization.as_deref() == Some("Bearer top-secret")));

    let pinned = Repository {
        identifier: "acme/model".to_owned(),
        commit_hash: Some(COMMIT.to_owned()),
        paths: None,
    };
    let cached = resolver.resolve_repository(&pinned).await.unwrap();
    assert_eq!(cached.files, resolved.files);
    assert_eq!(server.requests().len(), 3, "a pinned immutable tree should come entirely from cache");

    let cache_contents = tokio::fs::read_to_string(resolver.cache_path("acme/model", COMMIT)).await.unwrap();
    assert!(!cache_contents.contains("top-secret"));
}

#[tokio::test]
async fn resolved_repository_downloads_through_the_shared_file_group() {
    const CONTENTS: &str = "hello world\n";
    const CONTENTS_GIT_SHA1: &str = "3b18e512dba79e4c8300dd08aeb37f8e728b8dad";

    let server = TestServer::start().await;
    server.respond(
        &format!("/api/models/acme/model/tree/{COMMIT}?recursive=true"),
        TestResponse::json(format!(
            r#"[{{"type":"file","path":"nested/config.json","size":{},"oid":"{CONTENTS_GIT_SHA1}"}}]"#,
            CONTENTS.len()
        )),
    );
    server.respond(&format!("/acme/model/resolve/{COMMIT}/nested/config.json"), TestResponse::json(CONTENTS));

    let temporary = tempfile::tempdir().unwrap();
    let resolver = resolver_for(&server, &temporary.path().join("trees"), None);
    let repository = Repository {
        identifier: "acme/model".to_owned(),
        commit_hash: Some(COMMIT.to_owned()),
        paths: None,
    };
    let model = shoji::types::model::Model::external(
        "hf-fixture".to_owned(),
        "registry".to_owned(),
        "Registry".to_owned(),
        "backend".to_owned(),
        "Backend".to_owned(),
        "1".to_owned(),
        Vec::new(),
        shoji::types::model::ModelAccessibility::Local {
            reference: ModelReference::HuggingFace {
                repository: repository.clone(),
            },
        },
        Vec::new(),
    );
    let config = crate::storage::Config::new(
        crate::device::Device {
            os_name: None,
            cpu_name: None,
            memory_total: 0,
            home_path: temporary.path().to_string_lossy().into_owned(),
        },
        Some(temporary.path().to_path_buf()),
        "hf-group-test".to_owned(),
    );

    let resolved = resolver.resolve_repository(&repository).await.unwrap();
    let download = crate::storage::build_hugging_face_download(&config, &model, resolved).unwrap();
    assert_eq!(download.group_spec.files().len(), 1);
    assert!(matches!(download.group_spec.files()[0].check, download_manager::FileCheck::GitBlobSha1(_)));

    let manager = Arc::from(
        <dyn download_manager::FileDownloadManager>::new(
            download_manager::FileDownloadManagerType::Universal,
            kiban::rt::RuntimeHandle::current(),
        )
        .await
        .unwrap(),
    );
    let group = download_manager::FileDownloadGroup::open(manager, download.group_spec).await.unwrap();
    let completed = group.download().await.unwrap().wait().await.unwrap();

    assert_eq!(completed.phase, download_manager::FileDownloadGroupPhase::Downloaded);
    assert_eq!(completed.completed_files, 1);
    assert_eq!(tokio::fs::read(download.cache_path.join("nested/config.json")).await.unwrap(), CONTENTS.as_bytes());
}

#[tokio::test]
async fn resolves_default_revision_through_model_info() {
    let server = TestServer::start().await;
    server.respond("/api/models/acme/model", TestResponse::json(format!(r#"{{"sha":"{COMMIT}"}}"#)));
    server.respond(
        &format!("/api/models/acme/model/tree/{COMMIT}?recursive=true"),
        TestResponse::json(format!(r#"[{{"type":"file","path":"config.json","size":12,"blobId":"{GIT_SHA1}"}}]"#)),
    );
    let cache = tempfile::tempdir().unwrap();
    let resolver = resolver_for(&server, cache.path(), None);
    let repository = Repository {
        identifier: "acme/model".to_owned(),
        commit_hash: None,
        paths: None,
    };

    let resolved = resolver.resolve_repository(&repository).await.unwrap();

    assert_eq!(resolved.commit, COMMIT);
    assert_eq!(
        server.requests().iter().map(|request| request.target.as_str()).collect::<Vec<_>>(),
        vec!["/api/models/acme/model", &format!("/api/models/acme/model/tree/{COMMIT}?recursive=true"),]
    );
}

#[tokio::test]
async fn rejects_unsafe_tree_paths() {
    let server = TestServer::start().await;
    server.respond(
        &format!("/api/models/acme/model/tree/{COMMIT}?recursive=true"),
        TestResponse::json(format!(
            r#"[{{"type":"file","path":"../escape.safetensors","size":12,"oid":"{GIT_SHA1}"}}]"#
        )),
    );
    let cache = tempfile::tempdir().unwrap();
    let resolver = resolver_for(&server, cache.path(), None);
    let repository = Repository {
        identifier: "acme/model".to_owned(),
        commit_hash: Some(COMMIT.to_owned()),
        paths: None,
    };

    let error = resolver.resolve_repository(&repository).await.unwrap_err();

    assert!(matches!(error, HuggingFaceResolverError::UnsafePath(path) if path == "../escape.safetensors"));
}

#[tokio::test]
async fn rejects_file_metadata_without_a_size() {
    let server = TestServer::start().await;
    server.respond(
        &format!("/api/models/acme/model/tree/{COMMIT}?recursive=true"),
        TestResponse::json(format!(r#"[{{"type":"file","path":"config.json","oid":"{GIT_SHA1}"}}]"#)),
    );
    let cache = tempfile::tempdir().unwrap();
    let resolver = resolver_for(&server, cache.path(), None);
    let repository = Repository {
        identifier: "acme/model".to_owned(),
        commit_hash: Some(COMMIT.to_owned()),
        paths: None,
    };

    let error = resolver.resolve_repository(&repository).await.unwrap_err();

    assert!(matches!(error, HuggingFaceResolverError::MissingSize));
}

#[tokio::test]
async fn gated_repository_errors_are_model_specific_and_keep_the_token_redacted() {
    let server = TestServer::start().await;
    server.respond("/api/models/acme/private/revision/main", TestResponse::status(403, r#"{"error":"gated"}"#));
    let cache = tempfile::tempdir().unwrap();
    let resolver = resolver_for(&server, cache.path(), Some("top-secret"));
    let repository = Repository {
        identifier: "acme/private".to_owned(),
        commit_hash: Some("main".to_owned()),
        paths: None,
    };

    let error = resolver.resolve_repository(&repository).await.unwrap_err();

    assert!(matches!(
        &error,
        HuggingFaceResolverError::HttpStatus {
            operation: "model info",
            status: StatusCode::FORBIDDEN,
        }
    ));
    assert!(!format!("{error:?}").contains("top-secret"));
    assert_eq!(server.requests()[0].authorization.as_deref(), Some("Bearer top-secret"));
}

#[tokio::test]
async fn rejects_cross_origin_pagination_before_forwarding_authorization() {
    let server = TestServer::start().await;
    server.respond(
        &format!("/api/models/acme/model/tree/{COMMIT}?recursive=true"),
        TestResponse::json("[]").with_header("Link", "<https://example.invalid/steal>; rel=\"next\""),
    );
    let cache = tempfile::tempdir().unwrap();
    let resolver = resolver_for(&server, cache.path(), Some("top-secret"));
    let repository = Repository {
        identifier: "acme/model".to_owned(),
        commit_hash: Some(COMMIT.to_owned()),
        paths: None,
    };

    let error = resolver.resolve_repository(&repository).await.unwrap_err();

    assert!(matches!(error, HuggingFaceResolverError::CrossOriginPagination));
    assert_eq!(server.requests().len(), 1);
}

#[tokio::test]
async fn metadata_redirect_does_not_forward_authorization_to_another_origin() {
    let redirected = TestServer::start().await;
    redirected.respond("/model-info", TestResponse::json(format!(r#"{{"sha":"{COMMIT}"}}"#)));

    let source = TestServer::start().await;
    source.respond(
        "/api/models/acme/model/revision/main",
        TestResponse::status(302, "").with_header("Location", format!("{}model-info", redirected.base_url())),
    );
    source.respond(
        &format!("/api/models/acme/model/tree/{COMMIT}?recursive=true"),
        TestResponse::json(format!(r#"[{{"type":"file","path":"config.json","size":12,"oid":"{GIT_SHA1}"}}]"#)),
    );
    let cache = tempfile::tempdir().unwrap();
    let resolver = resolver_for(&source, cache.path(), Some("top-secret"));
    let repository = Repository {
        identifier: "acme/model".to_owned(),
        commit_hash: Some("main".to_owned()),
        paths: None,
    };

    let resolved = resolver.resolve_repository(&repository).await.unwrap();

    assert_eq!(resolved.commit, COMMIT);
    assert_eq!(redirected.requests().len(), 1);
    assert_eq!(redirected.requests()[0].authorization, None);
    assert!(source.requests().iter().all(|request| request.authorization.as_deref() == Some("Bearer top-secret")));
}

fn resolver_for(
    server: &TestServer,
    cache_path: &Path,
    token: Option<&str>,
) -> HuggingFaceResolver {
    HuggingFaceResolver::with_base_url(
        Client::new(),
        Url::parse(&server.base_url()).unwrap(),
        cache_path.to_path_buf(),
        token.map(str::to_owned),
    )
    .unwrap()
}

fn cached_tree(file_name: &str) -> CachedTree {
    CachedTree {
        schema_version: CACHE_SCHEMA_VERSION,
        repository_id: "acme/model".to_owned(),
        commit: COMMIT.to_owned(),
        files: vec![CachedFile {
            relative_path: file_name.to_owned(),
            size: 12,
            digest: HuggingFaceDigest::GitBlobSha1(GIT_SHA1.to_owned()),
        }],
    }
}

#[derive(Clone, Debug)]
struct RecordedRequest {
    target: String,
    authorization: Option<String>,
}

struct TestServer {
    address: std::net::SocketAddr,
    responses: Arc<Mutex<HashMap<String, TestResponse>>>,
    requests: Arc<Mutex<Vec<RecordedRequest>>>,
    task: JoinHandle<()>,
}

impl TestServer {
    async fn start() -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let responses = Arc::new(Mutex::new(HashMap::<String, TestResponse>::new()));
        let requests = Arc::new(Mutex::new(Vec::new()));
        let task_responses = Arc::clone(&responses);
        let task_requests = Arc::clone(&requests);
        let task = tokio::spawn(async move {
            loop {
                let Ok((mut stream, _)) = listener.accept().await else {
                    break;
                };
                let mut request_bytes = vec![0_u8; 16 * 1024];
                let Ok(bytes_read) = stream.read(&mut request_bytes).await else {
                    continue;
                };
                let request = String::from_utf8_lossy(&request_bytes[..bytes_read]);
                let mut lines = request.lines();
                let target =
                    lines.next().and_then(|line| line.split_ascii_whitespace().nth(1)).unwrap_or_default().to_owned();
                let authorization = lines.find_map(|line| {
                    let (name, value) = line.split_once(':')?;
                    name.eq_ignore_ascii_case("authorization").then(|| value.trim().to_owned())
                });
                task_requests.lock().unwrap().push(RecordedRequest {
                    target: target.clone(),
                    authorization,
                });
                let response = task_responses.lock().unwrap().get(&target).cloned().unwrap_or_else(|| TestResponse {
                    status: 404,
                    headers: Vec::new(),
                    body: "not found".to_owned(),
                });
                let reason = if response.status == 200 {
                    "OK"
                } else {
                    "Not Found"
                };
                let mut wire = format!(
                    "HTTP/1.1 {} {}\r\nContent-Length: {}\r\nContent-Type: application/json\r\nConnection: close\r\n",
                    response.status,
                    reason,
                    response.body.len()
                );
                for (name, value) in response.headers {
                    wire.push_str(&format!("{name}: {value}\r\n"));
                }
                wire.push_str("\r\n");
                wire.push_str(&response.body);
                let _ = stream.write_all(wire.as_bytes()).await;
            }
        });
        Self {
            address,
            responses,
            requests,
            task,
        }
    }

    fn base_url(&self) -> String {
        format!("http://{}/", self.address)
    }

    fn respond(
        &self,
        target: &str,
        response: TestResponse,
    ) {
        self.responses.lock().unwrap().insert(target.to_owned(), response);
    }

    fn requests(&self) -> Vec<RecordedRequest> {
        self.requests.lock().unwrap().clone()
    }
}

impl Drop for TestServer {
    fn drop(&mut self) {
        self.task.abort();
    }
}

#[derive(Clone)]
struct TestResponse {
    status: u16,
    headers: Vec<(String, String)>,
    body: String,
}

impl TestResponse {
    fn json(body: impl Into<String>) -> Self {
        Self {
            status: 200,
            headers: Vec::new(),
            body: body.into(),
        }
    }

    fn status(
        status: u16,
        body: impl Into<String>,
    ) -> Self {
        Self {
            status,
            headers: Vec::new(),
            body: body.into(),
        }
    }

    fn with_header(
        mut self,
        name: impl Into<String>,
        value: impl Into<String>,
    ) -> Self {
        self.headers.push((name.into(), value.into()));
        self
    }
}
