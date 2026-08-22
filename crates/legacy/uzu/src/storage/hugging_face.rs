use std::{
    collections::HashSet,
    fmt, io,
    path::{Path, PathBuf},
    time::Duration,
};

use reqwest::{
    Client, StatusCode, Url,
    header::{AUTHORIZATION, HeaderMap, HeaderValue, LINK},
};
use serde::{Deserialize, Serialize};
use shoji::types::basic::Repository;

const DEFAULT_BASE_URL: &str = "https://huggingface.co/";
const CACHE_SCHEMA_VERSION: u8 = 1;
static CACHE_WRITE_LOCK: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

/// Resolves a Hugging Face model reference into an immutable, commit-pinned file list.
pub(crate) struct HuggingFaceResolver {
    client: Client,
    base_url: Url,
    cache_root: PathBuf,
    authorization: Option<HeaderValue>,
}

impl fmt::Debug for HuggingFaceResolver {
    fn fmt(
        &self,
        formatter: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        formatter
            .debug_struct("HuggingFaceResolver")
            .field("base_url", &self.base_url)
            .field("cache_root", &self.cache_root)
            .field("has_authorization", &self.authorization.is_some())
            .finish()
    }
}

impl HuggingFaceResolver {
    pub(crate) fn new(
        cache_root: PathBuf,
        bearer_token: Option<String>,
    ) -> Result<Self, HuggingFaceResolverError> {
        let client = Client::builder()
            .connect_timeout(Duration::from_secs(10))
            .timeout(Duration::from_secs(30))
            .build()
            .map_err(HuggingFaceResolverError::Client)?;
        let base_url = Url::parse(DEFAULT_BASE_URL).map_err(|_| HuggingFaceResolverError::InvalidBaseUrl)?;
        Self::with_base_url(client, base_url, cache_root, bearer_token)
    }

    fn with_base_url(
        client: Client,
        base_url: Url,
        cache_root: PathBuf,
        bearer_token: Option<String>,
    ) -> Result<Self, HuggingFaceResolverError> {
        if !base_url.has_host() || !matches!(base_url.scheme(), "http" | "https") {
            return Err(HuggingFaceResolverError::InvalidBaseUrl);
        }

        let authorization = bearer_token.map(bearer_header_value).transpose()?;
        Ok(Self {
            client,
            base_url,
            cache_root,
            authorization,
        })
    }

    pub(crate) async fn resolve_repository(
        &self,
        repository: &Repository,
    ) -> Result<ResolvedHuggingFaceRepository, HuggingFaceResolverError> {
        let repository_segments = validate_repository_id(&repository.identifier)?;
        let commit = self.resolve_commit(&repository_segments, repository.commit_hash.as_deref()).await?;

        let cached_tree = match self.read_cache(&repository.identifier, &commit).await {
            Ok(Some(tree)) => tree,
            Ok(None) | Err(HuggingFaceResolverError::InvalidCache) => {
                let tree = self.fetch_tree(&repository.identifier, &repository_segments, &commit).await?;
                self.write_cache(&tree).await?;
                tree
            },
            Err(error) => return Err(error),
        };

        self.materialize(cached_tree)
    }

    async fn resolve_commit(
        &self,
        repository_segments: &[&str],
        revision: Option<&str>,
    ) -> Result<String, HuggingFaceResolverError> {
        if let Some(revision) = revision
            && is_full_commit(revision)
        {
            return Ok(revision.to_ascii_lowercase());
        }

        let mut segments = vec!["api", "models"];
        segments.extend(repository_segments.iter().copied());
        if let Some(revision) = revision {
            if revision.is_empty() {
                return Err(HuggingFaceResolverError::InvalidRevision);
            }
            segments.extend(["revision", revision]);
        }

        let url = self.url_with_segments(&segments)?;
        let response: ModelInfoResponse = self.get_json(url, "model info").await?;
        if !is_full_commit(&response.sha) {
            return Err(HuggingFaceResolverError::InvalidCommit(response.sha));
        }
        Ok(response.sha.to_ascii_lowercase())
    }

    async fn fetch_tree(
        &self,
        repository_id: &str,
        repository_segments: &[&str],
        commit: &str,
    ) -> Result<CachedTree, HuggingFaceResolverError> {
        let mut segments = vec!["api", "models"];
        segments.extend(repository_segments.iter().copied());
        segments.extend(["tree", commit]);
        let mut next_url = self.url_with_segments(&segments)?;
        next_url.query_pairs_mut().append_pair("recursive", "true");

        let mut visited_pages = HashSet::new();
        let mut files = Vec::new();
        loop {
            if !visited_pages.insert(next_url.as_str().to_owned()) {
                return Err(HuggingFaceResolverError::PaginationCycle);
            }

            let (entries, headers): (Vec<TreeEntry>, HeaderMap) =
                self.get_json_with_headers(next_url.clone(), "repository tree").await?;
            for entry in entries {
                match entry.kind.as_str() {
                    "directory" => {},
                    "file" => files.push(CachedFile::try_from(entry)?),
                    kind => return Err(HuggingFaceResolverError::UnsupportedTreeEntry(kind.to_owned())),
                }
            }

            let Some(candidate) = next_page_url(&headers, &next_url)? else {
                break;
            };
            if candidate.origin() != self.base_url.origin() {
                return Err(HuggingFaceResolverError::CrossOriginPagination);
            }
            next_url = candidate;
        }

        files.sort_by(|left, right| left.relative_path.cmp(&right.relative_path));
        let mut paths = HashSet::with_capacity(files.len());
        for file in &files {
            if !paths.insert(file.relative_path.clone()) {
                return Err(HuggingFaceResolverError::DuplicatePath(file.relative_path.clone()));
            }
        }

        Ok(CachedTree {
            schema_version: CACHE_SCHEMA_VERSION,
            repository_id: repository_id.to_owned(),
            commit: commit.to_owned(),
            files,
        })
    }

    fn materialize(
        &self,
        tree: CachedTree,
    ) -> Result<ResolvedHuggingFaceRepository, HuggingFaceResolverError> {
        if tree.schema_version != CACHE_SCHEMA_VERSION
            || !is_full_commit(&tree.commit)
            || validate_repository_id(&tree.repository_id).is_err()
        {
            return Err(HuggingFaceResolverError::InvalidCache);
        }

        let repository_segments = validate_repository_id(&tree.repository_id)?;
        let mut files = Vec::with_capacity(tree.files.len());
        let mut seen_paths = HashSet::with_capacity(tree.files.len());
        for cached_file in tree.files {
            let path_segments = validate_relative_path(&cached_file.relative_path)?;
            if !seen_paths.insert(cached_file.relative_path.clone()) {
                return Err(HuggingFaceResolverError::InvalidCache);
            }
            cached_file.digest.validate()?;

            let mut url_segments = repository_segments.clone();
            url_segments.extend(["resolve", tree.commit.as_str()]);
            url_segments.extend(path_segments);
            let source_url = self.url_with_segments(&url_segments)?.to_string();
            files.push(ResolvedHuggingFaceFile {
                relative_path: PathBuf::from(cached_file.relative_path),
                source_url,
                size: cached_file.size,
                digest: cached_file.digest,
            });
        }

        Ok(ResolvedHuggingFaceRepository {
            commit: tree.commit,
            files,
            authorization: self.authorization.clone(),
        })
    }

    async fn get_json<T: for<'de> Deserialize<'de>>(
        &self,
        url: Url,
        operation: &'static str,
    ) -> Result<T, HuggingFaceResolverError> {
        self.get_json_with_headers(url, operation).await.map(|(value, _)| value)
    }

    async fn get_json_with_headers<T: for<'de> Deserialize<'de>>(
        &self,
        url: Url,
        operation: &'static str,
    ) -> Result<(T, HeaderMap), HuggingFaceResolverError> {
        let mut request = self.client.get(url);
        if let Some(authorization) = &self.authorization {
            request = request.header(AUTHORIZATION, authorization.clone());
        }
        let response = request.send().await.map_err(|source| HuggingFaceResolverError::Request {
            operation,
            source,
        })?;
        let status = response.status();
        if !status.is_success() {
            return Err(HuggingFaceResolverError::HttpStatus {
                operation,
                status,
            });
        }
        let headers = response.headers().clone();
        let value = response.json().await.map_err(|source| HuggingFaceResolverError::Request {
            operation,
            source,
        })?;
        Ok((value, headers))
    }

    fn url_with_segments(
        &self,
        segments: &[&str],
    ) -> Result<Url, HuggingFaceResolverError> {
        let mut url = self.base_url.clone();
        let mut path = url.path_segments_mut().map_err(|_| HuggingFaceResolverError::InvalidBaseUrl)?;
        path.pop_if_empty();
        path.extend(segments.iter().copied());
        drop(path);
        Ok(url)
    }

    async fn read_cache(
        &self,
        repository_id: &str,
        commit: &str,
    ) -> Result<Option<CachedTree>, HuggingFaceResolverError> {
        let path = self.cache_path(repository_id, commit);
        self.validate_cache_path(&path).await?;
        let bytes = match tokio::fs::read(&path).await {
            Ok(bytes) => bytes,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(source) => {
                return Err(HuggingFaceResolverError::CacheIo {
                    path,
                    source,
                });
            },
        };
        let tree: CachedTree = serde_json::from_slice(&bytes).map_err(|_| HuggingFaceResolverError::InvalidCache)?;
        if tree.repository_id != repository_id || tree.commit != commit || !tree.is_valid() {
            return Err(HuggingFaceResolverError::InvalidCache);
        }
        Ok(Some(tree))
    }

    async fn write_cache(
        &self,
        tree: &CachedTree,
    ) -> Result<(), HuggingFaceResolverError> {
        use tokio::io::AsyncWriteExt;

        let path = self.cache_path(&tree.repository_id, &tree.commit);
        let parent = path.parent().ok_or(HuggingFaceResolverError::InvalidCache)?.to_path_buf();
        self.ensure_cache_directory(&parent).await?;
        let bytes = serde_json::to_vec(tree).map_err(HuggingFaceResolverError::CacheJson)?;
        let temporary_path = path.with_extension(format!("json.tmp-{}", uuid::Uuid::new_v4()));
        let mut temporary_file =
            tokio::fs::OpenOptions::new().write(true).create_new(true).open(&temporary_path).await.map_err(
                |source| HuggingFaceResolverError::CacheIo {
                    path: temporary_path.clone(),
                    source,
                },
            )?;
        if let Err(source) = temporary_file.write_all(&bytes).await {
            let _ = tokio::fs::remove_file(&temporary_path).await;
            return Err(HuggingFaceResolverError::CacheIo {
                path: temporary_path,
                source,
            });
        }
        if let Err(source) = temporary_file.flush().await {
            let _ = tokio::fs::remove_file(&temporary_path).await;
            return Err(HuggingFaceResolverError::CacheIo {
                path: temporary_path,
                source,
            });
        }
        if let Err(source) = temporary_file.sync_all().await {
            let _ = tokio::fs::remove_file(&temporary_path).await;
            return Err(HuggingFaceResolverError::CacheIo {
                path: temporary_path,
                source,
            });
        }
        drop(temporary_file);

        let _write_guard = CACHE_WRITE_LOCK.lock().await;
        match self.read_cache(&tree.repository_id, &tree.commit).await {
            Ok(Some(_)) => {
                let _ = tokio::fs::remove_file(&temporary_path).await;
                return Ok(());
            },
            Ok(None) | Err(HuggingFaceResolverError::InvalidCache) => {},
            Err(error) => {
                let _ = tokio::fs::remove_file(&temporary_path).await;
                return Err(error);
            },
        }

        self.validate_cache_path(&parent).await?;
        match replace_cache_file(&temporary_path, &path).await {
            Ok(()) => Ok(()),
            Err(source) => {
                let existing_cache = self.read_cache(&tree.repository_id, &tree.commit).await;
                let _ = tokio::fs::remove_file(&temporary_path).await;
                match existing_cache {
                    Ok(Some(_)) => Ok(()),
                    Ok(None) | Err(HuggingFaceResolverError::InvalidCache) => Err(HuggingFaceResolverError::CacheIo {
                        path,
                        source,
                    }),
                    Err(error) => Err(error),
                }
            },
        }
    }

    fn cache_path(
        &self,
        repository_id: &str,
        commit: &str,
    ) -> PathBuf {
        self.cache_root.join(encode_cache_component(repository_id)).join(format!("{commit}.json"))
    }

    async fn ensure_cache_directory(
        &self,
        path: &Path,
    ) -> Result<(), HuggingFaceResolverError> {
        self.validate_cache_path(path).await?;
        tokio::fs::create_dir_all(path).await.map_err(|source| HuggingFaceResolverError::CacheIo {
            path: path.to_path_buf(),
            source,
        })?;
        self.validate_cache_path(path).await
    }

    async fn validate_cache_path(
        &self,
        path: &Path,
    ) -> Result<(), HuggingFaceResolverError> {
        reject_symlink_components(path).await.map_err(|source| HuggingFaceResolverError::CacheIo {
            path: path.to_path_buf(),
            source,
        })
    }
}

#[cfg(not(windows))]
async fn replace_cache_file(
    source: &Path,
    destination: &Path,
) -> io::Result<()> {
    tokio::fs::rename(source, destination).await
}

#[cfg(windows)]
async fn replace_cache_file(
    source: &Path,
    destination: &Path,
) -> io::Result<()> {
    let source = source.to_path_buf();
    let destination = destination.to_path_buf();
    kiban::rt::run_blocking(move || replace_cache_file_sync(&source, &destination)).await
}

#[cfg(windows)]
fn replace_cache_file_sync(
    source: &Path,
    destination: &Path,
) -> io::Result<()> {
    use std::{iter, os::windows::ffi::OsStrExt};

    use windows_sys::Win32::Storage::FileSystem::{MOVEFILE_REPLACE_EXISTING, MOVEFILE_WRITE_THROUGH, MoveFileExW};

    let source = source.as_os_str().encode_wide().chain(iter::once(0)).collect::<Vec<_>>();
    let destination = destination.as_os_str().encode_wide().chain(iter::once(0)).collect::<Vec<_>>();
    // SAFETY: both path buffers are null-terminated and remain alive for the duration of the call.
    if unsafe { MoveFileExW(source.as_ptr(), destination.as_ptr(), MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH) }
        == 0
    {
        Err(io::Error::last_os_error())
    } else {
        Ok(())
    }
}

async fn reject_symlink_components(path: &Path) -> io::Result<()> {
    let mut current = PathBuf::new();
    for component in path.components() {
        current.push(component);
        match tokio::fs::symlink_metadata(&current).await {
            Ok(metadata) if metadata.file_type().is_symlink() && !is_platform_path_alias(&current) => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("Hugging Face cache path contains a symlink: {}", current.display()),
                ));
            },
            Ok(_) => {},
            Err(error) if error.kind() == io::ErrorKind::NotFound => break,
            Err(error) => return Err(error),
        }
    }
    Ok(())
}

fn is_platform_path_alias(path: &Path) -> bool {
    #[cfg(target_os = "macos")]
    {
        matches!(path.to_str(), Some("/var" | "/tmp" | "/etc"))
    }
    #[cfg(not(target_os = "macos"))]
    {
        let _ = path;
        false
    }
}

fn bearer_header_value(token: String) -> Result<HeaderValue, HuggingFaceResolverError> {
    if token.is_empty() {
        return Err(HuggingFaceResolverError::InvalidBearerToken);
    }
    let mut value =
        HeaderValue::from_str(&format!("Bearer {token}")).map_err(|_| HuggingFaceResolverError::InvalidBearerToken)?;
    value.set_sensitive(true);
    Ok(value)
}

#[derive(Clone, Debug)]
pub(crate) struct ResolvedHuggingFaceRepository {
    pub(crate) commit: String,
    pub(crate) files: Vec<ResolvedHuggingFaceFile>,
    pub(crate) authorization: Option<HeaderValue>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ResolvedHuggingFaceFile {
    pub(crate) relative_path: PathBuf,
    pub(crate) source_url: String,
    pub(crate) size: u64,
    pub(crate) digest: HuggingFaceDigest,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", content = "value", rename_all = "snake_case")]
pub(crate) enum HuggingFaceDigest {
    Sha256(String),
    GitBlobSha1(String),
}

impl HuggingFaceDigest {
    fn validate(&self) -> Result<(), HuggingFaceResolverError> {
        let (value, length) = match self {
            Self::Sha256(value) => (value, 64),
            Self::GitBlobSha1(value) => (value, 40),
        };
        if is_lower_hex(value, length) {
            Ok(())
        } else {
            Err(HuggingFaceResolverError::InvalidDigest)
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub(crate) enum HuggingFaceResolverError {
    #[error("invalid Hugging Face base URL")]
    InvalidBaseUrl,
    #[error("invalid Hugging Face repository identifier: {0}")]
    InvalidRepositoryId(String),
    #[error("invalid Hugging Face revision")]
    InvalidRevision,
    #[error("Hugging Face returned an invalid commit: {0}")]
    InvalidCommit(String),
    #[error("Hugging Face returned an unsafe path: {0}")]
    UnsafePath(String),
    #[error("Hugging Face returned a duplicate path: {0}")]
    DuplicatePath(String),
    #[error("Hugging Face returned an invalid digest")]
    InvalidDigest,
    #[error("Hugging Face returned file metadata without a size")]
    MissingSize,
    #[error("unsupported Hugging Face tree entry type: {0}")]
    UnsupportedTreeEntry(String),
    #[error("Hugging Face pagination contains a cycle")]
    PaginationCycle,
    #[error("Hugging Face pagination attempted to change origin")]
    CrossOriginPagination,
    #[error("invalid Hugging Face Link header")]
    InvalidLinkHeader,
    #[error("invalid Hugging Face bearer token")]
    InvalidBearerToken,
    #[error("unable to create Hugging Face HTTP client")]
    Client(#[source] reqwest::Error),
    #[error("Hugging Face {operation} request failed")]
    Request {
        operation: &'static str,
        #[source]
        source: reqwest::Error,
    },
    #[error("Hugging Face {operation} returned HTTP {status}")]
    HttpStatus {
        operation: &'static str,
        status: StatusCode,
    },
    #[error("Hugging Face cache I/O failed at {path}")]
    CacheIo {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("Hugging Face cache serialization failed")]
    CacheJson(#[source] serde_json::Error),
    #[error("invalid Hugging Face cache entry")]
    InvalidCache,
}

#[derive(Deserialize)]
struct ModelInfoResponse {
    sha: String,
}

#[derive(Deserialize)]
struct TreeEntry {
    #[serde(rename = "type")]
    kind: String,
    path: String,
    #[serde(default)]
    size: Option<u64>,
    #[serde(default, alias = "blobId", alias = "blob_id")]
    oid: Option<String>,
    #[serde(default)]
    lfs: Option<LfsMetadata>,
}

#[derive(Deserialize)]
struct LfsMetadata {
    #[serde(alias = "oid")]
    sha256: String,
    #[serde(default)]
    size: Option<u64>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct CachedTree {
    schema_version: u8,
    repository_id: String,
    commit: String,
    files: Vec<CachedFile>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct CachedFile {
    relative_path: String,
    size: u64,
    digest: HuggingFaceDigest,
}

impl TryFrom<TreeEntry> for CachedFile {
    type Error = HuggingFaceResolverError;

    fn try_from(entry: TreeEntry) -> Result<Self, Self::Error> {
        validate_relative_path(&entry.path)?;
        let (size, digest) = match entry.lfs {
            Some(lfs) => {
                let oid = lfs.sha256.strip_prefix("sha256:").unwrap_or(&lfs.sha256).to_ascii_lowercase();
                let digest = HuggingFaceDigest::Sha256(oid);
                digest.validate()?;
                (lfs.size.or(entry.size).ok_or(HuggingFaceResolverError::MissingSize)?, digest)
            },
            None => {
                let oid = entry.oid.ok_or(HuggingFaceResolverError::InvalidDigest)?.to_ascii_lowercase();
                let digest = HuggingFaceDigest::GitBlobSha1(oid);
                digest.validate()?;
                (entry.size.ok_or(HuggingFaceResolverError::MissingSize)?, digest)
            },
        };
        Ok(Self {
            relative_path: entry.path,
            size,
            digest,
        })
    }
}

impl CachedTree {
    fn is_valid(&self) -> bool {
        if self.schema_version != CACHE_SCHEMA_VERSION
            || !is_full_commit(&self.commit)
            || validate_repository_id(&self.repository_id).is_err()
        {
            return false;
        }

        let mut paths = HashSet::with_capacity(self.files.len());
        self.files.iter().all(|file| {
            validate_relative_path(&file.relative_path).is_ok()
                && file.digest.validate().is_ok()
                && paths.insert(file.relative_path.clone())
        })
    }
}

fn validate_repository_id(repository_id: &str) -> Result<Vec<&str>, HuggingFaceResolverError> {
    let segments = repository_id.split('/').collect::<Vec<_>>();
    let valid_shape = matches!(segments.len(), 1 | 2)
        && (1..=96).contains(&repository_id.len())
        && !repository_id.contains("--")
        && !repository_id.contains("..")
        && !repository_id.ends_with(".git")
        && segments.iter().all(|segment| is_valid_repository_segment(segment));
    if !valid_shape {
        return Err(HuggingFaceResolverError::InvalidRepositoryId(repository_id.to_owned()));
    }
    Ok(segments)
}

fn is_valid_repository_segment(segment: &str) -> bool {
    let valid_edge = |character: char| character.is_ascii_alphanumeric() || character == '_';
    segment.chars().next().is_some_and(valid_edge)
        && segment.chars().next_back().is_some_and(valid_edge)
        && segment.chars().all(|character| character.is_ascii_alphanumeric() || matches!(character, '-' | '_' | '.'))
}

fn validate_relative_path(path: &str) -> Result<Vec<&str>, HuggingFaceResolverError> {
    let segments = path.split('/').collect::<Vec<_>>();
    if segments.is_empty() || segments.iter().any(|segment| !is_safe_segment(segment)) {
        return Err(HuggingFaceResolverError::UnsafePath(path.to_owned()));
    }
    Ok(segments)
}

fn is_safe_segment(segment: &str) -> bool {
    !segment.is_empty()
        && !matches!(segment, "." | "..")
        && !segment.contains('\\')
        && !segment.contains('\0')
        && !segment.contains(':')
}

fn is_full_commit(value: &str) -> bool {
    is_hex(value, 40)
}

fn is_lower_hex(
    value: &str,
    length: usize,
) -> bool {
    value.len() == length && value.bytes().all(|byte| byte.is_ascii_digit() || matches!(byte, b'a'..=b'f'))
}

fn is_hex(
    value: &str,
    length: usize,
) -> bool {
    value.len() == length && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn encode_cache_component(value: &str) -> String {
    let mut encoded = String::with_capacity(value.len());
    for byte in value.bytes() {
        if byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.') {
            encoded.push(char::from(byte));
        } else {
            use std::fmt::Write;
            let _ = write!(encoded, "%{byte:02X}");
        }
    }
    encoded
}

fn next_page_url(
    headers: &HeaderMap,
    current_url: &Url,
) -> Result<Option<Url>, HuggingFaceResolverError> {
    for value in headers.get_all(LINK) {
        let value = value.to_str().map_err(|_| HuggingFaceResolverError::InvalidLinkHeader)?;
        let mut remainder = value;
        while let Some(open) = remainder.find('<') {
            remainder = &remainder[open + 1..];
            let close = remainder.find('>').ok_or(HuggingFaceResolverError::InvalidLinkHeader)?;
            let target = &remainder[..close];
            remainder = &remainder[close + 1..];
            let next_entry = remainder.find(',').unwrap_or(remainder.len());
            let parameters = &remainder[..next_entry];
            let is_next = parameters.split(';').any(|parameter| {
                let Some((name, value)) = parameter.trim().split_once('=') else {
                    return false;
                };
                name.eq_ignore_ascii_case("rel")
                    && value.trim_matches('"').split_ascii_whitespace().any(|relation| relation == "next")
            });
            if is_next {
                return current_url.join(target).map(Some).map_err(|_| HuggingFaceResolverError::InvalidLinkHeader);
            }
            remainder = remainder.get(next_entry.saturating_add(1)..).unwrap_or_default();
        }
    }
    Ok(None)
}

#[cfg(test)]
#[path = "../../tests/unit/storage/hugging_face_test.rs"]
mod tests;
