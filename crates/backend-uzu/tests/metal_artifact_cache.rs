#![cfg(all(feature = "metal", target_os = "macos"))]
#![allow(dead_code)]

use std::{
    collections::HashMap,
    fs,
    path::PathBuf,
    sync::atomic::{AtomicU64, Ordering},
    time::{Duration, Instant},
};

#[path = "../build/common/caching.rs"]
mod caching;

#[path = "../build/metal/cache_protocol.rs"]
mod cache_protocol;

use cache_protocol::{SharedArtifactCache, Stage, air_key, index_key, metallib_key, zstd_key};

static SCRATCH_SEQ: AtomicU64 = AtomicU64::new(0);

fn scratch() -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "uzu-metal-cache-test-{}-{}",
        std::process::id(),
        SCRATCH_SEQ.fetch_add(1, Ordering::Relaxed)
    ));
    fs::create_dir_all(&dir).unwrap();
    dir
}

fn sample_deps() -> HashMap<Box<str>, [u8; blake3::OUT_LEN]> {
    let mut deps = HashMap::new();
    deps.insert("a.metal".into(), *blake3::hash(b"a").as_bytes());
    deps.insert("b.h".into(), *blake3::hash(b"b").as_bytes());
    deps
}

#[test]
fn air_key_ignores_zstd_level() {
    let compile = blake3::hash(b"compile");
    let toolchain = blake3::hash(b"toolchain");
    let deps = sample_deps();
    let air = air_key("gemm/kernel.metal", &deps, "footer", &compile, &toolchain);
    let again = air_key("gemm/kernel.metal", &deps, "footer", &compile, &toolchain);
    assert_eq!(air, again);

    let other_footer = air_key("gemm/kernel.metal", &deps, "footer2", &compile, &toolchain);
    assert_ne!(air, other_footer);
}

#[test]
fn zstd_key_changes_with_level_not_air() {
    let air_hash = blake3::hash(b"air-bytes");
    let linker = blake3::hash(b"linker");
    let metallib = metallib_key(&air_hash, &linker);
    let z20 = zstd_key(&metallib, 20);
    let z22 = zstd_key(&metallib, 22);
    assert_ne!(z20, z22);
    assert_eq!(metallib, metallib_key(&air_hash, &linker));
}

#[test]
fn index_key_tracks_source_and_schemas() {
    let source = blake3::hash(b"source");
    let compile = blake3::hash(b"compile");
    let analyzer = blake3::hash(b"analyzer");
    let gpu = blake3::hash(b"gpu");
    let a = index_key("gemm/kernel.metal", &source, &compile, &analyzer, &gpu);
    let b = index_key("gemm/kernel.metal", &blake3::hash(b"source2"), &compile, &analyzer, &gpu);
    let c = index_key("gemm/kernel.metal", &source, &blake3::hash(b"compile2"), &analyzer, &gpu);
    let d = index_key("gemm/kernel.metal", &source, &compile, &blake3::hash(b"analyzer2"), &gpu);
    assert_ne!(a, b);
    assert_ne!(a, c);
    assert_ne!(a, d);
}

#[test]
fn store_lookup_roundtrip() {
    let cache = SharedArtifactCache::at(scratch()).unwrap();
    let key = blake3::hash(b"roundtrip");
    cache.store_bytes(Stage::Air, &key, b"air-bytes").unwrap();
    let artifact = cache.lookup(Stage::Air, &key).unwrap().expect("artifact");
    assert_eq!(artifact.hash, *blake3::hash(b"air-bytes").as_bytes());
    assert_eq!(fs::read(artifact.path).unwrap(), b"air-bytes");
}

#[test]
fn lookup_rejects_missing_manifest() {
    let cache = SharedArtifactCache::at(scratch()).unwrap();
    let key = blake3::hash(b"no-manifest");
    let dir = cache.stage_dir(Stage::Air, &key);
    fs::create_dir_all(&dir).unwrap();
    fs::write(dir.join("artifact"), b"air-bytes").unwrap();
    assert!(cache.lookup(Stage::Air, &key).unwrap().is_none());
}

#[test]
fn lookup_rejects_corrupt_artifact() {
    let cache = SharedArtifactCache::at(scratch()).unwrap();
    let key = blake3::hash(b"corrupt");
    cache.store_bytes(Stage::Metallib, &key, b"good").unwrap();
    let artifact = cache.stage_dir(Stage::Metallib, &key).join("artifact");
    fs::write(artifact, b"tampered").unwrap();
    assert!(cache.lookup(Stage::Metallib, &key).unwrap().is_none());
}

#[test]
fn lookup_rejects_wrong_stage_kind() {
    let cache = SharedArtifactCache::at(scratch()).unwrap();
    let key = blake3::hash(b"kind");
    cache.store_bytes(Stage::Air, &key, b"air").unwrap();
    let manifest = cache.stage_dir(Stage::Air, &key).join("manifest.json");
    let bytes = fs::read(&manifest).unwrap();
    let mut value: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    value["kind"] = serde_json::Value::String("zstd".into());
    fs::write(manifest, serde_json::to_vec(&value).unwrap()).unwrap();
    assert!(cache.lookup(Stage::Air, &key).unwrap().is_none());
}

#[tokio::test(flavor = "current_thread")]
async fn lock_serializes_producers() {
    let root = scratch();
    let cache = SharedArtifactCache::at(&root).unwrap();
    let key = blake3::hash(b"lock");
    let held = cache.lock_with_timeout(Stage::Zstd, &key, Duration::from_secs(5)).await.unwrap();
    let started = Instant::now();
    let waiter = {
        let cache = SharedArtifactCache::at(&root).unwrap();
        tokio::spawn(async move {
            let _lock = cache.lock_with_timeout(Stage::Zstd, &key, Duration::from_secs(5)).await.unwrap();
            started.elapsed()
        })
    };
    tokio::time::sleep(Duration::from_millis(150)).await;
    drop(held);
    let waited = waiter.await.unwrap();
    assert!(waited >= Duration::from_millis(100), "waiter elapsed {waited:?}");
}

#[tokio::test(flavor = "current_thread")]
async fn concurrent_miss_reuses_first_store() {
    let root = scratch();
    let key = blake3::hash(b"race");
    let producer = {
        let cache = SharedArtifactCache::at(&root).unwrap();
        tokio::spawn(async move {
            let _lock = cache.lock_with_timeout(Stage::Air, &key, Duration::from_secs(5)).await.unwrap();
            if cache.lookup(Stage::Air, &key).unwrap().is_none() {
                tokio::time::sleep(Duration::from_millis(120)).await;
                cache.store_bytes(Stage::Air, &key, b"first").unwrap();
            }
        })
    };
    tokio::time::sleep(Duration::from_millis(20)).await;
    let cache = SharedArtifactCache::at(&root).unwrap();
    let _lock = cache.lock_with_timeout(Stage::Air, &key, Duration::from_secs(5)).await.unwrap();
    let artifact = cache.lookup(Stage::Air, &key).unwrap().expect("second producer must see first store");
    assert_eq!(fs::read(artifact.path).unwrap(), b"first");
    drop(_lock);
    producer.await.unwrap();
}

#[tokio::test(flavor = "current_thread")]
async fn malformed_lock_owner_is_reclaimed() {
    let cache = SharedArtifactCache::at(scratch()).unwrap();
    let key = blake3::hash(b"malformed-owner");
    let lock_dir = cache.stage_dir(Stage::Air, &key).join(".lock");
    fs::create_dir_all(&lock_dir).unwrap();
    fs::write(lock_dir.join("pid"), "not-a-pid").unwrap();

    let _lock = cache.lock_with_timeout(Stage::Air, &key, Duration::from_secs(1)).await.unwrap();
}
