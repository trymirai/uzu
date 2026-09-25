const STAGING_PREFIX: &str = "mirai-update-";

pub(super) fn staging_path(version: &str) -> std::path::PathBuf {
    // pid-scoped so two instances staging the same version can't clobber each other.
    let pid = std::process::id();
    std::env::temp_dir().join(format!("{STAGING_PREFIX}{version}-{pid}.tar.gz"))
}

// Age gate so a peer's in-flight file survives; only crash-orphans get swept.
const STAGING_MAX_AGE: std::time::Duration = std::time::Duration::from_secs(24 * 60 * 60);

pub(super) fn clean_stale_staging() {
    let Ok(entries) = std::fs::read_dir(std::env::temp_dir()) else {
        return;
    };
    for entry in entries.flatten() {
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if !(name.starts_with(STAGING_PREFIX) && name.ends_with(".tar.gz")) {
            continue;
        }
        let old_enough = entry
            .metadata()
            .and_then(|m| m.modified())
            .ok()
            .and_then(|t| t.elapsed().ok())
            .is_some_and(|age| age > STAGING_MAX_AGE);
        if old_enough {
            let _ = std::fs::remove_file(entry.path());
        }
    }
}
