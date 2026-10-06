mod authorization;

pub use authorization::AuthError;
use serde::Serialize;
use tauri::{AppHandle, Manager};

use crate::error::{AppError, AppResult};

// Installs a /usr/local/bin/mirai wrapper resolving the CLI bundled in app
// resources. Writing there needs admin rights; a custom Authorization Services
// right (not system.privilege.admin) lets macOS remember the grant per app.

const WRAPPER_PATH: &str = "/usr/local/bin/mirai";
const WRAPPER_MARKER: &str = "# mirai-cli-wrapper";

const PROMPT_TEXT: &str = "Mirai is trying to install its command line interface (CLI) tool.";
const RIGHT_SUFFIX: &str = "cli-symlink";
// The wrapper resolves this one bundle; a build with another identifier owning
// /usr/local/bin/mirai would point it at the wrong app.
const BUNDLE_ID: &str = "com.mirai.tech.chat";

fn wrapper_script() -> String {
    format!(
        r#"#!/bin/bash
{WRAPPER_MARKER}
set -euo pipefail

APP_CANDIDATES=(
  "/Applications/Mirai.app"
)
FOUND_APP="$(/usr/bin/mdfind "kMDItemCFBundleIdentifier == '{BUNDLE_ID}'" | /usr/bin/grep -m1 -E '\.app$' || true)"
if [ -n "${{FOUND_APP}}" ]; then
  APP_CANDIDATES+=("${{FOUND_APP}}")
fi

for APP_PATH in "${{APP_CANDIDATES[@]}}"; do
  [ -d "${{APP_PATH}}" ] || continue
  CLI="${{APP_PATH}}/Contents/Resources/cli/mirai"
  if [ -x "${{CLI}}" ]; then
    exec "${{CLI}}" "$@"
  fi
  # Scoped to the bundle so it can't exec a stray 'mirai'.
  CLI="$(/usr/bin/find "${{APP_PATH}}/Contents" -type f -name 'mirai' -perm -111 -print -quit 2>/dev/null || true)"
  if [ -n "${{CLI}}" ]; then
    exec "${{CLI}}" "$@"
  fi
done

echo "mirai: Mirai CLI not found. Please reinstall Mirai." 1>&2
exit 1
"#
    )
}

// The wrapper execs the CLI shipped in app resources; a bundle built without it
// has nothing to install. Debug runs resolve resources to target/debug, where
// tauri-build leaves copies from earlier configs, so the file check alone is
// not enough there.
fn available(app: &AppHandle) -> bool {
    !cfg!(debug_assertions)
        && app.config().identifier == BUNDLE_ID
        && app.path().resource_dir().is_ok_and(|dir| dir.join("cli/mirai").is_file())
}

#[derive(Debug, PartialEq)]
enum WrapperState {
    Current,
    Missing,
    // A /usr/local/bin/mirai the app didn't write; leave it alone.
    Foreign,
}

fn wrapper_state() -> WrapperState {
    wrapper_state_at(std::path::Path::new(WRAPPER_PATH))
}

fn wrapper_state_at(path: &std::path::Path) -> WrapperState {
    match std::fs::read(path) {
        // The wrapper locates the CLI at run time, so an existing one needs no
        // reinstall.
        Ok(bytes) if String::from_utf8_lossy(&bytes).contains(WRAPPER_MARKER) => WrapperState::Current,
        Ok(_) => WrapperState::Foreign,
        // `read` follows symlinks, so NotFound alone may be a dangling link that
        // is still somebody's; only a path with no entry at all is missing.
        Err(error) if error.kind() == std::io::ErrorKind::NotFound && std::fs::symlink_metadata(path).is_err() => {
            WrapperState::Missing
        },
        Err(_) => WrapperState::Foreign,
    }
}

fn install_wrapper() -> AppResult<()> {
    let tmp = std::env::temp_dir().join("mirai-cli-wrapper.sh");
    std::fs::write(&tmp, wrapper_script())?;
    let sh_command =
        format!("mkdir -p /usr/local/bin && /usr/bin/install -m 755 '{}' '{}'", tmp.display(), WRAPPER_PATH);
    let result = authorization::run_privileged(BUNDLE_ID, RIGHT_SUFFIX, PROMPT_TEXT, &sh_command);
    let _ = std::fs::remove_file(&tmp);
    result?;
    if wrapper_state() != WrapperState::Current {
        return Err(AppError::msg("install command did not produce the wrapper"));
    }
    Ok(())
}

fn install_declined() -> bool {
    crate::storage::settings_load()
        .ok()
        .and_then(|s| s.get("cliInstallDeclined").and_then(|v| v.as_bool()))
        .unwrap_or(false)
}

fn persist_declined(declined: bool) {
    if let Err(e) = crate::storage::settings_patch(serde_json::json!({ "cliInstallDeclined": declined })) {
        crate::logger::warn(
            "cli-installer:settings-write-failed",
            Some(serde_json::json!({ "declined": declined, "error": e })),
        );
    }
}

#[derive(Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum CliStatus {
    Installed,
    Foreign,
    Missing,
    Unavailable,
}

#[derive(Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum CliInstalled {
    Installed,
    AlreadyInstalled,
    Cancelled,
}

#[tauri::command]
pub fn cli_status(app: AppHandle) -> CliStatus {
    if !available(&app) {
        return CliStatus::Unavailable;
    }
    match wrapper_state() {
        WrapperState::Current => CliStatus::Installed,
        WrapperState::Foreign => CliStatus::Foreign,
        WrapperState::Missing => CliStatus::Missing,
    }
}

#[tauri::command]
pub async fn cli_install(app: AppHandle) -> AppResult<CliInstalled> {
    if !available(&app) {
        return Err(AppError::msg("CLI install is unavailable in this build"));
    }
    match wrapper_state() {
        WrapperState::Current => Ok(CliInstalled::AlreadyInstalled),
        WrapperState::Foreign => {
            Err(AppError::msg(format!("{WRAPPER_PATH} is occupied by a binary Mirai didn't install")))
        },
        WrapperState::Missing => {
            crate::logger::info("cli-installer:prompt", Some(serde_json::json!({ "reason": "settings" })));
            let result = tokio::task::spawn_blocking(install_wrapper).await.map_err(AppError::msg)?;
            match &result {
                Ok(()) => {
                    persist_declined(false);
                    crate::logger::info("cli-installer:installed", Some(serde_json::json!({ "source": "settings" })));
                },
                Err(AppError::Authorization(AuthError::Cancelled)) => {
                    crate::logger::info("cli-installer:declined", Some(serde_json::json!({ "source": "settings" })));
                },
                Err(e) => {
                    crate::logger::warn(
                        "cli-installer:failed",
                        Some(serde_json::json!({ "source": "settings", "error": e })),
                    );
                },
            }
            match result {
                Ok(()) => Ok(CliInstalled::Installed),
                Err(AppError::Authorization(AuthError::Cancelled)) => Ok(CliInstalled::Cancelled),
                Err(e) => Err(e),
            }
        },
    }
}

pub fn trigger_if_needed(app: &AppHandle) {
    if !available(app) {
        return;
    }
    tauri::async_runtime::spawn(async move {
        // Let the window come up before the admin prompt appears.
        tokio::time::sleep(std::time::Duration::from_secs(3)).await;
        match wrapper_state() {
            WrapperState::Current => {
                crate::logger::info("cli-installer:skip", Some(serde_json::json!({ "reason": "installed" })));
            },
            WrapperState::Foreign => {
                crate::logger::warn("cli-installer:skip", Some(serde_json::json!({ "reason": "foreign-binary" })));
            },
            WrapperState::Missing => {
                if install_declined() {
                    crate::logger::info("cli-installer:skip", Some(serde_json::json!({ "reason": "declined" })));
                    return;
                }
                crate::logger::info("cli-installer:prompt", Some(serde_json::json!({ "reason": "install" })));
                match tokio::task::spawn_blocking(install_wrapper).await {
                    Ok(Ok(())) => crate::logger::info("cli-installer:installed", None),
                    Ok(Err(AppError::Authorization(AuthError::Cancelled))) => {
                        crate::logger::info("cli-installer:declined", None);
                        persist_declined(true);
                    },
                    Ok(Err(e)) => crate::logger::warn("cli-installer:failed", Some(serde_json::json!({ "error": e }))),
                    Err(e) => {
                        crate::logger::warn("cli-installer:failed", Some(serde_json::json!({ "error": e.to_string() })))
                    },
                }
            },
        }
    });
}

#[cfg(test)]
mod tests {
    use super::{WRAPPER_MARKER, WrapperState, wrapper_state_at};

    fn scratch(
        name: &str,
        bytes: Option<&[u8]>,
    ) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("mirai-cli-installer-{}-{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("mirai");
        if let Some(bytes) = bytes {
            std::fs::write(&path, bytes).unwrap();
        }
        path
    }

    #[test]
    fn missing_only_when_the_file_does_not_exist() {
        assert_eq!(wrapper_state_at(&scratch("missing", None)), WrapperState::Missing);
    }

    #[test]
    fn own_wrapper_is_current() {
        let path = scratch("current", Some(format!("#!/bin/sh\n{WRAPPER_MARKER}\n").as_bytes()));
        assert_eq!(wrapper_state_at(&path), WrapperState::Current);
    }

    #[test]
    fn a_text_file_without_the_marker_is_foreign() {
        assert_eq!(wrapper_state_at(&scratch("text", Some(b"#!/bin/sh\necho hi\n"))), WrapperState::Foreign);
    }

    #[test]
    fn a_dangling_symlink_is_foreign_not_missing() {
        let path = scratch("dangling", None);
        std::os::unix::fs::symlink("/nonexistent/mirai-target", &path).unwrap();
        assert_eq!(wrapper_state_at(&path), WrapperState::Foreign);
    }

    #[test]
    fn a_binary_that_is_not_utf8_is_foreign_not_missing() {
        let path = scratch("binary", Some(&[0xcf, 0xfa, 0xed, 0xfe, 0xff, 0x00, 0xfe, 0x80]));
        assert_eq!(wrapper_state_at(&path), WrapperState::Foreign);
    }
}
