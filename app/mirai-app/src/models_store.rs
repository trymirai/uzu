//! Shared model catalog + live download state as a GPUI entity. Loads models
//! from the engine, follows its download broadcast, and exposes download /
//! pause / resume / delete. Views hold an `Entity<ModelsStore>` and observe it.

use futures::{StreamExt, channel::mpsc};
use gpui::Context;
use std::{collections::HashMap, fs, path::PathBuf};
use uzu::{
    storage::types::{DownloadPhase, DownloadState},
    types::{basic::ImageTheme, model::Model},
};

use crate::engine;

/// Which capability a store lists.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum ModelKind {
    /// Local (downloadable) chat models.
    Chat,
    /// Remote/cloud chat models (loaded when a provider key is configured).
    CloudChat,
    Classification,
    TextToSpeech,
}

impl ModelKind {
    fn matches(self, model: &Model) -> bool {
        match self {
            ModelKind::Chat => model.is_chat_capable() && model.is_local(),
            ModelKind::CloudChat => model.is_chat_capable() && model.is_remote(),
            ModelKind::Classification => model.is_classification_capable(),
            ModelKind::TextToSpeech => model.is_text_to_speech_capable(),
        }
    }
}

/// One catalog entry: the full uzu `Model` (kept so actions can call the
/// downloader) plus its latest download state.
pub struct ModelRow {
    pub model: Model,
    pub state: Option<DownloadState>,
}

impl ModelRow {
    pub fn id(&self) -> &str {
        &self.model.identifier
    }

    pub fn name(&self) -> String {
        self.model.name()
    }

    pub fn vendor(&self) -> Option<String> {
        self.model.family.as_ref().map(|f| f.vendor.name())
    }

    /// Remote provider/family logo URL from the model's metadata, preferring the
    /// given theme. `None` when the family has no icons.
    pub fn icon_url(&self, prefer_dark: bool) -> Option<String> {
        let icons = &self.model.family.as_ref()?.vendor.metadata.icons;
        let want = if prefer_dark { ImageTheme::Dark } else { ImageTheme::Light };
        icons
            .iter()
            .find(|i| i.theme == want)
            .or_else(|| icons.first())
            .map(|i| i.url.clone())
    }

    pub fn size_bytes(&self) -> i64 {
        self.model
            .properties
            .as_ref()
            .map(|p| p.size)
            .unwrap_or(0)
    }

    pub fn phase(&self) -> DownloadPhase {
        self.state
            .as_ref()
            .map(|s| s.phase.clone())
            .unwrap_or(DownloadPhase::NotDownloaded {})
    }

    pub fn progress(&self) -> f32 {
        self.state.as_ref().map(|s| s.progress()).unwrap_or(0.0)
    }

    pub fn is_installed(&self) -> bool {
        matches!(self.phase(), DownloadPhase::Downloaded {})
    }

    pub fn is_downloading(&self) -> bool {
        matches!(self.phase(), DownloadPhase::Downloading {})
    }
}

pub struct ModelsStore {
    kind: ModelKind,
    pub rows: Vec<ModelRow>,
    pub loading: bool,
    pub error: Option<String>,
    installed_at: HashMap<String, u64>,
}

impl ModelsStore {
    pub fn new(kind: ModelKind, cx: &mut Context<Self>) -> Self {
        Self::spawn_load(kind, cx);
        Self::spawn_watch(cx);
        Self {
            kind,
            rows: Vec::new(),
            loading: true,
            error: None,
            installed_at: load_installed_at(),
        }
    }

    pub fn installed_at(&self, id: &str) -> u64 {
        self.installed_at.get(id).copied().unwrap_or(0)
    }

    fn installed_at_path() -> PathBuf {
        crate::persistence::mirai_data_dir().join("installed-at.json")
    }

    fn save_installed_at(map: &HashMap<String, u64>) {
        if fs::create_dir_all(crate::persistence::mirai_data_dir()).is_ok()
            && let Ok(json) = serde_json::to_string(map)
        {
            let _ = fs::write(Self::installed_at_path(), json);
        }
    }

    /// Re-fetch the catalog from the engine (e.g. after adding a cloud provider).
    pub fn reload(&mut self, cx: &mut Context<Self>) {
        self.loading = true;
        self.error = None;
        cx.notify();
        Self::spawn_load(self.kind, cx);
    }

    /// Loads the matching model catalog + initial download states.
    fn spawn_load(kind: ModelKind, cx: &mut Context<Self>) {
        let Some(engine) = engine::try_engine(cx) else {
            return;
        };
        let task = gpui_tokio::Tokio::spawn_result(cx, async move {
            let models = engine.models().await?;
            let states = engine.download_states().await;
            let rows: Vec<ModelRow> = models
                .into_iter()
                .filter(|m| kind.matches(m))
                .map(|model| {
                    let state = states.get(&model.identifier).cloned();
                    ModelRow { model, state }
                })
                .collect();
            anyhow::Ok(rows)
        });
        cx.spawn(async move |this, cx| {
            let result = task.await;
            this.update(cx, |store, cx| {
                store.loading = false;
                match result {
                    Ok(rows) => {
                        eprintln!("[mirai-app] loaded {} models", rows.len());
                        store.rows = rows;
                        store.error = None;
                    }
                    Err(err) => store.error = Some(err.to_string()),
                }
                cx.notify();
            })
            .ok();
        })
        .detach();
    }

    /// Subscribes once to the engine's global download broadcast and folds each
    /// `(identifier, state)` update into the matching row.
    fn spawn_watch(cx: &mut Context<Self>) {
        let Some(engine) = engine::try_engine(cx) else {
            return;
        };
        let (tx, mut rx) = mpsc::unbounded::<(String, DownloadState)>();

        gpui_tokio::Tokio::spawn(cx, async move {
            let mut stream = engine.storage_subscribe().await;
            while let Some(item) = stream.next().await {
                if let Ok(event) = item {
                    if tx.unbounded_send(event).is_err() {
                        break; // store dropped
                    }
                }
            }
        })
        .detach();

        cx.spawn(async move |this, cx| {
            while let Some((id, state)) = rx.next().await {
                if this
                    .update(cx, |store, cx| {
                        if let Some(row) = store.rows.iter_mut().find(|r| r.id() == id) {
                            let was_installed = row.is_installed();
                            row.state = Some(state);
                            let now_installed = row.is_installed();
                            let name = row.name();
                            if !was_installed && now_installed {
                                store
                                    .installed_at
                                    .insert(id.clone(), crate::persistence::now_ms());
                                Self::save_installed_at(&store.installed_at);
                            } else if was_installed && !now_installed {
                                store.installed_at.remove(&id);
                                Self::save_installed_at(&store.installed_at);
                            }
                            cx.notify();
                            if !was_installed && now_installed {
                                crate::toast::push(
                                    cx,
                                    format!("{name} downloaded"),
                                    crate::toast::ToastKind::Success,
                                );
                            }
                        }
                    })
                    .is_err()
                {
                    break;
                }
            }
        })
        .detach();
    }

    /// Start a download, or pause/resume it depending on current phase.
    pub fn toggle_download(&mut self, id: String, cx: &mut Context<Self>) {
        let Some(engine) = engine::try_engine(cx) else {
            return;
        };
        let Some(model) = self.rows.iter().find(|r| r.id() == id).map(|r| r.model.clone()) else {
            return;
        };
        gpui_tokio::Tokio::spawn(cx, async move {
            let downloader = engine.downloader(&model);
            match downloader.state().await.map(|s| s.phase) {
                Some(DownloadPhase::Downloading {}) => {
                    let _ = downloader.pause().await;
                }
                // NotDownloaded / Paused / Locked / Error / unknown → (re)start.
                _ => {
                    let _ = downloader.resume().await;
                }
            }
        })
        .detach();
        // Progress + phase changes arrive via the storage broadcast watcher.
    }

    /// Delete (or cancel an in-flight download of) a model's on-disk files.
    pub fn delete(&mut self, id: String, cx: &mut Context<Self>) {
        let Some(engine) = engine::try_engine(cx) else {
            return;
        };
        let Some(model) = self.rows.iter().find(|r| r.id() == id).map(|r| r.model.clone()) else {
            return;
        };
        gpui_tokio::Tokio::spawn(cx, async move {
            let _ = engine.downloader(&model).delete().await;
        })
        .detach();
    }
}

fn load_installed_at() -> HashMap<String, u64> {
    let path = ModelsStore::installed_at_path();
    fs::read_to_string(path)
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_default()
}
