mod model_kind;
mod model_row;

use std::{collections::HashMap, fs, path::PathBuf};

use futures::{StreamExt, channel::mpsc};
use gpui::Context;
use gpui_tokio::Tokio;
pub use model_kind::ModelKind;
pub use model_row::ModelRow;
use uzu::{
    storage::types::{DownloadPhase, DownloadState},
    types::model::Model,
};

use crate::{
    engine, persistence,
    toast::{self, ToastKind},
};

pub struct ModelsStore {
    kind: ModelKind,
    pub rows: Vec<ModelRow>,
    pub loading: bool,
    pub error: Option<String>,
    installed_at: HashMap<String, u64>,
}

impl ModelsStore {
    pub fn new(
        kind: ModelKind,
        cx: &mut Context<Self>,
    ) -> Self {
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

    pub fn installed_at(
        &self,
        id: &str,
    ) -> u64 {
        self.installed_at.get(id).copied().unwrap_or(0)
    }

    pub fn resolve_installed(
        &self,
        selected: Option<&Model>,
    ) -> Option<Model> {
        if let Some(model) = selected
            && self.rows.iter().any(|r| r.model.identifier == model.identifier && r.is_installed())
        {
            return Some(model.clone());
        }
        self.rows.iter().find(|r| r.is_installed()).map(|r| r.model.clone())
    }

    pub fn installed_model_by_id(
        &self,
        id: &str,
    ) -> Option<Model> {
        self.rows.iter().find(|row| row.id() == id && row.is_installed()).map(|row| row.model.clone())
    }

    fn installed_at_path() -> PathBuf {
        persistence::mirai_data_dir().join("installed-at.json")
    }

    fn save_installed_at(map: &HashMap<String, u64>) {
        if fs::create_dir_all(persistence::mirai_data_dir()).is_ok()
            && let Ok(json) = serde_json::to_string(map)
        {
            let _ = fs::write(Self::installed_at_path(), json);
        }
    }

    pub fn reload(
        &mut self,
        cx: &mut Context<Self>,
    ) {
        self.loading = true;
        self.error = None;
        cx.notify();
        Self::spawn_load(self.kind, cx);
    }

    fn spawn_load(
        kind: ModelKind,
        cx: &mut Context<Self>,
    ) {
        let Some(engine) = engine::try_engine(cx) else {
            cx.spawn(async move |this, cx| {
                let _ = this.update(cx, |store, cx| {
                    store.loading = false;
                    store.error = Some("Engine unavailable".to_string());
                    cx.notify();
                });
            })
            .detach();
            return;
        };
        let task = Tokio::spawn_result(cx, async move {
            let models = engine.models().await?;
            let states = engine.download_states().await;
            let rows: Vec<ModelRow> = models
                .into_iter()
                .filter(|m| kind.matches(m))
                .map(|model| {
                    let state = states.get(&model.identifier).cloned();
                    ModelRow {
                        model,
                        state,
                    }
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
                        store.rows = rows;
                        store.error = None;
                    },
                    Err(err) => store.error = Some(err.to_string()),
                }
                cx.notify();
            })
            .ok();
        })
        .detach();
    }

    fn spawn_watch(cx: &mut Context<Self>) {
        let Some(engine) = engine::try_engine(cx) else {
            return;
        };
        let (tx, mut rx) = mpsc::unbounded::<(String, DownloadState)>();

        Tokio::spawn(cx, async move {
            let mut stream = engine.storage_subscribe().await;
            while let Some(item) = stream.next().await {
                if let Ok(event) = item
                    && tx.unbounded_send(event).is_err()
                {
                    break;
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
                            let just_installed = !was_installed && now_installed;
                            if just_installed {
                                store.installed_at.insert(id.clone(), persistence::now_ms());
                                Self::save_installed_at(&store.installed_at);
                            } else if was_installed && !now_installed {
                                store.installed_at.remove(&id);
                                Self::save_installed_at(&store.installed_at);
                            }
                            let downloaded_name = just_installed.then(|| row.name());
                            cx.notify();
                            if let Some(name) = downloaded_name {
                                toast::push(cx, format!("{name} downloaded"), ToastKind::Success);
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

    pub fn toggle_download(
        &mut self,
        id: String,
        cx: &mut Context<Self>,
    ) {
        let Some(engine) = engine::try_engine(cx) else {
            return;
        };
        let Some(model) = self.rows.iter().find(|r| r.id() == id).map(|r| r.model.clone()) else {
            return;
        };
        Tokio::spawn(cx, async move {
            let downloader = engine.downloader(&model);
            match downloader.state().await.map(|s| s.phase) {
                Some(DownloadPhase::Downloading {}) => {
                    let _ = downloader.pause().await;
                },

                _ => {
                    let _ = downloader.resume().await;
                },
            }
        })
        .detach();
    }

    pub fn delete(
        &mut self,
        id: String,
        cx: &mut Context<Self>,
    ) {
        let Some(engine) = engine::try_engine(cx) else {
            return;
        };
        let Some(model) = self.rows.iter().find(|r| r.id() == id).map(|r| r.model.clone()) else {
            return;
        };
        Tokio::spawn(cx, async move {
            let _ = engine.downloader(&model).delete().await;
        })
        .detach();
    }
}

fn load_installed_at() -> HashMap<String, u64> {
    let path = ModelsStore::installed_at_path();
    fs::read_to_string(path).ok().and_then(|s| serde_json::from_str(&s).ok()).unwrap_or_default()
}
