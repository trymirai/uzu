use std::{collections::HashMap, fs, path::PathBuf};

use futures::{StreamExt, channel::mpsc};
use gpui::Context;
use uzu::{
    storage::types::{DownloadPhase, DownloadState},
    types::{basic::ImageTheme, model::Model},
};

use crate::engine;

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum ModelKind {
    Chat,

    CloudChat,
    Classification,
    TextToSpeech,
}

impl ModelKind {
    fn matches(
        self,
        model: &Model,
    ) -> bool {
        match self {
            ModelKind::Chat => model.is_chat_capable() && model.is_local(),
            ModelKind::CloudChat => model.is_chat_capable() && model.is_remote(),
            ModelKind::Classification => model.is_classification_capable(),
            ModelKind::TextToSpeech => model.is_text_to_speech_capable(),
        }
    }
}

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

    pub fn icon_url(
        &self,
        prefer_dark: bool,
    ) -> Option<String> {
        let icons = &self.model.family.as_ref()?.vendor.metadata.icons;
        let want = if prefer_dark {
            ImageTheme::Dark
        } else {
            ImageTheme::Light
        };
        let is_svg = |i: &&uzu::types::basic::Image| i.url.ends_with(".svg");
        icons
            .iter()
            .find(|i| i.theme == want && is_svg(i))
            .or_else(|| icons.iter().find(is_svg))
            .or_else(|| icons.iter().find(|i| i.theme == want))
            .or_else(|| icons.first())
            .map(|i| i.url.clone())
    }

    pub fn size_bytes(&self) -> i64 {
        self.model.properties.as_ref().map(|p| p.size).unwrap_or(0)
    }

    pub fn display_size_bytes(&self) -> i64 {
        self.state.as_ref().map(|s| s.total_bytes).filter(|b| *b > 0).unwrap_or_else(|| self.size_bytes())
    }

    pub fn phase(&self) -> DownloadPhase {
        self.state.as_ref().map(|s| s.phase.clone()).unwrap_or(DownloadPhase::NotDownloaded {})
    }

    pub fn progress(&self) -> f32 {
        self.state.as_ref().map(|s| s.progress()).unwrap_or(0.0)
    }

    pub fn is_installed(&self) -> bool {
        matches!(self.phase(), DownloadPhase::Downloaded {}) || (self.model.is_local() && !self.model.is_downloadable())
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
        crate::persistence::mirai_data_dir().join("installed-at.json")
    }

    fn save_installed_at(map: &HashMap<String, u64>) {
        if fs::create_dir_all(crate::persistence::mirai_data_dir()).is_ok()
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
        let task = gpui_tokio::Tokio::spawn_result(cx, async move {
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

        gpui_tokio::Tokio::spawn(cx, async move {
            let mut stream = engine.storage_subscribe().await;
            while let Some(item) = stream.next().await {
                if let Ok(event) = item {
                    if tx.unbounded_send(event).is_err() {
                        break;
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
                                store.installed_at.insert(id.clone(), crate::persistence::now_ms());
                                Self::save_installed_at(&store.installed_at);
                            } else if was_installed && !now_installed {
                                store.installed_at.remove(&id);
                                Self::save_installed_at(&store.installed_at);
                            }
                            cx.notify();
                            if !was_installed && now_installed {
                                crate::toast::push(cx, format!("{name} downloaded"), crate::toast::ToastKind::Success);
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
        gpui_tokio::Tokio::spawn(cx, async move {
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
        gpui_tokio::Tokio::spawn(cx, async move {
            let _ = engine.downloader(&model).delete().await;
        })
        .detach();
    }
}

fn load_installed_at() -> HashMap<String, u64> {
    let path = ModelsStore::installed_at_path();
    fs::read_to_string(path).ok().and_then(|s| serde_json::from_str(&s).ok()).unwrap_or_default()
}
