use std::{collections::HashMap, sync::Arc};

use crossterm::event::KeyCode;
use futures_util::StreamExt;
use ratatui::widgets::ListState;
use shoji::types::model::{Model, ModelIdentifier};
use tokio::{sync::Mutex as TokioMutex, task::JoinHandle};
use uzu::{
    engine::Engine,
    storage::{DownloadPhase, DownloadState},
};

use super::{events::AppEvent, models::ModelOrganizer, sections::Section};

/// Holds a Model handle and its cached state for UI rendering
#[derive(Clone)]
pub struct ModelWithState {
    pub model: Model,
    pub state: DownloadState,
}

pub struct App {
    pub engine: Arc<Engine>,
    pub models: Arc<TokioMutex<HashMap<ModelIdentifier, ModelWithState>>>,
    pub active_section: Section,
    pub list_states: HashMap<Section, ListState>,
    pub should_quit: bool,
    state_listener_handle: Arc<TokioMutex<Option<JoinHandle<()>>>>,
    tokio_handle: tokio::runtime::Handle,
}

impl App {
    pub async fn new(
        engine: Arc<Engine>,
        tokio_handle: tokio::runtime::Handle,
    ) -> Self {
        let models =
            engine.models().await.unwrap().into_iter().filter(|model| model.is_downloadable()).collect::<Vec<_>>();

        // Fetch initial state for all models
        let mut models_with_state = HashMap::new();
        for model in models {
            let state = engine.downloader(&model).state().await.unwrap();
            models_with_state.insert(
                model.identifier.clone(),
                ModelWithState {
                    model,
                    state,
                },
            );
        }

        let mut list_states = HashMap::new();
        for section in Section::all() {
            list_states.insert(section, ListState::default());
        }

        Self {
            engine,
            models: Arc::new(TokioMutex::new(models_with_state)),
            active_section: Section::Available,
            list_states,
            should_quit: false,
            state_listener_handle: Arc::new(TokioMutex::new(None)),
            tokio_handle,
        }
    }

    /// Spawn a background task that listens to model state updates
    pub async fn spawn_state_listener(&mut self) {
        let models = Arc::clone(&self.models);
        let engine = Arc::clone(&self.engine);
        let mut updates = self.engine.storage_subscribe();

        let handle = self.tokio_handle.spawn(async move {
            while let Some(Ok((model_id, state))) = updates.next().await {
                let mut models_guard = models.lock().await;

                if let Some(model_with_state) = models_guard.get_mut(&model_id) {
                    // Update the cached state
                    model_with_state.state = state;
                } else {
                    // Model not in local HashMap; fetch from storage and add it
                    drop(models_guard);
                    if let Some(fresh_model) = engine.model_by_identifier(model_id.clone()).await.unwrap() {
                        let state = engine.downloader(&fresh_model).state().await.unwrap();
                        let mut models_guard = models.lock().await;
                        models_guard.insert(
                            model_id,
                            ModelWithState {
                                model: fresh_model,
                                state,
                            },
                        );
                    }
                }
            }
        });

        *self.state_listener_handle.lock().await = Some(handle);
    }

    pub async fn handle_event(
        &mut self,
        event: AppEvent,
    ) {
        match event {
            AppEvent::Key(key) => match key {
                KeyCode::Char('q') => self.should_quit = true,
                KeyCode::Left => self.prev_section(),
                KeyCode::Right => self.next_section(),
                KeyCode::Up => self.previous_item().await,
                KeyCode::Down => self.next_item().await,
                KeyCode::Char('d') | KeyCode::Enter => self.download_or_resume_selected().await,
                KeyCode::Char('p') => self.pause_selected().await,
                KeyCode::Char('x') => self.delete_selected().await,
                _ => {},
            },
            AppEvent::Tick => {},
        }
    }

    fn next_section(&mut self) {
        self.active_section = self.active_section.next();
    }

    fn prev_section(&mut self) {
        self.active_section = self.active_section.prev();
    }

    async fn next_item(&mut self) {
        let section = self.active_section;
        let count = {
            let models = self.models.lock().await;
            ModelOrganizer::filter_for_section(&models, section).len()
        };

        if count == 0 {
            return;
        }

        if let Some(state) = self.list_states.get_mut(&section) {
            let i = state.selected().map_or(0, |i| {
                if i >= count - 1 {
                    count - 1
                } else {
                    i + 1
                }
            });
            state.select(Some(i));
        }
    }

    async fn previous_item(&mut self) {
        let section = self.active_section;
        let count = {
            let models = self.models.lock().await;
            ModelOrganizer::filter_for_section(&models, section).len()
        };

        if count == 0 {
            return;
        }

        if let Some(state) = self.list_states.get_mut(&section) {
            let i = state.selected().map_or(0, |i| {
                if i == 0 {
                    0
                } else {
                    i - 1
                }
            });
            state.select(Some(i));
        }
    }

    pub fn get_selected_model_id(
        &self,
        models: &HashMap<ModelIdentifier, ModelWithState>,
    ) -> Option<ModelIdentifier> {
        let state = self.list_states.get(&self.active_section)?;
        let selected_idx = state.selected()?;
        let section_models = ModelOrganizer::filter_for_section(models, self.active_section);
        section_models.get(selected_idx).map(|(id, _)| id.clone())
    }

    async fn download_or_resume_selected(&mut self) {
        let model_id = {
            let models = self.models.lock().await;
            self.get_selected_model_id(&models)
        };

        if let Some(id) = model_id {
            let models_guard = self.models.lock().await;
            if let Some(model_with_state) = models_guard.get(&id) {
                let _ = self.engine.downloader(&model_with_state.model).resume().await;
            }
        }
    }

    async fn pause_selected(&mut self) {
        let model_id = {
            let models = self.models.lock().await;
            self.get_selected_model_id(&models)
        };

        if let Some(id) = model_id {
            let models_guard = self.models.lock().await;
            if let Some(model_with_state) = models_guard.get(&id)
                && model_with_state.state.can_pause()
            {
                let _ = self.engine.downloader(&model_with_state.model).pause().await;
            }
        }
    }

    async fn delete_selected(&mut self) {
        let model_id = {
            let models = self.models.lock().await;
            self.get_selected_model_id(&models)
        };

        if let Some(id) = model_id {
            let models_guard = self.models.lock().await;
            if let Some(model_with_state) = models_guard.get(&id)
                && model_with_state.state.can_delete()
            {
                let _ = self.engine.downloader(&model_with_state.model).delete().await;
            }
        }
    }

    pub fn get_helpers(
        &self,
        models: &HashMap<ModelIdentifier, ModelWithState>,
    ) -> Vec<String> {
        let mut helpers = vec!["←→: Switch section".to_string()];
        if let Some(model_with_state) = self.get_selected_model_id(models).and_then(|id| models.get(&id)) {
            helpers.insert(0, "↑↓: Navigate".to_string());
            match &model_with_state.state.phase {
                DownloadPhase::NotDownloaded {} => helpers.push("d/Enter: Download".to_string()),
                DownloadPhase::Paused {}
                | DownloadPhase::Error {
                    ..
                } => {
                    helpers.push("d/Enter: Resume".to_string());
                    helpers.push("x: Delete".to_string());
                },
                DownloadPhase::Downloading {} => {
                    helpers.push("p: Pause".to_string());
                    helpers.push("x: Delete".to_string());
                },
                DownloadPhase::Downloaded {} => helpers.push("x: Delete".to_string()),
                DownloadPhase::Locked {
                    manager_id,
                } => helpers.push(format!("Locked by {manager_id}, waiting")),
            }
        }
        helpers.push("q: Quit".to_string());
        helpers
    }
}
