use std::{collections::HashMap, sync::Arc};

use crossterm::event::KeyCode;
use futures_util::StreamExt;
use ratatui::widgets::ListState;
use shoji::types::model::Model;
use tokio::{sync::Mutex as TokioMutex, task::JoinHandle};
use uzu::{engine::Engine, storage::types::DownloadState};

use crate::{events::AppEvent, models::ModelOrganizer, sections::Section};

/// Holds a Model handle and its cached state for UI rendering
#[derive(Clone)]
pub struct ModelWithState {
    pub model: Model,
    pub state: DownloadState,
}

pub struct App {
    pub engine: Arc<Engine>,
    pub models: Arc<TokioMutex<HashMap<String, ModelWithState>>>,
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
                model.identifier(),
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
        let mut updates = self.engine.storage_subscribe().await;

        let handle = self.tokio_handle.spawn(async move {
            while let Some(Ok((model_id, state))) = updates.next().await {
                let mut models_guard = models.lock().await;

                if let Some(model_with_state) = models_guard.get_mut(&model_id) {
                    // Update the cached state
                    model_with_state.state = state;
                } else {
                    // Model not in local HashMap; fetch from storage and add it
                    drop(models_guard);
                    if let Some(fresh_model) = engine.model_by_identifier(&model_id).await.unwrap() {
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
        models: &HashMap<String, ModelWithState>,
    ) -> Option<String> {
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
                use uzu::storage::types::DownloadPhase::*;
                match model_with_state.state.phase {
                    Downloaded {} => {
                        // Already installed; ignore download command
                    },
                    _ => {
                        let _ = self.engine.downloader(&model_with_state.model).resume().await;
                    },
                }
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
            if let Some(model_with_state) = models_guard.get(&id) {
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
            if let Some(model_with_state) = models_guard.get(&id) {
                // Can delete from any section: Downloading, Paused, Downloaded, Installed
                let _ = self.engine.downloader(&model_with_state.model).delete().await;
            }
        }
    }

    /// Get helper text based on current section and selection
    pub fn get_helpers(
        &self,
        models: &HashMap<String, ModelWithState>,
    ) -> Vec<String> {
        let selected_model_id = self.get_selected_model_id(models);

        match self.active_section {
            Section::Available => {
                if selected_model_id.is_some() {
                    vec![
                        "↑↓: Navigate".to_string(),
                        "←→: Switch section".to_string(),
                        "d/Enter: Download".to_string(),
                        "q: Quit".to_string(),
                    ]
                } else {
                    vec!["←→: Switch section".to_string(), "q: Quit".to_string()]
                }
            },
            Section::Downloading => {
                if let Some(_id) = &selected_model_id {
                    // Check if model is paused
                    // Note: This is a synchronous context, so we'll use a blocking approach
                    // In a real scenario, this should be refactored to async
                    let is_paused = false; // Simplified for now - models are dynamic

                    if is_paused {
                        vec![
                            "↑↓: Navigate".to_string(),
                            "←→: Switch section".to_string(),
                            "d/Enter: Resume".to_string(),
                            "x: Delete".to_string(),
                            "q: Quit".to_string(),
                        ]
                    } else {
                        vec![
                            "↑↓: Navigate".to_string(),
                            "←→: Switch section".to_string(),
                            "p: Pause".to_string(),
                            "x: Delete".to_string(),
                            "q: Quit".to_string(),
                        ]
                    }
                } else {
                    vec!["←→: Switch section".to_string(), "q: Quit".to_string()]
                }
            },
            Section::Installed => {
                if selected_model_id.is_some() {
                    vec![
                        "↑↓: Navigate".to_string(),
                        "←→: Switch section".to_string(),
                        "x: Delete".to_string(),
                        "q: Quit".to_string(),
                    ]
                } else {
                    vec!["←→: Switch section".to_string(), "q: Quit".to_string()]
                }
            },
        }
    }
}
