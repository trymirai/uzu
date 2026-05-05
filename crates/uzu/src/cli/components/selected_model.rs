use std::time::{Duration, Instant};

use iocraft::prelude::*;
use tokio_stream::StreamExt;

use crate::{
    cli::components::{ApplicationState, ProgressBar},
    storage::types::DownloadPhase,
};

const DOWNLOAD_PROGRESS_UPDATE_INTERVAL: Duration = Duration::from_millis(150);

#[derive(Clone, Copy)]
enum StorageAction {
    Toggle,
    Delete,
}

#[derive(Default, Props)]
pub struct SelectedModelProps {}

#[component]
pub fn SelectedModel(
    _props: &SelectedModelProps,
    mut hooks: Hooks,
) -> impl Into<AnyElement<'static>> {
    let state = *hooks.use_context::<State<ApplicationState>>();

    hooks.use_future({
        let engine = state.read().engine.clone();
        let model = state.read().model_state.as_ref().map(|model_state| model_state.model.clone());
        let mut state = state;
        async move {
            let Some(model) = model else {
                return;
            };
            if !model.is_downloadable() {
                return;
            }
            let identifier = model.identifier.clone();
            let downloader = engine.downloader(&model);

            let mut stream = engine.storage_subscribe().await;

            if let Some(initial) = downloader.state().await {
                if let Some(model_state) = state.write().model_state.as_mut() {
                    model_state.download_state = initial;
                }
            }

            if downloader.resume().await.is_err() {
                return;
            }

            let mut last_progress_rendered_at: Option<Instant> = None;
            while let Some(Ok((event_identifier, event_state))) = stream.next().await {
                if event_identifier != identifier {
                    continue;
                }

                if matches!(event_state.phase, DownloadPhase::Downloading {}) {
                    if last_progress_rendered_at
                        .is_some_and(|rendered_at| rendered_at.elapsed() < DOWNLOAD_PROGRESS_UPDATE_INTERVAL)
                    {
                        continue;
                    }
                    last_progress_rendered_at = Some(Instant::now());
                } else {
                    last_progress_rendered_at = None;
                }

                if let Some(model_state) = state.write().model_state.as_mut() {
                    model_state.download_state = event_state;
                }
            }
        }
    });

    let on_action = hooks.use_async_handler({
        let engine = state.read().engine.clone();
        let model = state.read().model_state.as_ref().map(|model_state| model_state.model.clone());
        let has_session =
            state.read().model_state.as_ref().map(|model_state| model_state.session_state.is_some()).unwrap_or(false);
        move |action: StorageAction| {
            let engine = engine.clone();
            let model = model.clone();
            async move {
                let Some(model) = model else {
                    return;
                };
                if !model.is_downloadable() {
                    return;
                }
                if has_session {
                    return;
                }
                let downloader = engine.downloader(&model);
                match action {
                    StorageAction::Toggle => {
                        let Some(current) = downloader.state().await else {
                            return;
                        };
                        match current.phase {
                            DownloadPhase::Downloading {} => {
                                let _ = downloader.pause().await;
                            },
                            DownloadPhase::Paused {}
                            | DownloadPhase::NotDownloaded {}
                            | DownloadPhase::Locked {}
                            | DownloadPhase::Error {
                                ..
                            } => {
                                let _ = downloader.resume().await;
                            },
                            DownloadPhase::Downloaded {} => {},
                        }
                    },
                    StorageAction::Delete => {
                        let _ = downloader.delete().await;
                    },
                }
            }
        }
    });

    let is_downloadable =
        state.read().model_state.as_ref().map(|model_state| model_state.model.is_downloadable()).unwrap_or(false);

    hooks.use_terminal_events(move |event| {
        if !is_downloadable {
            return;
        }
        let TerminalEvent::Key(KeyEvent {
            code,
            kind,
            modifiers,
            ..
        }) = event
        else {
            return;
        };
        if kind == KeyEventKind::Release {
            return;
        }
        if !modifiers.contains(KeyModifiers::CONTROL) {
            return;
        }
        match code {
            KeyCode::Char('p') => on_action(StorageAction::Toggle),
            KeyCode::Char('d') => on_action(StorageAction::Delete),
            _ => {},
        }
    });

    let theme = state.read().theme.clone();
    let model_data = state.read().model_state.as_ref().map(|model_state| {
        let session_status = model_state.session_state.as_deref().and_then(|session_state| session_state.status_text());
        (model_state.model.clone(), model_state.download_state.clone(), session_status)
    });

    let view: AnyElement<'static> = match model_data {
        None => element! { View }.into(),
        Some((model, download_state, session_status)) => {
            let is_downloaded = matches!(download_state.phase, DownloadPhase::Downloaded {});
            let is_downloading = matches!(download_state.phase, DownloadPhase::Downloading {});
            let status = if is_downloading {
                let percent = (download_state.progress() * 100.0).round() as u32;
                format!("{}%", percent)
            } else if is_downloaded {
                model
                    .specializations
                    .iter()
                    .map(|specializations| specializations.name())
                    .collect::<Vec<String>>()
                    .join(", ")
            } else {
                download_state.name()
            };
            let progress_value = download_state.progress();
            let progress_size = format!(
                "{:.2}/{:.2} GB",
                download_state.downloaded_bytes.max(0) as f64 / 1_000_000_000.0,
                download_state.total_bytes.max(0) as f64 / 1_000_000_000.0,
            );
            let padding = theme.padding();
            let padding_wide = theme.padding_wide();

            element! {
                View(
                    flex_direction: FlexDirection::Row,
                    align_items: AlignItems::Center,
                    width: 100pct,
                ) {
                    Text(content: model.name(), color: theme.accent_color)
                    #(model.is_downloadable().then(|| element! {
                        View(flex_direction: FlexDirection::Row) {
                            View(width: padding as u32)
                            Text(content: status, color: theme.subtitle_color)
                        }
                    }))
                    View(width: padding_wide as u32)
                    #(is_downloading.then(|| element! {
                        Fragment {
                            ProgressBar(progress: progress_value)
                            View(width: padding as u32)
                            Text(content: progress_size, color: theme.subtitle_color)
                        }
                    }))
                    #((!is_downloading).then(|| element! {
                        View(flex_grow: 1.0f32)
                    }))
                    #(session_status.map(|status| element! {
                        Text(content: status, color: theme.subtitle_color)
                    }))
                }
            }
            .into()
        },
    };
    view
}
