use iocraft::prelude::*;
use tokio_stream::StreamExt;

use crate::{
    cli::components::{ApplicationState, ProgressBar},
    storage::types::{DownloadPhase, DownloadState},
};

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
    let mut state = *hooks.use_context::<State<ApplicationState>>();
    let downloader_state = hooks.use_state(|| {
        let engine = state.read().engine.clone();
        let model = state.read().model.clone();
        model.and_then(|model| {
            if model.is_downloadable() {
                Some(engine.downloader(&model))
            } else {
                None
            }
        })
    });
    let download_status_state = hooks.use_state(|| None::<DownloadState>);

    hooks.use_future({
        let engine = state.read().engine.clone();
        let identifier = state.read().model.as_ref().map(|model| model.identifier.clone());
        let downloader = downloader_state.read().clone();
        let mut download_status_state = download_status_state;
        async move {
            let (Some(identifier), Some(downloader)) = (identifier, downloader) else {
                return;
            };

            let mut stream = engine.storage_subscribe().await;

            if let Some(initial_status) = downloader.state().await {
                download_status_state.set(Some(initial_status.clone()));
                state.write().model_download_state = Some(initial_status);
            }
            if downloader.resume().await.is_err() {
                return;
            }
            while let Some(Ok((event_identifier, event_state))) = stream.next().await {
                if event_identifier != identifier {
                    continue;
                }
                download_status_state.set(Some(event_state.clone()));
                state.write().model_download_state = Some(event_state);
            }
        }
    });

    let on_action = hooks.use_async_handler({
        let downloader = downloader_state.read().clone();
        move |action: StorageAction| {
            let downloader = downloader.clone();
            async move {
                let Some(downloader) = downloader else {
                    return;
                };
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

    let has_downloader = downloader_state.read().is_some();
    hooks.use_terminal_events(move |event| {
        if !has_downloader {
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

    let model = state.read().model.clone();
    let theme = state.read().theme.clone();
    let view: AnyElement<'static> = match model {
        None => element! { View }.into(),
        Some(model) => {
            let state = download_status_state.read().clone().unwrap_or(DownloadState::not_downloaded(0));

            let status = match state.phase {
                DownloadPhase::Downloading {} => {
                    let percent = (state.progress() * 100.0).round() as u32;
                    format!("{}%", percent)
                },
                _ => format!("{}", state.name()),
            };
            let is_downloading = matches!(state.phase, DownloadPhase::Downloading {});

            let padding = theme.padding();
            let padding_wide = theme.padding_wide();
            element! {
                View(flex_direction: FlexDirection::Row, align_items: AlignItems::Center) {
                    Text(content: model.name(), color: theme.accent_color)
                    #(model.is_downloadable().then(|| {
                        element! {
                            View(flex_direction: FlexDirection::Row) {
                                View(width: padding as u32) {}
                                Text(content: status, color: theme.subtitle_color)
                            }
                        }
                    }))
                    View(width: padding_wide as u32) {}
                    #(is_downloading.then(|| {
                        element! {
                            ProgressBar(progress: state.progress())
                        }
                    }))
                }
            }
            .into()
        },
    };
    view
}
