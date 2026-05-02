use iocraft::prelude::*;
use tokio_stream::StreamExt;

use crate::{
    cli::components::{ApplicationState, ProgressBar},
    storage::types::DownloadPhase,
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

            while let Some(Ok((event_identifier, event_state))) = stream.next().await {
                if event_identifier != identifier {
                    continue;
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
    let snapshot = state
        .read()
        .model_state
        .as_ref()
        .map(|model_state| (model_state.model.clone(), model_state.download_state.clone()));

    let view: AnyElement<'static> = match snapshot {
        None => element! { View }.into(),
        Some((model, download_state)) => {
            let is_downloading = matches!(download_state.phase, DownloadPhase::Downloading {});
            let status = if is_downloading {
                let percent = (download_state.progress() * 100.0).round() as u32;
                format!("{}%", percent)
            } else {
                download_state.name()
            };
            let progress_value = download_state.progress();
            let padding = theme.padding();
            let padding_wide = theme.padding_wide();

            element! {
                View(flex_direction: FlexDirection::Row, align_items: AlignItems::Center) {
                    Text(content: model.name(), color: theme.accent_color)
                    #(model.is_downloadable().then(|| element! {
                        View(flex_direction: FlexDirection::Row) {
                            View(width: padding as u32)
                            Text(content: status, color: theme.subtitle_color)
                        }
                    }))
                    View(width: padding_wide as u32)
                    #(is_downloading.then(|| element! {
                        ProgressBar(progress: progress_value)
                    }))
                }
            }
            .into()
        },
    };
    view
}
