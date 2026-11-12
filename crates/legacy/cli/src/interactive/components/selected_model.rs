use std::time::{Duration, Instant};

use iocraft::prelude::*;
use tokio_stream::StreamExt;
use unicode_width::UnicodeWidthStr;
use uzu::storage::DownloadPhase;

use crate::{
    common::thinking::ThinkingSupport,
    interactive::{
        components::{ApplicationState, ProgressBar, ThinkingSupportExt},
        flows::DeleteModelFlow,
        util::{types_text, use_terminal_width},
    },
};

const DOWNLOAD_PROGRESS_UPDATE_INTERVAL: Duration = Duration::from_millis(150);
/// The narrowest download progress bar worth drawing.
const PROGRESS_BAR_MINIMAL_WIDTH: usize = 10;

#[derive(Default, Props)]
pub struct SelectedModelProps {}

#[component]
pub fn SelectedModel(
    _props: &SelectedModelProps,
    mut hooks: Hooks,
) -> impl Into<AnyElement<'static>> {
    let state = *hooks.use_context::<State<ApplicationState>>();
    let width = use_terminal_width(&mut hooks) as usize;

    hooks.use_future({
        let engine = state.read().engine.clone();
        let model = state.read().model_state.as_ref().map(|model_state| model_state.model.clone());
        let mut state = state;
        async move {
            let Some(model) = model else {
                return;
            };
            let thinking = ThinkingSupport::for_model(&model);
            if let Some(model_state) = state.write().model_state.as_mut() {
                model_state.thinking = thinking;
            }
            if !model.is_downloadable() {
                return;
            }
            let identifier = model.identifier.clone();
            let downloader = engine.downloader(&model);

            let mut stream = engine.storage_subscribe();

            let mut was_downloaded = false;
            if let Some(initial) = downloader.state().await {
                was_downloaded = matches!(initial.phase, DownloadPhase::Downloaded {});
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

                if event_state.is_in_progress() {
                    if last_progress_rendered_at
                        .is_some_and(|rendered_at| rendered_at.elapsed() < DOWNLOAD_PROGRESS_UPDATE_INTERVAL)
                    {
                        continue;
                    }
                    last_progress_rendered_at = Some(Instant::now());
                } else {
                    last_progress_rendered_at = None;
                }

                let became_downloaded = !was_downloaded && matches!(event_state.phase, DownloadPhase::Downloaded {});
                was_downloaded = matches!(event_state.phase, DownloadPhase::Downloaded {});
                if let Some(model_state) = state.write().model_state.as_mut() {
                    model_state.download_state = event_state;
                }
                if became_downloaded {
                    let thinking = ThinkingSupport::for_model(&model);
                    if let Some(model_state) = state.write().model_state.as_mut() {
                        model_state.thinking = thinking;
                    }
                }
            }
        }
    });

    let on_toggle = hooks.use_async_handler({
        let engine = state.read().engine.clone();
        let model = state.read().model_state.as_ref().map(|model_state| model_state.model.clone());
        let has_session =
            state.read().model_state.as_ref().map(|model_state| model_state.session_state.is_some()).unwrap_or(false);
        move |()| {
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
                let Some(current) = downloader.state().await else {
                    return;
                };
                match current.phase {
                    DownloadPhase::Downloading {} => {
                        let _ = downloader.pause().await;
                    },
                    DownloadPhase::Paused {}
                    | DownloadPhase::NotDownloaded {}
                    | DownloadPhase::Locked {
                        ..
                    }
                    | DownloadPhase::Error {
                        ..
                    } => {
                        let _ = downloader.resume().await;
                    },
                    DownloadPhase::Initializing {} | DownloadPhase::Downloaded {} => {},
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
        if types_text(modifiers) {
            return;
        }
        match code {
            KeyCode::Char('p') => on_toggle(()),
            KeyCode::Char('d') if state.read().can_delete_model() => {
                let mut state = state;
                state.write().flow = Some(Box::new(DeleteModelFlow));
            },
            _ => {},
        }
    });

    let theme = state.read().theme().clone();
    let preferences = state.read().preferences().clone();
    let thinking = state.read().thinking();
    let model_data = state.read().model_state.as_ref().map(|model_state| {
        let session_status = model_state.session_state.as_deref().and_then(|session_state| session_state.status_text());
        (model_state.model.clone(), model_state.download_state.clone(), session_status, model_state.thinking)
    });

    let view: AnyElement<'static> = match model_data {
        None => element! { View }.into(),
        Some((model, download_state, session_status, thinking_support)) => {
            let is_downloaded = matches!(download_state.phase, DownloadPhase::Downloaded {});
            let is_downloading = download_state.is_in_progress();
            let percent = (download_state.progress() * 100.0).round() as u32;
            let status = match &download_state.phase {
                DownloadPhase::Downloading {} => format!("{percent}%"),
                DownloadPhase::Locked {
                    ..
                } => format!("Locked {percent}%"),
                DownloadPhase::Downloaded {} => model
                    .specializations
                    .iter()
                    .map(|specializations| specializations.name())
                    .collect::<Vec<String>>()
                    .join(", "),
                DownloadPhase::Error {
                    message,
                } => format!("Error: {message}"),
                _ => download_state.name(),
            };
            let progress_value = download_state.progress();
            let progress_size = format!(
                "{:.2}/{:.2} GB",
                download_state.downloaded_bytes as f64 / 1_000_000_000.0,
                download_state.total_bytes as f64 / 1_000_000_000.0,
            );
            let padding = theme.padding();
            let padding_wide = theme.padding_wide();
            let chat_indicator = (model.is_chat_capable() && (is_downloaded || !model.is_downloadable())).then(|| {
                let thinking = match thinking_support.with_preference(&thinking) {
                    ThinkingSupport::Unsupported => None,
                    support => Some(support.value_label()),
                };
                match thinking {
                    Some(thinking) => format!("think {} · {}", thinking, preferences.sampling.mode.label()),
                    None => format!("sampling {}", preferences.sampling.mode.label()),
                }
            });
            // Narrow terminals drop the parts that do not fit whole, in order of priority, instead of wrapping them.
            let name = model.name();
            let mut remaining = width.saturating_sub(name.width() + padding_wide as usize);
            let mut fits = |part_width: usize| {
                part_width <= remaining && {
                    remaining -= part_width;
                    true
                }
            };
            let session_status = session_status.filter(|status| fits(status.width()));
            let status =
                model.is_downloadable().then_some(status).filter(|status| fits(padding as usize + status.width()));
            let is_downloading =
                is_downloading && fits(PROGRESS_BAR_MINIMAL_WIDTH + padding as usize + progress_size.width());
            let chat_indicator = chat_indicator.filter(|indicator| fits(indicator.width() + padding_wide as usize));

            element! {
                View(
                    flex_direction: FlexDirection::Row,
                    align_items: AlignItems::Center,
                    width: 100pct,
                ) {
                    Text(content: name, color: theme.accent_color)
                    #(status.map(|status| element! {
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
                    #(chat_indicator.map(|indicator| element! {
                        View(flex_direction: FlexDirection::Row) {
                            Text(content: indicator, color: theme.subtitle_color)
                            View(width: padding_wide as u32)
                        }
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
