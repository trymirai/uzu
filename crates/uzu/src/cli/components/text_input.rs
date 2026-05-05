use std::{
    sync::{Arc, Mutex},
    time::Duration,
};

use iocraft::prelude::*;
use tokio::sync::watch;

use crate::cli::components::RenderedText;

const ENTER_SUBMIT_DELAY: Duration = Duration::from_millis(50);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum InputType {
    #[default]
    Text,
    Secret,
}

impl InputType {
    fn is_secret(&self) -> bool {
        *self == Self::Secret
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Default)]
pub enum TextInputFocus {
    #[default]
    Disabled,
    Minimal,
    Full,
}

impl TextInputFocus {
    pub fn is_disabled(&self) -> bool {
        *self == Self::Disabled
    }

    pub fn is_minimal(&self) -> bool {
        *self == Self::Minimal
    }
}

#[derive(Default, Props)]
pub struct TextInputProps {
    pub maximal_width: u16,
    pub focus: TextInputFocus,
    pub r#type: InputType,
    pub on_change: HandlerMut<'static, String>,
    pub on_submit: HandlerMut<'static, String>,
}

#[derive(Default)]
struct TextInputHandlers {
    on_change: HandlerMut<'static, String>,
    on_submit: HandlerMut<'static, String>,
}

#[component]
pub fn TextInput(
    props: &mut TextInputProps,
    mut hooks: Hooks,
) -> impl Into<AnyElement<'static>> {
    let maximal_width = props.maximal_width;
    let focus = props.focus;
    let r#type = props.r#type;
    let handlers = hooks.use_memo(|| Arc::new(Mutex::new(TextInputHandlers::default())), ());
    if let Ok(mut handlers) = handlers.lock() {
        handlers.on_change = props.on_change.take();
        handlers.on_submit = props.on_submit.take();
    }

    let mut state = hooks.use_state(|| RenderedText::new());
    let mut pending_enter_identifier = hooks.use_state(|| None::<u64>);
    let mut enter_identifier = hooks.use_state(|| 0u64);
    let (enter_timeout_sender, mut enter_timeout_receiver) = hooks.use_memo(|| watch::channel(None::<u64>), ());

    hooks.use_future({
        let handlers = handlers.clone();
        async move {
            while enter_timeout_receiver.changed().await.is_ok() {
                let Some(identifier) = *enter_timeout_receiver.borrow() else {
                    continue;
                };
                if pending_enter_identifier.try_get() != Some(Some(identifier)) {
                    continue;
                }
                pending_enter_identifier.set(None);

                let Some(mut state) = state.try_write() else {
                    continue;
                };
                let text = state.original_text.clone();
                state.reset();
                drop(state);

                if let Ok(mut handlers) = handlers.lock() {
                    (handlers.on_change)(String::new());
                    (handlers.on_submit)(text);
                }
            }
        }
    });

    hooks.use_terminal_events(move |event| {
        if focus.is_disabled() {
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

        let notify_change = {
            let handlers = handlers.clone();
            let state = state;
            move || {
                let Some(text) = state.try_read().map(|state| state.original_text.clone()) else {
                    return;
                };
                if let Ok(mut handlers) = handlers.lock() {
                    (handlers.on_change)(text);
                }
            }
        };

        if pending_enter_identifier.try_get().flatten().is_some() {
            pending_enter_identifier.set(None);
            let Some(mut state) = state.try_write() else {
                return;
            };
            state.add_character('\n');
            drop(state);
            notify_change();
        }

        match code {
            KeyCode::Char(character) if !modifiers.intersects(KeyModifiers::CONTROL) => {
                state.write().add_character(character);
                notify_change();
            },
            KeyCode::Backspace => {
                state.write().remove_character();
                notify_change();
            },
            KeyCode::Delete => {
                if focus.is_minimal() {
                    return;
                }
                state.write().reset();
                notify_change();
            },
            KeyCode::Left => state.write().move_position_left(),
            KeyCode::Right => state.write().move_position_right(),
            KeyCode::Up => {
                if focus.is_minimal() {
                    return;
                }
                state.write().move_position_up(maximal_width as usize);
            },
            KeyCode::Down => {
                if focus.is_minimal() {
                    return;
                }
                state.write().move_position_down(maximal_width as usize);
            },
            KeyCode::Home => {
                if focus.is_minimal() {
                    return;
                }
                state.write().move_position_to_start();
            },
            KeyCode::End => {
                if focus.is_minimal() {
                    return;
                }
                state.write().move_position_to_end();
            },
            KeyCode::Enter if modifiers.intersects(KeyModifiers::SHIFT) => {
                if focus.is_minimal() {
                    return;
                }
                state.write().add_character('\n');
                notify_change();
            },
            KeyCode::Enter => {
                if focus.is_minimal() {
                    return;
                }
                let indetifier = enter_identifier.get() + 1;
                enter_identifier.set(indetifier);
                pending_enter_identifier.set(Some(indetifier));

                let enter_timeout_sender = enter_timeout_sender.clone();
                tokio::spawn(async move {
                    tokio::time::sleep(ENTER_SUBMIT_DELAY).await;
                    let _ = enter_timeout_sender.send(Some(indetifier));
                });
            },
            _ => {},
        }
    });

    let segments = if r#type.is_secret() {
        let state = state.read();
        let hidden_text: String = state.original_text.chars().map(|_| '*').collect();
        RenderedText {
            original_text: hidden_text,
            position: state.position,
        }
        .segments(maximal_width as usize)
    } else {
        state.read().segments(maximal_width as usize)
    };
    element! {
        View(width: maximal_width, height: segments.len().max(1) as u16, flex_direction: FlexDirection::Column) {
            #(segments.into_iter().map(|segment| element! {
                MixedText(contents: segment, wrap: TextWrap::NoWrap)
            }))
        }
    }
}
