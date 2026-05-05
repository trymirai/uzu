use iocraft::prelude::*;

use crate::cli::components::RenderedText;

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

#[component]
pub fn TextInput(
    props: &mut TextInputProps,
    mut hooks: Hooks,
) -> impl Into<AnyElement<'static>> {
    let maximal_width = props.maximal_width;
    let focus = props.focus;
    let r#type = props.r#type;
    let mut on_change = props.on_change.take();
    let mut on_submit = props.on_submit.take();

    let mut state = hooks.use_state(|| RenderedText::new());

    let mut notify_change = move || {
        on_change(state.read().original_text.clone());
    };

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
                let text = state.read().original_text.clone();
                state.write().reset();
                notify_change();
                on_submit(text);
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
