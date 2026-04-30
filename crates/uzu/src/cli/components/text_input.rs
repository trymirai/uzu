use iocraft::prelude::*;

use crate::cli::components::RenderedText;

#[derive(Default, Props)]
pub struct TextInputProps {
    pub maximal_width: u16,
    pub has_focus: bool,
    pub on_submit: HandlerMut<'static, String>,
}

#[component]
pub fn TextInput(
    props: &mut TextInputProps,
    mut hooks: Hooks,
) -> impl Into<AnyElement<'static>> {
    let maximal_width = props.maximal_width;
    let has_focus = props.has_focus;
    let mut on_submit = props.on_submit.take();

    let mut state = hooks.use_state(|| RenderedText::new());

    hooks.use_terminal_events(move |event| {
        if !has_focus {
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
            KeyCode::Char(character) => {
                state.write().add_character(character);
            },
            KeyCode::Backspace => {
                state.write().remove_character();
            },
            KeyCode::Delete => {
                state.write().reset();
            },
            KeyCode::Left => state.write().move_position_left(),
            KeyCode::Right => state.write().move_position_right(),
            KeyCode::Up => state.write().move_position_up(maximal_width as usize),
            KeyCode::Down => state.write().move_position_down(maximal_width as usize),
            KeyCode::Home => state.write().move_position_to_start(),
            KeyCode::End => state.write().move_position_to_end(),
            KeyCode::Enter if modifiers.intersects(KeyModifiers::SHIFT | KeyModifiers::ALT) => {
                state.write().reset();
                on_submit(state.read().original_text.clone());
            },
            KeyCode::Enter => {
                state.write().add_character('\n');
            },
            _ => {},
        }
    });

    let segments = state.read().segments(maximal_width as usize);
    element! {
        View(width: maximal_width, height: segments.len().max(1) as u16, flex_direction: FlexDirection::Column) {
            #(segments.into_iter().map(|segment| element! {
                MixedText(contents: segment, wrap: TextWrap::NoWrap)
            }))
        }
    }
}
