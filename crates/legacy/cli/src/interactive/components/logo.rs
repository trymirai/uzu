use std::path::Path;

use indoc::indoc;
use iocraft::prelude::*;
use unicode_width::UnicodeWidthStr;
use uzu_engine::{VERSION, backends::selected_backend};

use crate::interactive::{components::ApplicationState, util::use_terminal_width};

const LOGO: &str = indoc! {r"
  _ __ ___   (_)  _ __   __ _  (_)
 | '_ ` _ \  | | | '__| / _` | | |
 | | | | | | | | | |   | (_| | | |
 |_| |_| |_| |_| |_|    \__,_| |_|"};

fn about_text() -> String {
    let current_path = std::env::current_dir().ok();
    let home_path = dirs::home_dir();

    let mut lines = vec![
        format!("v{} · {} backend", VERSION, selected_backend()),
        "A high-performance inference engine for AI models".to_string(),
        "Zero latency, full data privacy, no inference costs".to_string(),
    ];

    if let Some(current_path) = current_path {
        let path = match home_path.as_ref().and_then(|home_path| current_path.strip_prefix(home_path).ok()) {
            Some(relative_path) if relative_path.as_os_str().is_empty() => "~".to_string(),
            Some(relative_path) => Path::new("~").join(relative_path).display().to_string(),
            None => current_path.display().to_string(),
        };

        lines.push(path);
    }

    lines.join("\n")
}

fn text_width(text: &str) -> u16 {
    text.lines().map(UnicodeWidthStr::width).max().unwrap_or(0) as u16
}

/// The art beside the about text where both fit, the about text alone on narrower terminals.
#[component]
pub fn Logo(mut hooks: Hooks) -> impl Into<AnyElement<'static>> {
    let state = hooks.use_context::<State<ApplicationState>>();
    let about = hooks.use_const(about_text);
    let width = use_terminal_width(&mut hooks);
    let theme = state.read().theme().clone();
    let fits = text_width(LOGO) + theme.padding_wide() + text_width(&about) + 2 * theme.padding() <= width;

    element! {
        View(
            flex_direction: FlexDirection::Row,
            align_items: AlignItems::FlexStart,
            width: 100pct,
            column_gap: theme.padding_wide(),
        ) {
            #(fits.then(|| element! {
                Text(content: LOGO, color: theme.accent_color)
            }))
            Text(
                content: about,
                color: theme.subtitle_color
            )
        }
    }
}
