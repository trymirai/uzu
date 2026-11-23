use std::path::Path;

use indoc::indoc;
use iocraft::prelude::*;
use unicode_width::UnicodeWidthStr;
use uzu_engine::{VERSION, backends::selected_backend};

use crate::interactive::components::ApplicationState;

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

#[component]
pub fn Logo(mut hooks: Hooks) -> impl Into<AnyElement<'static>> {
    let state = hooks.use_context::<State<ApplicationState>>();
    let about = hooks.use_const(about_text);
    let (width, _) = hooks.use_terminal_size();
    let theme = state.read().theme().clone();
    // The art keeps its shape and the about text wraps beside it, down to where the version line no longer fits.
    let art_width = LOGO.lines().map(str::len).max().unwrap_or(0);
    let version_width = about.lines().next().map_or(0, UnicodeWidthStr::width);
    let shows_art = art_width + version_width + (theme.padding_wide() + 2 * theme.padding()) as usize <= width as usize;

    element! {
        View(
            flex_direction: FlexDirection::Row,
            align_items: AlignItems::FlexStart,
            width: 100pct,
            column_gap: theme.padding_wide(),
        ) {
            #(shows_art.then(|| element! {
                View(flex_shrink: 0.0) {
                    Text(content: LOGO, color: theme.accent_color)
                }
            }))
            Text(
                content: about,
                color: theme.subtitle_color
            )
        }
    }
}
