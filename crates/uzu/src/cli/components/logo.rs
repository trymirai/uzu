use backend_uzu::VERSION;
use indoc::indoc;
use iocraft::prelude::*;

use crate::{cli::components::ApplicationState, device::home_path};

const LOGO: &str = indoc! {r"
  _ __ ___   (_)  _ __   __ _  (_)
 | '_ ` _ \  | | | '__| / _` | | |
 | | | | | | | | | |   | (_| | | |
 |_| |_| |_| |_| |_|    \__,_| |_|"};

fn about_text() -> String {
    let current_path = std::env::current_dir().ok();
    let home_path = home_path();

    let mut lines = vec![
        format!("v{}", VERSION),
        "A high-performance inference engine for AI models".to_string(),
        "Zero latency, full data privacy, no inference costs".to_string(),
    ];

    if let Some(current_path) = current_path {
        let path = match home_path.as_ref().and_then(|home_path| current_path.strip_prefix(home_path).ok()) {
            Some(relative_path) if relative_path.as_os_str().is_empty() => "~".to_string(),
            Some(relative_path) => format!("~/{}", relative_path.display()),
            None => current_path.display().to_string(),
        };

        lines.push(path);
    }

    lines.join("\n")
}

#[component]
pub fn Logo(mut hooks: Hooks) -> impl Into<AnyElement<'static>> {
    let state = hooks.use_context::<State<ApplicationState>>();
    let about = hooks.use_const(|| about_text());

    element! {
        View(
            flex_direction: FlexDirection::Row,
            align_items: AlignItems::FlexStart,
            width: 100pct,
            column_gap: state.read().theme.padding_wide(),
        ) {
            Text(
                content: LOGO,
                color: state.read().theme.accent_color
            )
            Text(
                content: about,
                color: state.read().theme.subtitle_color
            )
        }
    }
}
