//! Interface zoom, as in Claude and browsers: View ▸ Zoom In (⌘=), Zoom Out (⌘-) and Actual Size
//! (⌘0) scale the whole window's content. The level steps through the same stops a browser uses and
//! is saved in the app settings ("zoom"), so the next launch opens at it. Every change is announced
//! to the interface as `app:zoom` (it shows the level for a moment, with buttons to step it), and the
//! traffic lights move to stay level with the sidebar button.

use std::sync::Mutex;

use tauri::{AppHandle, Emitter, Manager};

const LEVELS: [f64; 9] = [0.67, 0.75, 0.8, 0.9, 1.0, 1.1, 1.25, 1.5, 1.75];
const SETTINGS_KEY: &str = "zoom";

#[derive(Default)]
pub struct ZoomState(Mutex<Option<f64>>);

#[derive(serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Step {
    In,
    Out,
    Reset,
}

/// The stop nearest to a level, so a hand-edited or old value still steps cleanly.
fn nearest_index(level: f64) -> usize {
    LEVELS
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| (*a - level).abs().total_cmp(&(*b - level).abs()))
        .map(|(index, _)| index)
        .unwrap_or(4)
}

fn next_level(
    current: f64,
    step: &Step,
) -> f64 {
    let index = nearest_index(current);
    match step {
        Step::In => LEVELS[(index + 1).min(LEVELS.len() - 1)],
        Step::Out => LEVELS[index.saturating_sub(1)],
        Step::Reset => 1.0,
    }
}

/// The level the window is at now: 1.0 until something was zoomed or restored.
pub fn current(app: &AppHandle) -> f64 {
    app.state::<ZoomState>().0.lock().expect("zoom lock poisoned").unwrap_or(1.0)
}

fn apply(
    app: &AppHandle,
    level: f64,
) {
    let Some(window) = app.get_webview_window("main") else {
        return;
    };
    if let Err(error) = window.set_zoom(level) {
        crate::logger::warn("zoom:apply", Some(serde_json::json!({ "error": error.to_string() })));
    }
    #[cfg(target_os = "macos")]
    crate::traffic_lights::place(&window);
}

/// Opens at the saved level, if one was saved.
pub fn restore(app: &AppHandle) {
    let saved = match crate::storage::settings_load() {
        Ok(serde_json::Value::Object(settings)) => settings.get(SETTINGS_KEY).and_then(serde_json::Value::as_f64),
        _ => None,
    };
    let Some(level) = saved.filter(|level| level.is_finite()).map(|level| LEVELS[nearest_index(level)]) else {
        return;
    };
    *app.state::<ZoomState>().0.lock().expect("zoom lock poisoned") = Some(level);
    if level != 1.0 {
        apply(app, level);
    }
}

pub fn step(
    app: &AppHandle,
    step: Step,
) -> f64 {
    let state = app.state::<ZoomState>();
    let mut current = state.0.lock().expect("zoom lock poisoned");
    let level = next_level(current.unwrap_or(1.0), &step);
    *current = Some(level);
    drop(current);

    apply(app, level);
    let mut patch = serde_json::Map::new();
    patch.insert(SETTINGS_KEY.to_string(), level.into());
    if let Err(error) = crate::storage::settings_merge(patch) {
        crate::logger::warn("zoom:save", Some(serde_json::json!({ "error": error.to_string() })));
    }
    let _ = app.emit("app:zoom", level);
    level
}

#[tauri::command]
pub fn zoom_get(app: AppHandle) -> f64 {
    current(&app)
}

/// The interface's own − / + / Reset buttons, the same as the View menu.
#[tauri::command]
pub fn zoom_step(
    app: AppHandle,
    step: Step,
) -> f64 {
    self::step(&app, step)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn steps_through_the_stops_and_stops_at_the_ends() {
        assert_eq!(next_level(1.0, &Step::In), 1.1);
        assert_eq!(next_level(1.0, &Step::Out), 0.9);
        assert_eq!(next_level(1.75, &Step::In), 1.75);
        assert_eq!(next_level(0.67, &Step::Out), 0.67);
        assert_eq!(next_level(1.5, &Step::Reset), 1.0);
    }

    #[test]
    fn an_off_stop_level_snaps_to_the_nearest_stop() {
        assert_eq!(next_level(1.12, &Step::In), 1.25);
        assert_eq!(next_level(0.93, &Step::Out), 0.8);
    }
}
