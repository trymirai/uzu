use std::{
    sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering},
    time::Duration,
};

use ratatui::style::Color;

const THEMES: [(&str, Color); 7] = [
    ("green", Color::Green),
    ("cyan", Color::Cyan),
    ("blue", Color::Blue),
    ("magenta", Color::Magenta),
    ("yellow", Color::Yellow),
    ("red", Color::Red),
    ("white", Color::White),
];

const MIN_INTERVAL_MS: u64 = 100;
const MAX_INTERVAL_MS: u64 = 5000;
const INTERVAL_STEP_MS: u64 = 100;

static INTERVAL_MS: AtomicU64 = AtomicU64::new(1000);
static THEME_INDEX: AtomicUsize = AtomicUsize::new(0);
static DARK_BACKGROUND: AtomicBool = AtomicBool::new(true);
static SHOW_INFO: AtomicBool = AtomicBool::new(false);
static DATA_VERSION: AtomicU64 = AtomicU64::new(0);

pub(crate) fn theme() -> (&'static str, Color) {
    THEMES[THEME_INDEX.load(Ordering::Relaxed) % THEMES.len()]
}

pub(crate) fn accent() -> Color {
    theme().1
}

pub(crate) fn background() -> Color {
    if DARK_BACKGROUND.load(Ordering::Relaxed) {
        Color::Black
    } else {
        Color::Reset
    }
}

pub(crate) fn interval() -> Duration {
    Duration::from_millis(INTERVAL_MS.load(Ordering::Relaxed))
}

pub(crate) fn interval_ms() -> u64 {
    INTERVAL_MS.load(Ordering::Relaxed)
}

pub(crate) fn cycle_theme() {
    THEME_INDEX.fetch_add(1, Ordering::Relaxed);
}

pub(crate) fn toggle_background() {
    DARK_BACKGROUND.fetch_xor(true, Ordering::Relaxed);
}

pub(crate) fn toggle_info() {
    SHOW_INFO.fetch_xor(true, Ordering::Relaxed);
}

pub(crate) fn show_info() -> bool {
    SHOW_INFO.load(Ordering::Relaxed)
}

pub(crate) fn speed_up() {
    adjust(-(INTERVAL_STEP_MS as i64));
}

pub(crate) fn slow_down() {
    adjust(INTERVAL_STEP_MS as i64);
}

fn adjust(delta: i64) {
    let current = INTERVAL_MS.load(Ordering::Relaxed) as i64;
    let next = (current + delta).clamp(MIN_INTERVAL_MS as i64, MAX_INTERVAL_MS as i64) as u64;
    INTERVAL_MS.store(next, Ordering::Relaxed);
}

pub(crate) fn bump_data_version() {
    DATA_VERSION.fetch_add(1, Ordering::Relaxed);
}

pub(crate) fn data_version() -> u64 {
    DATA_VERSION.load(Ordering::Relaxed)
}
