use std::sync::OnceLock;

use ratatui::style::{Color, Modifier, Style};

const LEVEL_ALPHAS: [f32; 5] = [0.14, 0.22, 0.42, 0.68, 1.00];

pub(crate) fn blend(
    foreground: (u8, u8, u8),
    background: (u8, u8, u8),
    alpha: f32,
) -> (u8, u8, u8) {
    let red = (foreground.0 as f32 * alpha + background.0 as f32 * (1.0 - alpha)) as u8;
    let green = (foreground.1 as f32 * alpha + background.1 as f32 * (1.0 - alpha)) as u8;
    let blue = (foreground.2 as f32 * alpha + background.2 as f32 * (1.0 - alpha)) as u8;
    (red, green, blue)
}

pub(crate) fn supports_truecolor() -> bool {
    static SUPPORTS_TRUECOLOR: OnceLock<bool> = OnceLock::new();
    *SUPPORTS_TRUECOLOR.get_or_init(|| {
        std::env::var("COLORTERM")
            .map(|value| {
                let value = value.to_ascii_lowercase();
                value.contains("truecolor") || value.contains("24bit")
            })
            .unwrap_or(false)
    })
}

pub(crate) fn level_for(fraction: f64) -> usize {
    let fraction = fraction.clamp(0.0, 1.0);
    if fraction <= 0.0 {
        0
    } else if fraction > 0.75 {
        4
    } else if fraction > 0.5 {
        3
    } else if fraction > 0.25 {
        2
    } else {
        1
    }
}

pub(crate) fn level_style(
    level: usize,
    accent: (u8, u8, u8),
    foreground: (u8, u8, u8),
    background: (u8, u8, u8),
    fallback_accent: Color,
) -> Style {
    let level = level.min(4);
    if supports_truecolor() {
        let anchor = if level == 0 { foreground } else { accent };
        let (red, green, blue) = blend(anchor, background, LEVEL_ALPHAS[level]);
        let style = Style::default().fg(Color::Rgb(red, green, blue));
        if level == 0 {
            style
        } else {
            style.add_modifier(Modifier::BOLD)
        }
    } else if level == 0 {
        Style::default().fg(Color::DarkGray)
    } else {
        Style::default().fg(fallback_accent).add_modifier(Modifier::BOLD)
    }
}
