//! Pure formatting helpers for the local-models lists (no view state).

use gpui::{FontWeight, IntoElement, div, prelude::*, px};
use uzu::types::model::Model;

/// Column header row for a family detail section (Installed / Available).
pub(super) fn section_header(label: &str, theme: &crate::theme::Theme) -> impl IntoElement {
    div()
        .flex()
        .items_center()
        .gap_4()
        .pt_4()
        .pb_1()
        .px_4()
        .text_xs()
        .font_weight(FontWeight::MEDIUM)
        .text_color(theme.text_muted)
        .child(div().flex_1().child(label.to_uppercase()))
        .child(div().w(px(80.)).child("SIZE"))
        .child(div().w(px(140.)).child("QUANTIZATION"))
        .child(div().w(px(100.)))
}

pub(super) fn quant_label(model: &Model) -> String {
    // `method` is already the clean short label, e.g. "mirai-m", "mirai-l",
    // "mlx" → "MIRAI-M · 4-bit", "MLX · 4-bit".
    match &model.quantization {
        Some(q) => format!("{} · {}-bit", q.method.to_uppercase(), q.bits_per_weight),
        None => "Unquantized".to_string(),
    }
}

pub(super) fn format_params(millions: f64) -> String {
    if millions >= 1000.0 {
        let b = millions / 1000.0;
        if (b.fract()).abs() < f64::EPSILON {
            format!("{b:.0}B")
        } else {
            format!("{b:.1}B")
        }
    } else {
        format!("{millions:.0}M")
    }
}

pub(crate) fn format_size(bytes: i64) -> String {
    if bytes <= 0 {
        return "—".to_string();
    }
    let b = bytes as f64;
    const GB: f64 = 1_000_000_000.0;
    const MB: f64 = 1_000_000.0;
    if b >= GB {
        format!("{:.1} GB", b / GB)
    } else {
        format!("{:.0} MB", b / MB)
    }
}
