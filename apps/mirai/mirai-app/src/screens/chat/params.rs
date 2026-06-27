//! Pure UI builders for the generation-settings drawer: stepper buttons, value
//! boxes, checkboxes, and slider/stepper parameter rows. None touch view state.

use gpui::{AnyElement, App, ClickEvent, CursorStyle, Hsla, IntoElement, Window, div, prelude::*, px};

use crate::{
    components::{Icon, IconEl, Slider},
    theme::Theme,
};

/// Small square −/+ stepper button.
fn step_button(
    id: &'static str,
    symbol: &'static str,
    border: Hsla,
    fg: Hsla,
    hover: Hsla,
    handler: impl Fn(&ClickEvent, &mut Window, &mut App) + 'static,
) -> impl IntoElement {
    div()
        .id(id)
        .w(px(24.))
        .h(px(24.))
        .flex()
        .items_center()
        .justify_center()
        .rounded_md()
        .border_1()
        .border_color(border)
        .text_color(fg)
        .cursor(CursorStyle::PointingHand)
        .hover(move |s| s.bg(hover))
        .on_click(handler)
        .child(symbol)
}

/// Round a 0–1 sampling value to 2 decimals (avoids float drift across steps).
pub(super) fn round2(v: f32) -> f32 {
    (v * 100.0).round() / 100.0
}

/// Read-only numeric value box on the right of a parameter row.
fn value_box(text: String, theme: &Theme) -> impl IntoElement {
    div()
        .min_w(px(76.))
        .px_3()
        .py_1p5()
        .rounded_md()
        .border_1()
        .border_color(theme.border)
        .bg(theme.bg)
        .text_sm()
        .text_color(theme.text)
        .child(text)
}

/// Small square checkbox used to enable/disable an optional sampling param.
pub(super) fn param_checkbox(
    id: &'static str,
    checked: bool,
    theme: &Theme,
    on_click: impl Fn(&ClickEvent, &mut Window, &mut App) + 'static,
) -> impl IntoElement {
    let mut b = div()
        .id(id)
        .size(px(18.))
        .flex_none()
        .rounded(crate::tokens::radius::SM)
        .border_1()
        .border_color(theme.border)
        .flex()
        .items_center()
        .justify_center()
        .cursor(CursorStyle::PointingHand)
        .on_click(on_click);
    if checked {
        b = b.bg(theme.text).child(IconEl::new(Icon::Check, theme.bg).size(crate::tokens::icon::XS));
    }
    b
}

/// A sampling parameter: label (+ optional checkbox) and value box on top, a
/// slider below. `frac` is the slider's normalized position; `on_change`
/// receives the new fraction.
#[allow(clippy::too_many_arguments)]
pub(super) fn slider_param(
    label: &'static str,
    checkbox: Option<AnyElement>,
    value_text: String,
    frac: f32,
    slider_id: &'static str,
    theme: &Theme,
    on_change: impl Fn(f32, &mut Window, &mut App) + 'static,
) -> impl IntoElement {
    div()
        .flex()
        .flex_col()
        .gap_2()
        .child(
            div()
                .flex()
                .items_center()
                .justify_between()
                .child(
                    div()
                        .flex()
                        .items_center()
                        .gap_2()
                        .child(div().text_sm().text_color(theme.text).child(label))
                        .children(checkbox),
                )
                .child(value_box(value_text, theme)),
        )
        .child(Slider::new(slider_id, frac).on_change(on_change))
}

/// A labeled −/+ stepper row used in the generation-settings panel.
#[allow(clippy::too_many_arguments)]
pub(super) fn param_row(
    label: &str,
    value: String,
    dec_id: &'static str,
    inc_id: &'static str,
    border: Hsla,
    fg: Hsla,
    hover: Hsla,
    on_dec: impl Fn(&ClickEvent, &mut Window, &mut App) + 'static,
    on_inc: impl Fn(&ClickEvent, &mut Window, &mut App) + 'static,
) -> impl IntoElement {
    div()
        .flex()
        .items_center()
        .justify_between()
        .child(div().text_sm().text_color(fg).child(label.to_string()))
        .child(
            div()
                .flex()
                .items_center()
                .gap_2()
                .child(step_button(dec_id, "−", border, fg, hover, on_dec))
                .child(div().w(px(44.)).text_color(fg).child(value))
                .child(step_button(inc_id, "+", border, fg, hover, on_inc)),
        )
}
