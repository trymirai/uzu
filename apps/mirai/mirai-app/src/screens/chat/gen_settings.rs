use gpui::{AnyElement, Context, FontWeight, IntoElement, black, div, prelude::*, px};

use super::{
    params::{param_checkbox, param_row, round2, slider_param},
    sampling::SamplingMode,
    state::ChatState,
    view::{ChatView, dark_icon_url},
};
use crate::{
    components::{Icon, IconButton, IconEl, SegmentedControl, Toggle, VendorIcon},
    settings_state,
    theme::{ActiveTheme, Theme},
    tokens,
};

impl ChatView {
    pub(super) fn gen_settings_overlay(
        &self,
        cx: &mut Context<Self>,
    ) -> Option<AnyElement> {
        if !self.state.gen_settings_open {
            return None;
        }
        let theme = cx.theme().clone();
        let hover = theme.bg_hover;
        let border = theme.border;
        let fg = theme.text;

        let mode_row = div().flex().flex_col().gap_1().child(div().text_sm().text_color(fg).child("Sampling")).child(
            SegmentedControl::new("sampling-mode", self.state.sampling_mode as usize)
                .segment(
                    "Default",
                    cx.listener(|this, _, _, cx| {
                        this.state.sampling_mode = SamplingMode::Default;
                        cx.notify();
                    }),
                )
                .segment(
                    "Argmax",
                    cx.listener(|this, _, _, cx| {
                        this.state.sampling_mode = SamplingMode::Argmax;
                        cx.notify();
                    }),
                )
                .segment(
                    "Stochastic",
                    cx.listener(|this, _, _, cx| {
                        this.state.sampling_mode = SamplingMode::Stochastic;
                        cx.notify();
                    }),
                ),
        );

        let resolved = self.resolved_model(cx);
        let model_name = resolved.as_ref().map(|m| m.name()).unwrap_or_else(|| "No model".to_string());
        let vendor = resolved.as_ref().and_then(|m| m.family.as_ref().map(|f| f.vendor.name())).unwrap_or_default();
        let icon_url = resolved.as_ref().and_then(dark_icon_url);
        let reasoning_on = settings_state::current(cx).reasoning;

        let header = div()
            .flex()
            .items_center()
            .justify_between()
            .child(div().text_lg().font_weight(FontWeight::SEMIBOLD).text_color(fg).child("Edit parameters"))
            .child(
                IconButton::new("gen-close", Icon::Close)
                    .color(theme.text_muted)
                    .icon_size(16.)
                    .hit_size(28.)
                    .on_click(cx.listener(|this, _, _, cx| {
                        this.state.gen_settings_open = false;
                        cx.notify();
                    })),
            );

        let model_row = div()
            .flex()
            .items_center()
            .gap_2()
            .child(VendorIcon::new(vendor).size(tokens::icon::XL).icon_url(icon_url))
            .child(div().flex_1().min_w_0().truncate().text_sm().text_color(fg).child(model_name))
            .child(IconEl::new(Icon::ChevronDown, theme.text_muted).size(tokens::icon::MD));

        let reasoning_row = div()
            .flex()
            .items_center()
            .justify_between()
            .child(div().text_sm().text_color(fg).child("Reasoning"))
            .child(Toggle::new("gen-reasoning", reasoning_on).on_click(|_, _, cx| {
                let mut s = settings_state::current(cx);
                s.reasoning = !s.reasoning;
                settings_state::set(cx, s);
            }));

        let mut card = div()
            .occlude()
            .absolute()
            .top_0()
            .right_0()
            .bottom_0()
            .w(px(380.))
            .flex()
            .flex_col()
            .gap_4()
            .p_5()
            .bg(theme.bg)
            .border_l_1()
            .border_color(theme.border)
            .child(header)
            .child(model_row)
            .child(mode_row);

        if self.state.sampling_mode == SamplingMode::Stochastic {
            card = card
                .child(self.param_slider(
                    cx,
                    &theme,
                    "Temperature",
                    "temp-slider",
                    round2(self.state.temperature).to_string(),
                    (self.state.temperature / 2.0).clamp(0., 1.),
                    None,
                    |state, frac| state.temperature = round2(frac * 2.0),
                ))
                .child(self.param_slider(
                    cx,
                    &theme,
                    "Top K",
                    "topk-slider",
                    self.state.top_k.to_string(),
                    (self.state.top_k as f32 / 200.0).clamp(0., 1.),
                    None,
                    |state, frac| state.top_k = (frac * 200.0).round() as u32,
                ));

            let top_p_toggle = self.param_toggle(cx, &theme, "topp-cb", self.state.top_p > 0.0, |state| {
                state.top_p = if state.top_p > 0.0 {
                    0.0
                } else {
                    0.95
                };
            });
            card = card.child(self.param_slider(
                cx,
                &theme,
                "Top P",
                "topp-slider",
                round2(self.state.top_p).to_string(),
                self.state.top_p.clamp(0., 1.),
                Some(top_p_toggle),
                |state, frac| state.top_p = round2(frac),
            ));

            let min_p_toggle = self.param_toggle(cx, &theme, "minp-cb", self.state.min_p > 0.0, |state| {
                state.min_p = if state.min_p > 0.0 {
                    0.0
                } else {
                    0.05
                };
            });
            card = card.child(self.param_slider(
                cx,
                &theme,
                "Min P",
                "minp-slider",
                round2(self.state.min_p).to_string(),
                self.state.min_p.clamp(0., 1.),
                Some(min_p_toggle),
                |state, frac| state.min_p = round2(frac),
            ));
        }

        card = card.child(div().h_px().bg(border)).child(reasoning_row);

        let tokens_str = if self.state.max_tokens == 0 {
            "∞".to_string()
        } else {
            self.state.max_tokens.to_string()
        };
        card = card.child(param_row(
            "Max tokens",
            tokens_str,
            "tok-dec",
            "tok-inc",
            border,
            fg,
            hover,
            cx.listener(|this, _, _, cx| {
                this.state.max_tokens = this.state.max_tokens.saturating_sub(128);
                cx.notify();
            }),
            cx.listener(|this, _, _, cx| {
                this.state.max_tokens = (this.state.max_tokens + 128).min(8192);
                cx.notify();
            }),
        ));

        Some(
            div()
                .id("gen-settings-backdrop")
                .absolute()
                .size_full()
                .bg(black().opacity(0.4))
                .occlude()
                .on_click(cx.listener(|this, _, _, cx| {
                    this.state.gen_settings_open = false;
                    cx.notify();
                }))
                .child(card)
                .into_any_element(),
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn param_slider(
        &self,
        cx: &mut Context<Self>,
        theme: &Theme,
        label: &'static str,
        id: &'static str,
        value_text: String,
        fraction: f32,
        checkbox: Option<AnyElement>,
        set: fn(&mut ChatState, f32),
    ) -> AnyElement {
        let view = cx.entity();
        slider_param(label, checkbox, value_text, fraction, id, theme, move |frac, _, cx| {
            view.update(cx, |this, cx| {
                set(&mut this.state, frac);
                cx.notify();
            });
        })
        .into_any_element()
    }

    fn param_toggle(
        &self,
        cx: &mut Context<Self>,
        theme: &Theme,
        id: &'static str,
        on: bool,
        toggle: fn(&mut ChatState),
    ) -> AnyElement {
        let view = cx.entity();
        param_checkbox(id, on, theme, move |_, _, cx| {
            view.update(cx, |this, cx| {
                toggle(&mut this.state);
                cx.notify();
            });
        })
        .into_any_element()
    }
}
