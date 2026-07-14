use gpui::{Context, CursorStyle, FontWeight, IntoElement, SharedString, div, prelude::*, px, transparent_black};

use super::{view::RoutersView, vm::RouterVm};
use crate::{
    components::{Icon, IconButton, IconEl, VendorIcon},
    theme::{ActiveTheme, Theme},
    tokens,
};

impl RoutersView {
    pub(super) fn router_row(
        &self,
        cx: &mut Context<Self>,
        vm: &RouterVm,
        selected: bool,
    ) -> impl IntoElement {
        let theme = cx.theme().clone();
        let id = vm.id.clone();

        let control = if vm.installed {
            let del = id.clone();
            div()
                .flex()
                .items_center()
                .gap_1()
                .child(IconEl::new(Icon::Check, theme.success).size(15.))
                .child(
                    IconButton::new(SharedString::from(format!("del-{}", vm.id)), Icon::Trash)
                        .color(theme.text_muted)
                        .icon_size(15.)
                        .on_click(cx.listener(move |this, _, _, cx| {
                            let del = del.clone();
                            this.store.update(cx, |s, cx| s.delete(del, cx));
                        })),
                )
                .into_any_element()
        } else if vm.downloading || vm.paused {
            let toggle = id.clone();
            div()
                .flex()
                .items_center()
                .gap_2()
                .child(div().text_xs().text_color(theme.text_muted).child(format!("{:.0}%", vm.progress * 100.0)))
                .child(
                    IconButton::new(
                        SharedString::from(format!("tog-{}", vm.id)),
                        if vm.paused {
                            Icon::Download
                        } else {
                            Icon::Pause
                        },
                    )
                    .color(theme.text_muted)
                    .icon_size(15.)
                    .on_click(cx.listener(move |this, _, _, cx| {
                        let toggle = toggle.clone();
                        this.store.update(cx, |s, cx| s.toggle_download(toggle, cx));
                    })),
                )
                .into_any_element()
        } else {
            let dl = id.clone();
            IconButton::new(SharedString::from(format!("dl-{}", vm.id)), Icon::Download)
                .color(theme.text_muted)
                .icon_size(15.)
                .on_click(cx.listener(move |this, _, _, cx| {
                    let dl = dl.clone();
                    this.store.update(cx, |s, cx| s.toggle_download(dl, cx));
                }))
                .into_any_element()
        };

        let select_id = id.clone();
        let bg = if selected {
            theme.bg_hover
        } else {
            transparent_black()
        };
        div()
            .id(SharedString::from(vm.id.clone()))
            .flex()
            .items_center()
            .gap_3()
            .h(px(56.))
            .px_3()
            .rounded_lg()
            .bg(bg)
            .when(vm.installed, |el| el.cursor(CursorStyle::PointingHand))
            .on_click(cx.listener(move |this, _, _, cx| {
                if let Some(model) = this.store.read(cx).installed_model_by_id(&select_id) {
                    this.selected = Some(model);
                    cx.notify();
                }
            }))
            .child(VendorIcon::new(vm.vendor.clone()).size(tokens::icon::XL).icon_url(vm.icon_url.clone()))
            .child(
                div()
                    .flex_1()
                    .min_w_0()
                    .flex()
                    .flex_col()
                    .child(
                        div().text_sm().font_weight(FontWeight::MEDIUM).text_color(theme.text).child(vm.name.clone()),
                    )
                    .child(div().text_xs().text_color(theme.text_muted).child(vm.vendor.clone())),
            )
            .child(div().text_xs().text_color(theme.text_muted).child(vm.size.clone()))
            .child(control)
            .child(IconEl::new(Icon::ChevronRight, theme.text_muted).size(tokens::icon::MD))
    }
}

pub(super) fn router_section(
    label: &str,
    theme: &Theme,
) -> impl IntoElement {
    div()
        .pt_4()
        .pb_1()
        .px_3()
        .text_xs()
        .font_weight(FontWeight::MEDIUM)
        .text_color(theme.text_muted)
        .child(label.to_uppercase())
}

pub(super) fn tag_chip(
    label: &str,
    prob: f64,
    theme: &Theme,
) -> impl IntoElement {
    div()
        .flex()
        .items_center()
        .gap_2()
        .px_3()
        .py_1()
        .rounded_full()
        .border_1()
        .border_color(theme.border)
        .bg(theme.bg_sub)
        .text_sm()
        .text_color(theme.text)
        .child(label.to_string())
        .child(div().text_xs().text_color(theme.text_muted).child(format!("{:.0}%", prob * 100.0)))
}
