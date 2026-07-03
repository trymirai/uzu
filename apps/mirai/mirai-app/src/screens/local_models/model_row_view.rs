use gpui::{Context, CursorStyle, IntoElement, SharedString, div, prelude::*, px, relative};

use super::{event::LocalModelsEvent, model_view_model::ModelViewModel, view::LocalModelsView};
use crate::{
    components::{Icon, IconButton, IconEl, VendorIcon},
    theme::ActiveTheme,
    tokens,
};

impl LocalModelsView {
    pub(super) fn model_row(
        &self,
        cx: &mut Context<Self>,
        vm: &ModelViewModel,
        vendor: &str,
        icon_url: Option<&str>,
    ) -> impl IntoElement {
        let theme = cx.theme().clone();
        let hover = theme.bg_hover;
        let id = vm.id.clone();

        let vendor_icon =
            VendorIcon::new(vendor.to_string()).size(tokens::icon::XL).icon_url(icon_url.map(|u| u.to_string()));
        let name_label = div()
            .flex()
            .items_center()
            .gap_2()
            .child(div().text_sm().text_color(theme.text).child(vm.name.clone()))
            .when(vm.recommended, |el| el.child(self.chip("Recommended".to_string(), true, cx)));
        let chat_id = id.clone();
        let info = div()
            .id(SharedString::from(format!("use-{}", vm.id)))
            .flex_1()
            .flex()
            .items_center()
            .gap_2()
            .h_full()
            .when(vm.installed(), |el| {
                el.cursor(CursorStyle::PointingHand).on_click(cx.listener(move |this, _, _, cx| {
                    if let Some(model) =
                        this.store.read(cx).rows.iter().find(|r| r.id() == chat_id).map(|r| r.model.clone())
                    {
                        cx.emit(LocalModelsEvent::UseModel(model));
                    }
                }))
            })
            .child(vendor_icon)
            .child(name_label)
            .into_any_element();

        let action = if vm.installed() {
            let del_id = id.clone();
            let del_name = vm.name.clone();
            div()
                .flex()
                .items_center()
                .gap_2()
                .child(IconEl::new(Icon::Check, theme.success).size(15.))
                .child(
                    IconButton::new(SharedString::from(format!("del-{}", vm.id)), Icon::Trash)
                        .color(theme.text_muted)
                        .on_click(cx.listener(move |this, _, _, cx| {
                            this.confirm_delete = Some((del_id.clone(), del_name.clone()));
                            cx.notify();
                        })),
                )
                .into_any_element()
        } else if vm.downloading() || vm.paused() {
            let toggle_id = id.clone();
            let cancel_id = id.clone();
            div()
                .flex()
                .items_center()
                .gap_1p5()
                .child(div().text_sm().text_color(theme.text_muted).child(format!("{:.0}%", vm.progress * 100.0)))
                .child(
                    IconButton::new(
                        SharedString::from(format!("tog-{}", vm.id)),
                        if vm.paused() {
                            Icon::Download
                        } else {
                            Icon::Pause
                        },
                    )
                    .icon_size(14.)
                    .hit_size(26.)
                    .color(theme.text)
                    .background(theme.bg_hover)
                    .on_click(cx.listener(move |this, _, _, cx| {
                        let id = toggle_id.clone();
                        this.store.update(cx, |s, cx| s.toggle_download(id, cx));
                    })),
                )
                .child(
                    IconButton::new(SharedString::from(format!("cancel-{}", vm.id)), Icon::Close)
                        .icon_size(14.)
                        .hit_size(26.)
                        .color(theme.text)
                        .background(theme.bg_hover)
                        .on_click(cx.listener(move |this, _, _, cx| {
                            let id = cancel_id.clone();
                            this.store.update(cx, |s, cx| s.delete(id, cx));
                        })),
                )
                .into_any_element()
        } else {
            let dl_id = id.clone();
            IconButton::new(SharedString::from(format!("dl-{}", vm.id)), Icon::Download)
                .color(theme.text_muted)
                .on_click(cx.listener(move |this, _, _, cx| {
                    let id = dl_id.clone();
                    this.store.update(cx, |s, cx| s.toggle_download(id, cx));
                }))
                .into_any_element()
        };

        let active_dl = vm.downloading() || vm.paused();
        let track = theme.bg_hover;
        let fill = theme.text;
        let content = div()
            .flex()
            .items_center()
            .gap_4()
            .h(px(52.))
            .w_full()
            .child(info)
            .child(div().w(px(80.)).text_sm().text_color(theme.text_muted).child(vm.size.clone()))
            .child(div().w(px(140.)).text_sm().text_color(theme.text_muted).child(vm.quant.clone()))
            .child(div().w(px(100.)).flex().justify_end().child(action));

        div()
            .id(SharedString::from(format!("model-{}", vm.id)))
            .flex()
            .flex_col()
            .px_4()
            .rounded_lg()
            .border_1()
            .border_color(theme.border)
            .bg(theme.card)
            .cursor(CursorStyle::PointingHand)
            .when(vm.is_mirai, |el| el.bg(theme.success.opacity(0.12)))
            .hover(move |s| s.bg(hover))
            .child(content)
            .when(active_dl, |el| {
                el.child(
                    div().w_full().pb(px(6.)).child(
                        div()
                            .w_full()
                            .h(px(4.))
                            .rounded_full()
                            .bg(track)
                            .child(div().h_full().w(relative(vm.progress.clamp(0., 1.))).rounded_full().bg(fill)),
                    ),
                )
            })
    }
}
