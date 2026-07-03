use gpui::{Context, CursorStyle, FontWeight, IntoElement, SharedString, div, prelude::*, px};

use super::{event::CloudEvent, view::CloudModelsView, vm::CloudVm};
use crate::{
    components::{Button, ButtonKind, ButtonSize, Icon, IconEl, VendorIcon},
    provider_keys::{self, CloudProvider},
    theme::ActiveTheme,
    tokens,
};

impl CloudModelsView {
    fn provider_row(
        &self,
        cx: &mut Context<Self>,
        provider: &'static CloudProvider,
    ) -> impl IntoElement {
        let theme = cx.theme().clone();
        let connected = self.configured.get(provider.id).copied().unwrap_or(false);
        let label = if connected {
            "Manage"
        } else {
            "Connect"
        };
        div()
            .flex()
            .items_center()
            .justify_between()
            .gap_3()
            .h(px(44.))
            .px_3()
            .rounded_lg()
            .child(
                div()
                    .flex()
                    .items_center()
                    .gap_2()
                    .child(VendorIcon::new(provider.label.to_string()).size(tokens::icon::MD))
                    .child(
                        div()
                            .flex()
                            .flex_col()
                            .child(
                                div()
                                    .text_sm()
                                    .font_weight(FontWeight::MEDIUM)
                                    .text_color(theme.text)
                                    .child(provider.label),
                            )
                            .child(div().text_xs().text_color(theme.text_muted).child(if connected {
                                "Connected"
                            } else {
                                "Not connected"
                            })),
                    ),
            )
            .child(
                Button::new(SharedString::from(format!("connect-{}", provider.id)), label)
                    .kind(ButtonKind::Secondary)
                    .size(ButtonSize::Small)
                    .on_click(cx.listener(move |this, _, _, cx| this.open_key_editor(provider, cx))),
            )
    }

    pub(super) fn connectors_section(
        &self,
        cx: &mut Context<Self>,
    ) -> impl IntoElement {
        let theme = cx.theme().clone();
        let mut rows = div().flex().flex_col().gap_1();
        for provider in provider_keys::PROVIDERS {
            rows = rows.child(self.provider_row(cx, provider));
        }
        div()
            .flex()
            .flex_col()
            .gap_1()
            .pb_4()
            .child(
                div()
                    .pb_2()
                    .text_sm()
                    .font_weight(FontWeight::MEDIUM)
                    .text_color(theme.text)
                    .child("Connect providers"),
            )
            .child(
                div()
                    .text_xs()
                    .text_color(theme.text_muted)
                    .pb_2()
                    .child("Add an API key to load models from a cloud provider."),
            )
            .child(rows)
    }

    pub(super) fn row(
        &self,
        cx: &mut Context<Self>,
        vm: &CloudVm,
    ) -> impl IntoElement {
        let theme = cx.theme().clone();
        let hover = theme.bg_hover;
        let id = vm.id.clone();

        div()
            .id(SharedString::from(format!("use-{}", vm.id)))
            .flex()
            .items_center()
            .gap_3()
            .h(px(52.))
            .px_3()
            .rounded_lg()
            .cursor(CursorStyle::PointingHand)
            .hover(move |s| s.bg(hover))
            .on_click(cx.listener(move |this, _, _, cx| {
                if let Some(model) = this.store.read(cx).rows.iter().find(|r| r.id() == id).map(|r| r.model.clone()) {
                    cx.emit(CloudEvent::UseModel(model));
                }
            }))
            .child(VendorIcon::new(vm.vendor.clone()).size(tokens::icon::XL).icon_url(vm.icon_url.clone()))
            .child(
                div()
                    .flex_1()
                    .min_w_0()
                    .text_sm()
                    .font_weight(FontWeight::MEDIUM)
                    .text_color(theme.text)
                    .child(vm.name.clone()),
            )
            .child(IconEl::new(Icon::ChevronRight, theme.text_muted).size(tokens::icon::MD))
    }
}
