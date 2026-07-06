use gpui::{
    AnyElement, Context, CursorStyle, FontWeight, IntoElement, SharedString, deferred, div, prelude::*, px,
    transparent_black,
};

use super::{family_view_model::FamilyViewModel, model_view_model::ModelViewModel, view::LocalModelsView};
use crate::{
    components::{Icon, IconEl, Loader, VendorIcon},
    model_sort::ModelSort,
    theme::ActiveTheme,
    tokens,
};

impl LocalModelsView {
    pub(super) fn chip(
        &self,
        text: String,
        accent: bool,
        cx: &Context<Self>,
    ) -> impl IntoElement {
        let theme = cx.theme().clone();

        let (fg, bg, border) = if accent {
            (theme.success, theme.success.opacity(0.08), theme.success.opacity(0.45))
        } else {
            (theme.text_muted, transparent_black(), theme.button_border)
        };
        div()
            .flex()
            .items_center()
            .h(px(26.))
            .px_2p5()
            .rounded_lg()
            .border_1()
            .border_color(border)
            .bg(bg)
            .text_color(fg)
            .text_xs()
            .child(text)
    }

    pub(super) fn family_row(
        &self,
        cx: &mut Context<Self>,
        fam: &FamilyViewModel,
    ) -> impl IntoElement {
        let theme = cx.theme().clone();
        let hover = theme.bg_hover;
        let key = fam.key.clone();
        let installed = fam.installed_count();
        let total = fam.models.len();

        let mut chips = div().flex().items_center().gap_2();
        if fam.has_mirai {
            chips = chips.child(self.chip("Mirai quantizations".to_string(), true, cx));
        }
        chips = chips.child(self.chip(
            format!(
                "{total} model{}",
                if total == 1 {
                    ""
                } else {
                    "s"
                }
            ),
            false,
            cx,
        ));
        if installed > 0 {
            chips = chips.child(self.chip(
                format!(
                    "{installed} installed model{}",
                    if installed == 1 {
                        ""
                    } else {
                        "s"
                    }
                ),
                true,
                cx,
            ));
        }
        if let Some(range) = &fam.range {
            chips = chips.child(self.chip(range.clone(), false, cx));
        }

        div()
            .id(SharedString::from(format!("fam-{}", fam.key)))
            .flex()
            .items_center()
            .gap_3()
            .h(px(56.))
            .px_4()
            .rounded_lg()
            .border_1()
            .border_color(theme.border)
            .bg(theme.card)
            .cursor(CursorStyle::PointingHand)
            .hover(move |s| s.bg(hover))
            .on_click(cx.listener(move |this, _, _, cx| this.open_family(key.clone(), cx)))
            .child(VendorIcon::new(fam.vendor.clone()).size(tokens::icon::XL).icon_url(fam.icon_url.clone()))
            .child(
                div()
                    .flex_1()
                    .flex()
                    .items_baseline()
                    .gap_2()
                    .child(div().font_weight(FontWeight::MEDIUM).text_color(theme.text).child(fam.name.clone()))
                    .child(div().text_sm().text_color(theme.text_muted).child(format!("from {}", fam.vendor))),
            )
            .child(chips)
            .child(IconEl::new(Icon::ChevronRight, theme.text_muted).size(tokens::icon::MD))
    }

    pub(super) fn recommended_row(
        &self,
        cx: &mut Context<Self>,
        vm: &ModelViewModel,
        family_key: &str,
    ) -> impl IntoElement {
        let theme = cx.theme().clone();
        let hover = theme.bg_hover;
        let key = family_key.to_string();
        let name = vm.name.clone();
        div()
            .id("recommended-model")
            .flex()
            .items_center()
            .gap_3()
            .h(px(56.))
            .px_4()
            .mb_2()
            .rounded_lg()
            .border_1()
            .border_color(theme.success.opacity(0.45))
            .bg(theme.success.opacity(0.08))
            .cursor(CursorStyle::PointingHand)
            .hover(move |s| s.bg(hover))
            .on_click(cx.listener(move |this, _, _, cx| this.open_family(key.clone(), cx)))
            .child(
                div()
                    .flex_1()
                    .flex()
                    .flex_col()
                    .gap_0p5()
                    .child(div().text_xs().text_color(theme.success).child("Recommended for your device"))
                    .child(div().text_sm().font_weight(FontWeight::MEDIUM).text_color(theme.text).child(name)),
            )
            .child(IconEl::new(Icon::ChevronRight, theme.text_muted).size(tokens::icon::MD))
    }

    pub(super) fn sort_control(
        &self,
        cx: &mut Context<Self>,
    ) -> impl IntoElement {
        let theme = cx.theme().clone();
        let hover = theme.bg_hover;
        let label = self.sort.label();
        let mut menu = div().flex().flex_col().gap_0p5();
        for sort in [ModelSort::Size, ModelSort::Name, ModelSort::Newest] {
            let active = self.sort == sort;
            menu = menu.child(
                div()
                    .id(SharedString::from(format!("sort-{label}", label = sort.label())))
                    .px_3()
                    .py_1p5()
                    .rounded_md()
                    .text_sm()
                    .text_color(if active {
                        theme.text
                    } else {
                        theme.text_muted
                    })
                    .cursor(CursorStyle::PointingHand)
                    .hover(move |s| s.bg(hover))
                    .on_click(cx.listener(move |this, _, _, cx| {
                        this.sort = sort;
                        this.sort_open = false;
                        cx.notify();
                    }))
                    .child(sort.label()),
            );
        }
        div()
            .relative()
            .child(
                div()
                    .id("sort-trigger")
                    .flex()
                    .items_center()
                    .gap_1()
                    .h(px(32.))
                    .px_2p5()
                    .rounded_lg()
                    .border_1()
                    .border_color(theme.border)
                    .bg(theme.card)
                    .cursor(CursorStyle::PointingHand)
                    .on_click(cx.listener(|this, _, _, cx| {
                        this.sort_open = !this.sort_open;
                        cx.notify();
                    }))
                    .child(div().text_sm().text_color(theme.text_muted).child(format!("Sort: {label}")))
                    .child(IconEl::new(Icon::ChevronDown, theme.text_muted).size(tokens::icon::SM)),
            )
            .when(self.sort_open, |el| {
                el.child(
                    deferred(
                        div()
                            .absolute()
                            .top(px(36.))
                            .right_0()
                            .w(px(140.))
                            .p_1()
                            .rounded_lg()
                            .border_1()
                            .border_color(theme.border)
                            .bg(theme.card)
                            .shadow_md()
                            .child(menu),
                    )
                    .priority(1),
                )
            })
    }

    pub(super) fn empty_message(
        &self,
        cx: &Context<Self>,
    ) -> AnyElement {
        let theme = cx.theme().clone();
        let store = self.store.read(cx);
        if store.loading {
            return div()
                .py_8()
                .flex()
                .justify_center()
                .child(Loader::new().label("Loading models…"))
                .into_any_element();
        }
        let msg = match &store.error {
            Some(err) => format!("Failed to load models: {err}"),
            None => "No models".to_string(),
        };
        div().py_8().text_color(theme.text_muted).child(msg).into_any_element()
    }
}
