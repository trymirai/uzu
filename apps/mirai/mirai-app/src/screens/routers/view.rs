use gpui::{Context, Entity, FontWeight, IntoElement, Render, Window, div, prelude::*, px};
use uzu::types::model::Model;

use super::{
    rows::{router_section, tag_chip},
    vm::RouterVm,
};
use crate::{
    components::{Button, ButtonKind, Icon, IconEl, InputEvent, Loader, TextInput},
    models_store::ModelsStore,
    theme::{ActiveTheme, layout::CONTENT_MAX_WIDTH},
    tokens,
};

pub struct RoutersView {
    pub(super) store: Entity<ModelsStore>,
    pub(super) input: Entity<TextInput>,
    pub(super) selected: Option<Model>,
    pub(super) result: Option<Vec<(String, f64)>>,
    pub(super) classifying: bool,
    pub(super) error: Option<String>,

    pub(super) classify_gen: u64,
}

impl RoutersView {
    pub fn new(
        store: Entity<ModelsStore>,
        cx: &mut Context<Self>,
    ) -> Self {
        let input = cx.new(|cx| TextInput::new(cx, "Enter text to classify…").multiline(false, 6, 18));
        cx.subscribe(&input, |_, _input, event, cx| match event {
            InputEvent::Submit(_) | InputEvent::Changed(_) => cx.notify(),
        })
        .detach();
        cx.observe(&store, |_, _, cx| cx.notify()).detach();
        Self {
            store,
            input,
            selected: None,
            result: None,
            classifying: false,
            error: None,
            classify_gen: 0,
        }
    }
}

impl Render for RoutersView {
    fn render(
        &mut self,
        _window: &mut Window,
        cx: &mut Context<Self>,
    ) -> impl IntoElement {
        let theme = cx.theme().clone();
        let selected_id = self.selected.as_ref().map(|m| m.identifier.clone());
        let resolved_name = self.resolved_router(cx).map(|m| m.name()).unwrap_or_else(|| "No router".to_string());

        let (loading, routers): (bool, Vec<RouterVm>) = {
            let store = self.store.read(cx);
            let rows = store.rows.iter().map(|r| RouterVm::from_row(r, theme.dark)).collect();
            (store.loading, rows)
        };
        let any_installed = routers.iter().any(|r| r.installed);

        let mut list = div().flex().flex_col().gap_1();
        if routers.is_empty() {
            list = list.child(if loading {
                div().py_6().flex().justify_center().child(Loader::new().label("Loading routers…")).into_any_element()
            } else {
                div().py_6().text_color(theme.text_muted).child("No routers available.").into_any_element()
            });
        } else {
            let installed: Vec<&RouterVm> = routers.iter().filter(|r| r.installed).collect();
            let available: Vec<&RouterVm> = routers.iter().filter(|r| !r.installed).collect();
            if !installed.is_empty() {
                list = list.child(router_section("Installed", &theme));
                for vm in installed {
                    let selected = selected_id.as_deref() == Some(vm.id.as_str());
                    list = list.child(self.router_row(cx, vm, selected));
                }
            }
            if !available.is_empty() {
                list = list.child(router_section("Available", &theme));
                for vm in available {
                    list = list.child(self.router_row(cx, vm, false));
                }
            }
        }

        let result_block = if self.classifying {
            div().pt_2().text_sm().text_color(theme.text_muted).child("Classifying…").into_any_element()
        } else if let Some(err) = &self.error {
            div()
                .mt_2()
                .p_3()
                .rounded_md()
                .border_1()
                .border_color(theme.error.opacity(0.3))
                .bg(theme.error.opacity(0.08))
                .text_sm()
                .text_color(theme.error)
                .child(err.clone())
                .into_any_element()
        } else if let Some(values) = &self.result {
            let mut pills = div().pt_2().flex().flex_wrap().gap_2();
            for (label, prob) in values {
                pills = pills.child(tag_chip(label, *prob, &theme));
            }
            pills.into_any_element()
        } else {
            div().into_any_element()
        };

        div().size_full().flex().flex_col().items_center().child(
            div()
                .id("routers-scroll")
                .w_full()
                .max_w(px(CONTENT_MAX_WIDTH))
                .flex_1()
                .min_h_0()
                .flex()
                .flex_col()
                .overflow_y_scroll()
                .px_6()
                .child(
                    div()
                        .pt_10()
                        .pb_3()
                        .flex()
                        .items_center()
                        .gap_2()
                        .child(IconEl::new(Icon::Routers, theme.text).size(tokens::icon::XXL))
                        .child(div().text_xl().font_weight(FontWeight::MEDIUM).child("Choose router")),
                )
                .child(div().h_px().bg(theme.border))
                .child(list)
                .child(
                    div()
                        .pt_6()
                        .pb_6()
                        .flex()
                        .flex_col()
                        .gap_2()
                        .child(
                            div()
                                .text_xs()
                                .font_weight(FontWeight::MEDIUM)
                                .text_color(theme.text_muted)
                                .child(format!("Classify with {resolved_name}")),
                        )
                        .child(
                            div()
                                .px_3()
                                .py_2()
                                .rounded_lg()
                                .border_1()
                                .border_color(theme.border)
                                .bg(theme.card)
                                .child(self.input.clone()),
                        )
                        .child(
                            div()
                                .flex()
                                .items_center()
                                .justify_end()
                                .gap_2()
                                .child(
                                    Button::new("clear", "Clear")
                                        .kind(ButtonKind::Secondary)
                                        .on_click(cx.listener(|this, _, _, cx| this.clear(cx))),
                                )
                                .child(
                                    Button::new("classify", "Classify")
                                        .kind(ButtonKind::Primary)
                                        .disabled(!any_installed || self.classifying)
                                        .on_click(cx.listener(|this, _, _, cx| this.classify_from_button(cx))),
                                ),
                        )
                        .child(result_block),
                ),
        )
    }
}
