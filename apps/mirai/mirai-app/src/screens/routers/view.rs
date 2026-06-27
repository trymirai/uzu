//! Routers screen: list classification ("router") models with download/delete,
//! pick one, and classify input text in a playground (probability bars).

use futures::{StreamExt, channel::mpsc};
use gpui::{Context, CursorStyle, Entity, FontWeight, IntoElement, Render, Window, div, prelude::*, px};
use uzu::types::{model::Model, session::classification::ClassificationMessage};

use super::vm::RouterVm;
use crate::{
    components::{Button, ButtonKind, Icon, IconButton, IconEl, InputEvent, Loader, TextInput, VendorIcon},
    engine,
    models_store::ModelsStore,
    theme::{ActiveTheme, Theme, layout::CONTENT_MAX_WIDTH},
};

enum ClassMsg {
    Ok(Vec<(String, f64)>),
    Err(String),
}

pub struct RoutersView {
    store: Entity<ModelsStore>,
    input: Entity<TextInput>,
    selected: Option<Model>,
    result: Option<Vec<(String, f64)>>,
    classifying: bool,
    error: Option<String>,
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
        }
    }

    fn clear(
        &mut self,
        cx: &mut Context<Self>,
    ) {
        self.input.update(cx, |input, cx| input.set_text("", cx));
        self.result = None;
        self.error = None;
        cx.notify();
    }

    fn resolved_router(
        &self,
        cx: &Context<Self>,
    ) -> Option<Model> {
        self.store.read(cx).resolve_installed(self.selected.as_ref())
    }

    fn classify(
        &mut self,
        text: String,
        cx: &mut Context<Self>,
    ) {
        if self.classifying {
            return;
        }
        let text = text.trim().to_string();
        if text.is_empty() {
            return;
        }
        let Some(model) = self.resolved_router(cx) else {
            self.error = Some("Download and select a router first.".to_string());
            cx.notify();
            return;
        };
        self.selected = Some(model.clone());
        self.classifying = true;
        self.error = None;
        self.result = None;
        cx.notify();

        let Some(engine) = engine::try_engine(cx) else {
            self.classifying = false;
            self.error = Some("engine unavailable".to_string());
            cx.notify();
            return;
        };

        let (tx, mut rx) = mpsc::unbounded::<ClassMsg>();
        gpui_tokio::Tokio::spawn(cx, async move {
            match engine.classification(model).await {
                Ok(session) => match session.classify(vec![ClassificationMessage::user(text)]).await {
                    Ok(output) => {
                        let mut values: Vec<(String, f64)> = output.probabilities.values.into_iter().collect();
                        values.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                        let _ = tx.unbounded_send(ClassMsg::Ok(values));
                    },
                    Err(err) => {
                        let _ = tx.unbounded_send(ClassMsg::Err(format!("{err:?}")));
                    },
                },
                Err(err) => {
                    let _ = tx.unbounded_send(ClassMsg::Err(err.to_string()));
                },
            }
        })
        .detach();

        cx.spawn(async move |this, cx| {
            while let Some(msg) = rx.next().await {
                if this
                    .update(cx, |view, cx| {
                        match msg {
                            ClassMsg::Ok(values) => view.result = Some(values),
                            ClassMsg::Err(err) => view.error = Some(err),
                        }
                        view.classifying = false;
                        cx.notify();
                    })
                    .is_err()
                {
                    break;
                }
            }
        })
        .detach();
    }

    fn classify_from_button(
        &mut self,
        cx: &mut Context<Self>,
    ) {
        let text = self.input.read(cx).text();
        self.classify(text, cx);
    }

    fn router_row(
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
                    IconButton::new(gpui::SharedString::from(format!("del-{}", vm.id)), Icon::Trash)
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
                        gpui::SharedString::from(format!("tog-{}", vm.id)),
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
            IconButton::new(gpui::SharedString::from(format!("dl-{}", vm.id)), Icon::Download)
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
            gpui::transparent_black()
        };
        div()
            .id(gpui::SharedString::from(vm.id.clone()))
            .flex()
            .items_center()
            .gap_3()
            .h(px(56.))
            .px_3()
            .rounded_lg()
            .bg(bg)
            .when(vm.installed, |el| el.cursor(CursorStyle::PointingHand))
            .on_click(cx.listener(move |this, _, _, cx| {
                if let Some(model) = this
                    .store
                    .read(cx)
                    .rows
                    .iter()
                    .find(|r| r.id() == select_id && r.is_installed())
                    .map(|r| r.model.clone())
                {
                    this.selected = Some(model);
                    cx.notify();
                }
            }))
            .child(VendorIcon::new(vm.vendor.clone()).size(crate::tokens::icon::XL).icon_url(vm.icon_url.clone()))
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
            .child(IconEl::new(Icon::ChevronRight, theme.text_muted).size(crate::tokens::icon::MD))
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

        // Split into Installed / Available sections (mirai-chat parity).
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
                        .child(IconEl::new(Icon::Routers, theme.text).size(crate::tokens::icon::XXL))
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

/// Section label ("INSTALLED" / "AVAILABLE") above a router group.
fn router_section(
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

/// A rounded probability pill for a classification result.
fn tag_chip(
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
