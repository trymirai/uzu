//! Routers screen: list classification ("router") models with download/delete,
//! pick one, and classify input text in a playground (probability bars).

use futures::{StreamExt, channel::mpsc};
use gpui::{
    Context, CursorStyle, Entity, FontWeight, IntoElement, Render, Window, div, prelude::*, px,
};
use uzu::{
    storage::types::DownloadPhase,
    types::{model::Model, session::classification::ClassificationMessage},
};

use crate::{
    components::{
        Button, ButtonKind, ButtonSize, Icon, IconButton, IconEl, InputEvent, Loader, TextInput,
    },
    engine,
    models_store::ModelsStore,
    theme::{ActiveTheme, layout::CONTENT_MAX_WIDTH},
};

enum ClassMsg {
    Ok(Vec<(String, f64)>),
    Err(String),
}

struct RouterVm {
    id: String,
    name: String,
    vendor: String,
    installed: bool,
    downloading: bool,
    progress: f32,
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
    pub fn new(store: Entity<ModelsStore>, cx: &mut Context<Self>) -> Self {
        let input = cx.new(|cx| TextInput::new(cx, "Text to classify…"));
        cx.subscribe(&input, |this, _input, event, cx| match event {
            InputEvent::Submit(text) => this.classify(text.clone(), cx),
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

    fn resolved_router(&self, cx: &Context<Self>) -> Option<Model> {
        self.selected.clone().or_else(|| {
            self.store
                .read(cx)
                .rows
                .iter()
                .find(|r| r.is_installed())
                .map(|r| r.model.clone())
        })
    }

    fn classify(&mut self, text: String, cx: &mut Context<Self>) {
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
                Ok(session) => {
                    match session.classify(vec![ClassificationMessage::user(text)]).await {
                        Ok(output) => {
                            let mut values: Vec<(String, f64)> =
                                output.probabilities.values.into_iter().collect();
                            values.sort_by(|a, b| {
                                b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
                            });
                            let _ = tx.unbounded_send(ClassMsg::Ok(values));
                        }
                        Err(err) => {
                            let _ = tx.unbounded_send(ClassMsg::Err(format!("{err:?}")));
                        }
                    }
                }
                Err(err) => {
                    let _ = tx.unbounded_send(ClassMsg::Err(err.to_string()));
                }
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

    fn classify_from_button(&mut self, cx: &mut Context<Self>) {
        let text = self.input.read(cx).text();
        self.classify(text, cx);
    }

    fn router_row(&self, cx: &mut Context<Self>, vm: &RouterVm, selected: bool) -> impl IntoElement {
        let theme = cx.theme().clone();
        let id = vm.id.clone();

        let control = if vm.installed {
            let id = id.clone();
            IconButton::new(gpui::SharedString::from(format!("del-{}", vm.id)), Icon::Trash)
                .color(theme.text_muted)
                .on_click(cx.listener(move |this, _, _, cx| {
                    let id = id.clone();
                    this.store.update(cx, |s, cx| s.delete(id, cx));
                }))
                .into_any_element()
        } else if vm.downloading {
            div()
                .text_xs()
                .text_color(theme.text_muted)
                .child(format!("{:.0}%", vm.progress * 100.0))
                .into_any_element()
        } else {
            let id = id.clone();
            Button::new(gpui::SharedString::from(format!("dl-{}", vm.id)), "Download")
                .kind(ButtonKind::Secondary)
                .size(ButtonSize::Small)
                .icon(Icon::Download)
                .on_click(cx.listener(move |this, _, _, cx| {
                    let id = id.clone();
                    this.store.update(cx, |s, cx| s.toggle_download(id, cx));
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
            .justify_between()
            .gap_3()
            .h(px(52.))
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
            .child(
                div()
                    .flex()
                    .flex_col()
                    .child(
                        div()
                            .text_sm()
                            .font_weight(FontWeight::MEDIUM)
                            .text_color(theme.text)
                            .child(vm.name.clone()),
                    )
                    .child(div().text_xs().text_color(theme.text_muted).child(vm.vendor.clone())),
            )
            .child(control)
    }
}

impl Render for RoutersView {
    fn render(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let theme = cx.theme().clone();
        let selected_id = self.selected.as_ref().map(|m| m.identifier.clone());
        let resolved_name = self
            .resolved_router(cx)
            .map(|m| m.name())
            .unwrap_or_else(|| "No router".to_string());

        let (loading, routers): (bool, Vec<RouterVm>) = {
            let store = self.store.read(cx);
            let rows = store
                .rows
                .iter()
                .map(|r| RouterVm {
                    id: r.id().to_string(),
                    name: r.name(),
                    vendor: r.vendor().unwrap_or_else(|| "Other".to_string()),
                    installed: r.is_installed(),
                    downloading: matches!(
                        r.phase(),
                        DownloadPhase::Downloading {} | DownloadPhase::Paused {}
                    ),
                    progress: r.progress(),
                })
                .collect();
            (store.loading, rows)
        };
        let any_installed = routers.iter().any(|r| r.installed);

        let mut list = div().flex().flex_col().gap_1();
        if routers.is_empty() {
            if loading {
                list = list.child(
                    div().py_6().flex().justify_center().child(Loader::new().label("Loading routers…")),
                );
            } else {
                list = list.child(
                    div().py_6().text_color(theme.text_muted).child("No router models available."),
                );
            }
        } else {
            for vm in &routers {
                let selected = selected_id.as_deref() == Some(vm.id.as_str());
                list = list.child(self.router_row(cx, vm, selected));
            }
        }

        // Result bars.
        let mut result_block = div().flex().flex_col().gap_2().pt_2();
        if self.classifying {
            result_block = result_block.child(
                div().text_sm().text_color(theme.text_muted).child("Classifying…"),
            );
        } else if let Some(err) = &self.error {
            result_block =
                result_block.child(div().text_sm().text_color(theme.error).child(err.clone()));
        } else if let Some(values) = &self.result {
            for (label, prob) in values {
                result_block = result_block.child(
                    div()
                        .flex()
                        .items_center()
                        .gap_2()
                        .child(
                            div()
                                .w(px(160.))
                                .text_sm()
                                .text_color(theme.text)
                                .child(label.clone()),
                        )
                        .child(
                            div()
                                .flex_1()
                                .h(px(6.))
                                .rounded_full()
                                .bg(theme.bg_sub)
                                .child(
                                    div()
                                        .h(px(6.))
                                        .w(px(200.0 * (*prob as f32).clamp(0.0, 1.0)))
                                        .rounded_full()
                                        .bg(theme.accent),
                                ),
                        )
                        .child(
                            div()
                                .w(px(48.))
                                .text_xs()
                                .text_color(theme.text_muted)
                                .child(format!("{:.1}%", prob * 100.0)),
                        ),
                );
            }
        }

        div()
            .size_full()
            .flex()
            .flex_col()
            .items_center()
            .child(
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
                            .pb_2()
                            .flex()
                            .items_center()
                            .gap_2()
                            .child(IconEl::new(Icon::Routers, theme.text).size(22.))
                            .child(
                                div()
                                    .text_xl()
                                    .font_weight(FontWeight::MEDIUM)
                                    .child("Routers"),
                            ),
                    )
                    .child(list)
                    // Classify playground.
                    .child(
                        div()
                            .pt_6()
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
                                    .flex()
                                    .items_center()
                                    .gap_2()
                                    .child(
                                        div()
                                            .flex_1()
                                            .px_3()
                                            .py_2()
                                            .rounded_lg()
                                            .border_1()
                                            .border_color(theme.border)
                                            .bg(theme.card)
                                            .child(self.input.clone()),
                                    )
                                    .child(
                                        Button::new("classify", "Classify")
                                            .kind(ButtonKind::Primary)
                                            .disabled(!any_installed || self.classifying)
                                            .on_click(cx.listener(|this, _, _, cx| {
                                                this.classify_from_button(cx)
                                            })),
                                    ),
                            )
                            .child(result_block),
                    ),
            )
    }
}
