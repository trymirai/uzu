//! Cloud Models screen: remote chat models (from providers configured via API
//! keys at startup). Picking one opens a chat pinned to it. No downloads —
//! these run remotely.

use gpui::{
    Context, Entity, EventEmitter, FontWeight, IntoElement, Render, Window, div, prelude::*, px,
};
use uzu::types::model::Model;

use crate::{
    components::{Button, ButtonKind, ButtonSize, Icon, IconEl, VendorIcon},
    models_store::ModelsStore,
    theme::{ActiveTheme, layout::CONTENT_MAX_WIDTH},
};

/// Emitted to the shell to start a chat with the chosen cloud model.
pub enum CloudEvent {
    UseModel(Model),
}

struct CloudVm {
    id: String,
    name: String,
    vendor: String,
}

pub struct CloudModelsView {
    store: Entity<ModelsStore>,
}

impl EventEmitter<CloudEvent> for CloudModelsView {}

impl CloudModelsView {
    pub fn new(store: Entity<ModelsStore>, cx: &mut Context<Self>) -> Self {
        cx.observe(&store, |_, _, cx| cx.notify()).detach();
        Self { store }
    }

    fn row(&self, cx: &mut Context<Self>, vm: &CloudVm) -> impl IntoElement {
        let theme = cx.theme().clone();
        let id = vm.id.clone();
        div()
            .flex()
            .items_center()
            .justify_between()
            .gap_3()
            .h(px(52.))
            .px_3()
            .rounded_lg()
            .child(
                div()
                    .text_sm()
                    .font_weight(FontWeight::MEDIUM)
                    .text_color(theme.text)
                    .child(vm.name.clone()),
            )
            .child(
                Button::new(gpui::SharedString::from(format!("use-{}", vm.id)), "Chat")
                    .kind(ButtonKind::Secondary)
                    .size(ButtonSize::Small)
                    .on_click(cx.listener(move |this, _, _, cx| {
                        if let Some(model) = this
                            .store
                            .read(cx)
                            .rows
                            .iter()
                            .find(|r| r.id() == id)
                            .map(|r| r.model.clone())
                        {
                            cx.emit(CloudEvent::UseModel(model));
                        }
                    })),
            )
    }
}

impl Render for CloudModelsView {
    fn render(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let theme = cx.theme().clone();

        let (loading, mut models): (bool, Vec<CloudVm>) = {
            let store = self.store.read(cx);
            let rows = store
                .rows
                .iter()
                .map(|r| CloudVm {
                    id: r.id().to_string(),
                    name: r.name(),
                    vendor: r.vendor().unwrap_or_else(|| "Other".to_string()),
                })
                .collect();
            (store.loading, rows)
        };
        models.sort_by(|a, b| a.vendor.cmp(&b.vendor).then(a.name.cmp(&b.name)));

        let mut list = div().flex().flex_col().gap_1().pb_6();
        if models.is_empty() {
            let msg = if loading {
                "Loading…".to_string()
            } else {
                "No cloud models. Set provider API keys (e.g. OPENAI_API_KEY, ANTHROPIC_API_KEY) before launching.".to_string()
            };
            list = list.child(div().py_8().text_color(theme.text_muted).child(msg));
        } else {
            let mut current_vendor: Option<String> = None;
            for vm in &models {
                if current_vendor.as_deref() != Some(vm.vendor.as_str()) {
                    current_vendor = Some(vm.vendor.clone());
                    list = list.child(
                        div()
                            .flex()
                            .items_center()
                            .gap_2()
                            .pt_4()
                            .pb_1()
                            .px_3()
                            .child(VendorIcon::new(vm.vendor.clone()).size(16.))
                            .child(
                                div()
                                    .text_xs()
                                    .font_weight(FontWeight::MEDIUM)
                                    .text_color(theme.text_muted)
                                    .child(vm.vendor.clone()),
                            ),
                    );
                }
                list = list.child(self.row(cx, vm));
            }
        }

        div()
            .size_full()
            .flex()
            .flex_col()
            .items_center()
            .child(
                div()
                    .w_full()
                    .max_w(px(CONTENT_MAX_WIDTH))
                    .h_full()
                    .min_h_0()
                    .flex()
                    .flex_col()
                    .px_6()
                    .child(
                        div()
                            .pt_10()
                            .pb_2()
                            .flex()
                            .items_center()
                            .gap_2()
                            .child(IconEl::new(Icon::ModelMenu, theme.text).size(22.))
                            .child(
                                div()
                                    .text_xl()
                                    .font_weight(FontWeight::MEDIUM)
                                    .child("Cloud models"),
                            ),
                    )
                    .child(
                        div()
                            .id("cloud-list")
                            .flex_1()
                            .min_h_0()
                            .overflow_y_scroll()
                            .child(list),
                    ),
            )
    }
}
