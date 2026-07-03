use std::collections::HashMap;

use gpui::{Context, Entity, EventEmitter, FontWeight, IntoElement, Render, Window, div, prelude::*, px};

use super::{event::CloudEvent, vm::CloudVm};
use crate::{
    components::{Icon, IconEl, Loader, TextInput, VendorIcon},
    engine,
    models_store::ModelsStore,
    provider_keys,
    theme::{ActiveTheme, layout::CONTENT_MAX_WIDTH},
    tokens,
};

pub struct CloudModelsView {
    pub(super) store: Entity<ModelsStore>,
    pub(super) configured: HashMap<&'static str, bool>,
    pub(super) key_editor: Option<&'static str>,
    pub(super) key_input: Entity<TextInput>,
    pub(super) key_busy: bool,
}

impl EventEmitter<CloudEvent> for CloudModelsView {}

impl CloudModelsView {
    pub fn new(
        store: Entity<ModelsStore>,
        cx: &mut Context<Self>,
    ) -> Self {
        let key_input = cx.new(|cx| TextInput::new(cx, "Paste API key…"));
        cx.observe(&store, |_, _, cx| cx.notify()).detach();

        let Some(engine) = engine::try_engine(cx) else {
            return Self {
                store,
                configured: HashMap::new(),
                key_editor: None,
                key_input,
                key_busy: false,
            };
        };
        cx.spawn(async move |this, cx| {
            let list = provider_keys::configured_providers(&engine).await;
            let _ = this.update(cx, |view, cx| {
                view.configured.clear();
                for (provider, on) in list {
                    view.configured.insert(provider.id, on);
                }
                cx.notify();
            });
        })
        .detach();

        Self {
            store,
            configured: HashMap::new(),
            key_editor: None,
            key_input,
            key_busy: false,
        }
    }
}

impl Render for CloudModelsView {
    fn render(
        &mut self,
        _window: &mut Window,
        cx: &mut Context<Self>,
    ) -> impl IntoElement {
        let theme = cx.theme().clone();

        let (loading, mut models): (bool, Vec<CloudVm>) = {
            let store = self.store.read(cx);
            let rows = store.rows.iter().map(|r| CloudVm::from_row(r, theme.dark)).collect();
            (store.loading, rows)
        };
        models.sort_by(|a, b| a.vendor.cmp(&b.vendor).then(a.name.cmp(&b.name)));

        let mut list = div().flex().flex_col().gap_1().pb_6();
        list = list.child(self.connectors_section(cx));

        if !models.is_empty() {
            list = list.child(
                div()
                    .pt_4()
                    .pb_2()
                    .text_sm()
                    .font_weight(FontWeight::MEDIUM)
                    .text_color(theme.text)
                    .child("Available Models"),
            );
        }

        if models.is_empty() {
            if loading {
                list = list.child(div().py_8().flex().justify_center().child(Loader::new().label("Loading…")));
            } else {
                list = list.child(div().py_4().text_color(theme.text_muted).child("No cloud models available."));
            }
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
                            .child(VendorIcon::new(vm.vendor.clone()).size(tokens::icon::MD))
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

        let modal = self.key_modal(cx);

        let mut root = div().size_full().relative().flex().flex_col().items_center().child(
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
                        .child(IconEl::new(Icon::ModelMenu, theme.text).size(tokens::icon::XXL))
                        .child(div().text_xl().font_weight(FontWeight::MEDIUM).child("Choose cloud model to chat")),
                )
                .child(div().id("cloud-list").flex_1().min_h_0().overflow_y_scroll().child(list)),
        );
        if let Some(modal) = modal {
            root = root.child(modal);
        }
        root
    }
}
