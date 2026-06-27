use std::collections::HashMap;

use gpui::{Context, Entity, EventEmitter, FontWeight, IntoElement, Render, SharedString, Window, div, prelude::*, px};

use super::{event::CloudEvent, vm::CloudVm};
use crate::{
    components::{Button, ButtonKind, ButtonSize, Icon, IconEl, Loader, TextInput, VendorIcon},
    engine,
    models_store::ModelsStore,
    provider_keys::{self, CloudProvider},
    theme::{ActiveTheme, layout::CONTENT_MAX_WIDTH},
};

pub struct CloudModelsView {
    store: Entity<ModelsStore>,
    configured: HashMap<&'static str, bool>,
    key_editor: Option<&'static str>,
    key_input: Entity<TextInput>,
    key_busy: bool,
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

    fn open_key_editor(
        &mut self,
        provider: &'static CloudProvider,
        cx: &mut Context<Self>,
    ) {
        self.key_editor = Some(provider.id);
        self.key_input.update(cx, |input, cx| {
            input.set_text(String::new(), cx);
        });
        cx.notify();

        let Some(engine) = engine::try_engine(cx) else {
            return;
        };
        let settings_key = provider.settings_key;
        let input = self.key_input.clone();
        cx.spawn(async move |_, cx| {
            let existing = provider_keys::load_key(&engine, settings_key).await.unwrap_or_default();
            input.update(cx, |field, cx| {
                field.set_text(existing, cx);
            });
        })
        .detach();
    }

    fn close_key_editor(
        &mut self,
        cx: &mut Context<Self>,
    ) {
        self.key_editor = None;
        self.key_busy = false;
        cx.notify();
    }

    fn save_key(
        &mut self,
        cx: &mut Context<Self>,
    ) {
        let Some(provider_id) = self.key_editor else {
            return;
        };
        let value = self.key_input.read(cx).text().trim().to_string();
        if value.is_empty() {
            crate::toast::push(cx, "Enter an API key", crate::toast::ToastKind::Info);
            return;
        }
        self.key_busy = true;
        cx.notify();

        let Some(engine) = engine::try_engine(cx) else {
            self.key_busy = false;
            crate::toast::push(cx, "Engine unavailable", crate::toast::ToastKind::Error);
            return;
        };
        let store = self.store.clone();
        let view = cx.entity();
        cx.spawn(async move |_, cx| {
            let result = provider_keys::set_provider_key(&engine, provider_id, Some(value)).await;
            view.update(cx, |this, cx| {
                this.key_busy = false;
                match result {
                    Ok(()) => {
                        this.configured.insert(provider_id, true);
                        this.key_editor = None;
                        store.update(cx, |s, cx| s.reload(cx));
                        crate::toast::push(cx, "Provider connected", crate::toast::ToastKind::Success);
                    },
                    Err(err) => {
                        crate::toast::push(cx, format!("Failed: {err}"), crate::toast::ToastKind::Error);
                    },
                }
                cx.notify();
            });
        })
        .detach();
    }

    fn remove_key(
        &mut self,
        cx: &mut Context<Self>,
    ) {
        let Some(provider_id) = self.key_editor else {
            return;
        };
        self.key_busy = true;
        cx.notify();

        let Some(engine) = engine::try_engine(cx) else {
            self.key_busy = false;
            return;
        };
        let store = self.store.clone();
        let view = cx.entity();
        cx.spawn(async move |_, cx| {
            let result = provider_keys::set_provider_key(&engine, provider_id, None).await;
            view.update(cx, |this, cx| {
                this.key_busy = false;
                match result {
                    Ok(()) => {
                        this.configured.insert(provider_id, false);
                        this.key_editor = None;
                        store.update(cx, |s, cx| s.reload(cx));
                        crate::toast::push(cx, "Provider disconnected", crate::toast::ToastKind::Info);
                    },
                    Err(err) => {
                        crate::toast::push(cx, format!("Failed: {err}"), crate::toast::ToastKind::Error);
                    },
                }
                cx.notify();
            });
        })
        .detach();
    }

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
                    .child(VendorIcon::new(provider.label.to_string()).size(crate::tokens::icon::MD))
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

    fn connectors_section(
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

    fn key_modal(
        &self,
        cx: &mut Context<Self>,
    ) -> Option<impl IntoElement> {
        let provider = self.key_editor?;
        let provider = provider_keys::provider_by_id(provider)?;
        let theme = cx.theme().clone();
        let connected = self.configured.get(provider.id).copied().unwrap_or(false);
        let busy = self.key_busy;

        let mut actions = div().flex().justify_end().gap_2();
        actions = actions.child(
            Button::new("key-cancel", "Cancel")
                .kind(ButtonKind::Secondary)
                .size(ButtonSize::Small)
                .disabled(busy)
                .on_click(cx.listener(|this, _, _, cx| this.close_key_editor(cx))),
        );
        if connected {
            actions = actions.child(
                Button::new("key-remove", "Remove")
                    .kind(ButtonKind::Danger)
                    .size(ButtonSize::Small)
                    .disabled(busy)
                    .on_click(cx.listener(|this, _, _, cx| this.remove_key(cx))),
            );
        }
        actions = actions.child(
            Button::new("key-save", "Save")
                .kind(ButtonKind::Primary)
                .size(ButtonSize::Small)
                .disabled(busy)
                .on_click(cx.listener(|this, _, _, cx| this.save_key(cx))),
        );

        Some(
            div()
                .absolute()
                .size_full()
                .flex()
                .items_center()
                .justify_center()
                .bg(gpui::black().opacity(0.5))
                .occlude()
                .child(
                    div()
                        .w(px(400.))
                        .flex()
                        .flex_col()
                        .gap_3()
                        .p_4()
                        .rounded_xl()
                        .bg(theme.card)
                        .border_1()
                        .border_color(theme.border)
                        .child(
                            div()
                                .text_color(theme.text)
                                .font_weight(FontWeight::MEDIUM)
                                .child(format!("Connect {}", provider.label)),
                        )
                        .child(
                            div()
                                .text_xs()
                                .text_color(theme.text_muted)
                                .child("Stored securely in the system keychain."),
                        )
                        .child(self.key_input.clone())
                        .child(actions),
                ),
        )
    }

    fn row(
        &self,
        cx: &mut Context<Self>,
        vm: &CloudVm,
    ) -> impl IntoElement {
        let theme = cx.theme().clone();
        let hover = theme.bg_hover;
        let id = vm.id.clone();
        // Whole row is clickable → starts a chat (mirai-chat ModelCard).
        div()
            .id(gpui::SharedString::from(format!("use-{}", vm.id)))
            .flex()
            .items_center()
            .gap_3()
            .h(px(52.))
            .px_3()
            .rounded_lg()
            .cursor(gpui::CursorStyle::PointingHand)
            .hover(move |s| s.bg(hover))
            .on_click(cx.listener(move |this, _, _, cx| {
                if let Some(model) = this.store.read(cx).rows.iter().find(|r| r.id() == id).map(|r| r.model.clone()) {
                    cx.emit(CloudEvent::UseModel(model));
                }
            }))
            .child(VendorIcon::new(vm.vendor.clone()).size(crate::tokens::icon::XL).icon_url(vm.icon_url.clone()))
            .child(
                div()
                    .flex_1()
                    .min_w_0()
                    .text_sm()
                    .font_weight(FontWeight::MEDIUM)
                    .text_color(theme.text)
                    .child(vm.name.clone()),
            )
            .child(IconEl::new(Icon::ChevronRight, theme.text_muted).size(crate::tokens::icon::MD))
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
                            .child(VendorIcon::new(vm.vendor.clone()).size(crate::tokens::icon::MD))
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
                        .child(IconEl::new(Icon::ModelMenu, theme.text).size(crate::tokens::icon::XXL))
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
