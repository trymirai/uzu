use gpui::{Context, FontWeight, IntoElement, black, div, prelude::*, px};

use super::view::CloudModelsView;
use crate::{
    components::{Button, ButtonKind, ButtonSize},
    engine,
    provider_keys::{self, CloudProvider},
    theme::ActiveTheme,
    toast::{self, ToastKind},
};

impl CloudModelsView {
    pub(super) fn open_key_editor(
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
        let provider_id = provider.id;
        let input = self.key_input.clone();
        cx.spawn(async move |this, cx| {
            let existing = provider_keys::load_key(&engine, settings_key).await.unwrap_or_default();
            let _ = this.update(cx, |this, cx| {
                if this.key_editor == Some(provider_id) {
                    input.update(cx, |field, cx| field.set_text(existing, cx));
                }
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
            toast::push(cx, "Enter an API key", ToastKind::Info);
            return;
        }
        self.key_busy = true;
        cx.notify();

        let Some(engine) = engine::try_engine(cx) else {
            self.key_busy = false;
            toast::push(cx, "Engine unavailable", ToastKind::Error);
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
                        toast::push(cx, "Provider connected", ToastKind::Success);
                    },
                    Err(err) => {
                        toast::push(cx, format!("Failed: {err}"), ToastKind::Error);
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
                        toast::push(cx, "Provider disconnected", ToastKind::Info);
                    },
                    Err(err) => {
                        toast::push(cx, format!("Failed: {err}"), ToastKind::Error);
                    },
                }
                cx.notify();
            });
        })
        .detach();
    }

    pub(super) fn key_modal(
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
                .bg(black().opacity(0.5))
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
}
