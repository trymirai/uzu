use std::time::Duration;

use gpui::{App, Context, Global, IntoElement, SharedString, Subscription, div, prelude::*, px};

use crate::theme::ActiveTheme;

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum ToastKind {
    Info,
    Success,
    Error,
}

struct Toast {
    id: u64,
    message: SharedString,
    kind: ToastKind,
}

#[derive(Default)]
struct GlobalToasts {
    items: Vec<Toast>,
    next_id: u64,
}

impl Global for GlobalToasts {}

pub fn init(cx: &mut App) {
    cx.set_global(GlobalToasts::default());
}

pub fn push<T: 'static>(
    cx: &mut Context<T>,
    message: impl Into<SharedString>,
    kind: ToastKind,
) {
    let message = message.into();
    let id = cx.update_global::<GlobalToasts, _>(|toasts, _| {
        toasts.next_id += 1;
        let id = toasts.next_id;
        toasts.items.push(Toast {
            id,
            message,
            kind,
        });
        id
    });
    cx.spawn(async move |_this, cx| {
        cx.background_executor().timer(Duration::from_secs(4)).await;
        let _ = cx.update(|cx| {
            if cx.has_global::<GlobalToasts>() {
                cx.update_global::<GlobalToasts, _>(|toasts, _| {
                    toasts.items.retain(|t| t.id != id);
                });
            }
        });
    })
    .detach();
}

pub fn observe<V: 'static>(
    cx: &mut Context<V>,
    mut on_change: impl FnMut(&mut V, &mut Context<V>) + 'static,
) -> Subscription {
    cx.observe_global::<GlobalToasts>(move |this, cx| on_change(this, cx))
}

pub fn render_overlay(cx: &App) -> Option<gpui::AnyElement> {
    let toasts = cx.try_global::<GlobalToasts>()?;
    if toasts.items.is_empty() {
        return None;
    }
    let theme = cx.theme().clone();
    let mut col = div().absolute().top_8().right_4().flex().flex_col().gap_2();
    for toast in &toasts.items {
        let (bg, fg) = match toast.kind {
            ToastKind::Success => (theme.success, gpui::white()),
            ToastKind::Error => (theme.error, gpui::white()),
            ToastKind::Info => (theme.card, theme.text),
        };
        col = col.child(
            div()
                .min_w(px(220.))
                .max_w(px(360.))
                .px_3()
                .py_2()
                .rounded_lg()
                .border_1()
                .border_color(theme.border)
                .bg(bg)
                .text_color(fg)
                .text_sm()
                .child(toast.message.clone()),
        );
    }
    Some(col.into_any_element())
}
