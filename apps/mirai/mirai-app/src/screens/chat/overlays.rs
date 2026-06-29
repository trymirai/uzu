use gpui::{Anchor, App, Context, FontWeight, IntoElement, anchored, deferred, div, prelude::*, px};
use uzu::types::model::Model;

use super::{
    conversation::Version,
    view::{ChatEvent, ChatView},
};
use crate::{
    components::{Icon, IconEl, VendorIcon},
    theme::ActiveTheme,
    toast::{self, ToastKind},
};

type ModelEntry = (String, Model, String, String, Option<String>);

fn model_entry(r: &crate::models_store::ModelRow) -> ModelEntry {
    (r.id().to_string(), r.model.clone(), r.name(), r.vendor().unwrap_or_default(), r.icon_url(true))
}

impl ChatView {
    fn pick_file(
        &mut self,
        cx: &mut Context<Self>,
    ) {
        const MAX_SIZE: u64 = 256 * 1024;
        const MAX_FILES: usize = 5;
        const SUPPORTED: &[&str] = &[
            "txt", "md", "markdown", "json", "csv", "tsv", "yaml", "yml", "py", "js", "ts", "tsx", "jsx", "rs", "html",
            "css", "xml",
        ];

        if self.state.attached_files.len() >= MAX_FILES {
            toast::push(cx, "Maximum 5 files per message", ToastKind::Info);
            return;
        }
        let remaining_slots = MAX_FILES - self.state.attached_files.len();

        let rx = cx.prompt_for_paths(gpui::PathPromptOptions {
            files: true,
            directories: false,
            multiple: true,
            prompt: Some("Attach text file".into()),
        });

        cx.spawn(async move |this, cx| {
            let Ok(Ok(Some(paths))) = rx.await else {
                return;
            };
            for path in paths.iter().take(remaining_slots) {
                let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("txt").to_lowercase();
                if !SUPPORTED.contains(&ext.as_str()) {
                    let _ = this.update(cx, |_, cx| {
                        toast::push(cx, format!("Unsupported file type: .{ext}"), ToastKind::Info);
                    });
                    continue;
                }
                let meta = std::fs::metadata(path);
                if meta.map(|m| m.len()).unwrap_or(0) > MAX_SIZE {
                    let _ = this.update(cx, |_, cx| {
                        toast::push(cx, "File too large (max 256 KB)", ToastKind::Info);
                    });
                    continue;
                }
                match std::fs::read_to_string(path) {
                    Ok(content) => {
                        let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("file").to_string();
                        let _ = this.update(cx, |this, cx| {
                            if this.state.attached_files.len() < MAX_FILES {
                                this.state.attached_files.push((name, ext, content));
                                cx.notify();
                            }
                        });
                    },
                    Err(_) => {
                        let _ = this.update(cx, |_, cx| {
                            toast::push(cx, "Could not read file", ToastKind::Info);
                        });
                    },
                }
            }
        })
        .detach();
    }

    fn collect_model_entries(
        &self,
        cx: &App,
    ) -> (Vec<ModelEntry>, Vec<ModelEntry>) {
        let local = self.store.read(cx).rows.iter().filter(|r| r.is_installed()).map(model_entry).collect();
        let cloud = self.cloud_store.read(cx).rows.iter().map(model_entry).collect();
        (local, cloud)
    }

    pub(super) fn anchored_popover(
        panel: impl gpui::IntoElement + 'static,
        corner: Anchor,
        close: impl Fn(&mut Self, &gpui::MouseDownEvent, &mut gpui::Window, &mut gpui::Context<Self>) + 'static,
        cx: &mut Context<Self>,
    ) -> gpui::AnyElement {
        deferred(
            anchored()
                .anchor(corner)
                .snap_to_window_with_margin(px(8.))
                .child(div().on_mouse_down_out(cx.listener(close)).child(panel)),
        )
        .priority(1)
        .into_any_element()
    }

    fn picker_row(
        &self,
        cx: &mut Context<Self>,
        id: String,
        model: Model,
        name: String,
        vendor: String,
        icon_url: Option<String>,
        is_local: bool,
        hover: gpui::Hsla,
        for_message: Option<usize>,
    ) -> gpui::AnyElement {
        let theme = cx.theme().clone();

        let badge =
            div().flex_none().text_size(crate::tokens::font::SMALL).text_color(theme.text_muted).child(if is_local {
                "Local"
            } else {
                "Cloud"
            });

        div()
            .id(gpui::SharedString::from(format!("pick-{id}")))
            .flex()
            .items_center()
            .gap_3()
            .w_full()
            .px_4()
            .h(px(48.))
            .cursor(gpui::CursorStyle::PointingHand)
            .hover(move |s| s.bg(hover))
            .child(VendorIcon::new(vendor).size(crate::tokens::icon::XXL).icon_url(icon_url))
            .child(div().flex_1().min_w_0().text_sm().text_color(theme.text).child(name))
            .child(badge)
            .on_click(cx.listener(move |this, _, _, cx| {
                if let Some(msg_idx) = for_message {
                    this.regenerate_at_with_model(msg_idx, Some(model.clone()), cx);
                } else {
                    if this.state.streaming {
                        this.cancel_stream();
                    }
                    this.state.model = Some(model.clone());
                    this.clear_session();
                    this.close_popovers();
                    cx.notify();
                }
            }))
            .into_any_element()
    }

    pub(super) fn model_picker_panel(
        &self,
        cx: &mut Context<Self>,
        for_message: Option<usize>,
    ) -> gpui::AnyElement {
        let theme = cx.theme().clone();
        let hover = theme.bg_hover;
        let (local, cloud) = self.collect_model_entries(cx);

        let mut list = div().id("model-picker-list").flex().flex_col().gap_2().max_h(px(360.)).overflow_y_scroll();
        if local.is_empty() && cloud.is_empty() {
            list = list
                .child(div().px(px(14.)).py_2().text_sm().text_color(theme.text_muted).child("No models available"));
        } else {
            for (id, model, name, vendor, icon_url) in local {
                list = list.child(self.picker_row(cx, id, model, name, vendor, icon_url, true, hover, for_message));
            }
            for (id, model, name, vendor, icon_url) in cloud {
                list = list.child(self.picker_row(cx, id, model, name, vendor, icon_url, false, hover, for_message));
            }
        }

        div()
            .occlude()
            .w(px(520.))
            .flex()
            .flex_col()
            .rounded_xl()
            .bg(theme.card)
            .border_1()
            .border_color(theme.border)
            .child(div().py_2().child(list))
            .child(div().h_px().bg(theme.border))
            .child(
                div()
                    .id("model-picker-more")
                    .flex()
                    .items_center()
                    .justify_between()
                    .w_full()
                    .px_4()
                    .h(px(48.))
                    .cursor(gpui::CursorStyle::PointingHand)
                    .hover(move |s| s.bg(hover))
                    .on_click(cx.listener(move |this, _, _, cx| {
                        this.state.pending_regen = for_message;
                        this.close_popovers();
                        cx.emit(ChatEvent::OpenLocalModels);
                        cx.notify();
                    }))
                    .child(div().text_sm().text_color(theme.text).child("More local models"))
                    .child(IconEl::new(Icon::ChevronRight, theme.text_muted).size(crate::tokens::icon::MD)),
            )
            .into_any_element()
    }

    pub(super) fn performance_panel(
        &self,
        msg_idx: usize,
        cur: &Version,
        cx: &mut Context<Self>,
    ) -> Option<gpui::AnyElement> {
        if self.state.perf_open_msg != Some(msg_idx) {
            return None;
        }
        let theme = cx.theme().clone();
        let tps = cur
            .tps
            .filter(|value| value.is_finite())
            .map(|value| (value.round() as i64).to_string())
            .unwrap_or_else(|| "—".into());
        let format_seconds = |seconds: Option<f32>| {
            seconds.filter(|value| value.is_finite()).map(|value| format!("{value:.3}s")).unwrap_or_else(|| "—".into())
        };
        let ttft = format_seconds(cur.ttft);
        let total = format_seconds(cur.total_time);
        let stat = |value: String, label: &'static str| {
            div()
                .flex()
                .flex_col()
                .items_center()
                .gap_1()
                .child(
                    div()
                        .text_size(crate::tokens::font::HEADING)
                        .font_weight(FontWeight::MEDIUM)
                        .text_color(theme.text)
                        .child(value),
                )
                .child(div().text_size(crate::tokens::font::CAPTION).text_color(theme.text_muted).child(label))
        };
        let sep = || div().w_px().h(px(40.)).bg(theme.border).self_center();
        Some(
            div()
                .occlude()
                .py_4()
                .px_6()
                .rounded_xl()
                .bg(theme.card)
                .border_1()
                .border_color(theme.border)
                .child(
                    div()
                        .flex()
                        .items_center()
                        .gap_6()
                        .child(stat(ttft, "T to 1st token"))
                        .child(sep())
                        .child(stat(tps, "№ of t/s"))
                        .child(sep())
                        .child(stat(total, "Total time")),
                )
                .into_any_element(),
        )
    }

    pub(super) fn file_upload_panel(
        &self,
        cx: &mut Context<Self>,
    ) -> Option<gpui::AnyElement> {
        if !self.state.file_upload_open {
            return None;
        }
        let theme = cx.theme().clone();
        let hover = theme.bg_hover;
        Some(
            div()
                .occlude()
                .rounded_md()
                .bg(theme.card)
                .border_1()
                .border_color(theme.border)
                .p(px(6.))
                .child(
                    div()
                        .id("file-upload-option")
                        .flex()
                        .items_center()
                        .gap_2()
                        .w(px(300.))
                        .px(px(14.))
                        .py_2()
                        .rounded_md()
                        .cursor(gpui::CursorStyle::PointingHand)
                        .hover(move |s| s.bg(hover))
                        .on_click(cx.listener(|this, _, _, cx| {
                            this.state.file_upload_open = false;
                            this.pick_file(cx);
                        }))
                        .child(IconEl::new(Icon::Rename, theme.text_muted).size(crate::tokens::icon::MD))
                        .child(
                            div()
                                .flex()
                                .flex_col()
                                .child(div().text_sm().text_color(theme.text).child("Upload a file"))
                                .child(
                                    div()
                                        .text_size(crate::tokens::font::CAPTION)
                                        .text_color(theme.text_muted)
                                        .child("TXT · MD · JSON · CSV · YAML · PY · JS · TS · RS"),
                                ),
                        ),
                )
                .into_any_element(),
        )
    }
}
