//! The multi-step "Clear data" wizard (select categories → confirm → result).

use gpui::{AnyElement, Context, CursorStyle, FontWeight, IntoElement, SharedString, div, prelude::*, px};

use super::view::SettingsView;
use crate::{
    components::{Button, ButtonKind, ButtonSize},
    data_ops::{self, CleanupCategory},
    theme::ActiveTheme,
};

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum ClearDataStep {
    Select,
    Confirm,
    Result,
}

fn confirm_summary(categories: &[CleanupCategory]) -> String {
    let labels: Vec<&str> = categories.iter().map(|c| c.label()).collect();
    match labels.len() {
        0 => String::new(),
        1 => labels[0].to_string(),
        2 => format!("{} and {}", labels[0], labels[1]),
        _ => format!("{}, and {}", labels[..labels.len() - 1].join(", "), labels[labels.len() - 1]),
    }
}

impl SettingsView {
    pub(super) fn clear_data_modal(
        &self,
        cx: &mut Context<Self>,
    ) -> Option<AnyElement> {
        if !self.clear_data_open {
            return None;
        }
        let theme = cx.theme().clone();
        let preview = self.clear_data_preview.clone();
        let selected = self.clear_data_selected.clone();
        let busy = self.clear_data_busy;
        let step = self.clear_data_step;

        let checkbox_row = |cat: CleanupCategory| {
            let checked = selected.contains(&cat);
            let desc = data_ops::category_description(cat, &preview);
            let box_color = if checked {
                theme.info
            } else {
                theme.border
            };
            div()
                .id(SharedString::from(format!("clear-cat-{:?}", cat)))
                .flex()
                .items_start()
                .gap_3()
                .pb_2()
                .cursor(CursorStyle::PointingHand)
                .on_click(cx.listener(move |this, _, _, cx| this.toggle_clear_category(cat, cx)))
                .child(
                    div()
                        .mt(px(2.))
                        .size(px(14.))
                        .rounded_sm()
                        .border_1()
                        .border_color(box_color)
                        .bg(if checked {
                            theme.info.opacity(0.15)
                        } else {
                            gpui::transparent_black()
                        })
                        .flex()
                        .items_center()
                        .justify_center()
                        .text_xs()
                        .text_color(theme.info)
                        .child(if checked {
                            "✓"
                        } else {
                            ""
                        }),
                )
                .child(
                    div()
                        .flex()
                        .flex_col()
                        .gap_0p5()
                        .child(
                            div().text_sm().font_weight(FontWeight::MEDIUM).text_color(theme.text).child(cat.label()),
                        )
                        .child(div().text_xs().text_color(theme.text_muted).child(desc)),
                )
        };

        let (title, body, footer): (SharedString, AnyElement, AnyElement) =
            match step {
                ClearDataStep::Select => {
                    let selected_count = CleanupCategory::ALL.iter().filter(|c| selected.contains(c)).count();
                    let review = Button::new("clear-review", "Review")
                        .kind(ButtonKind::Danger)
                        .size(ButtonSize::Small)
                        .disabled(selected_count == 0 || busy)
                        .on_click(cx.listener(|this, _, _, cx| {
                            this.clear_data_step = ClearDataStep::Confirm;
                            cx.notify();
                        }));
                    let cancel = Button::new("clear-cancel", "Cancel")
                        .kind(ButtonKind::Secondary)
                        .size(ButtonSize::Small)
                        .on_click(cx.listener(|this, _, _, cx| this.close_clear_data(cx)));
                    (
                        "Clear data".into(),
                        div()
                            .flex()
                            .flex_col()
                            .gap_3()
                            .child(
                                div()
                                    .text_sm()
                                    .text_color(theme.text_muted)
                                    .child("Select the data categories you want to permanently delete."),
                            )
                            .child(div().flex().flex_col().children(CleanupCategory::ALL.map(checkbox_row)))
                            .into_any_element(),
                        div().flex().justify_end().gap_2().child(cancel).child(review).into_any_element(),
                    )
                },
                ClearDataStep::Confirm => {
                    let cats: Vec<CleanupCategory> =
                        CleanupCategory::ALL.into_iter().filter(|c| selected.contains(c)).collect();
                    let summary = confirm_summary(&cats);
                    let delete = Button::new("clear-delete", "Delete")
                        .kind(ButtonKind::Danger)
                        .size(ButtonSize::Small)
                        .disabled(busy)
                        .on_click(cx.listener(|this, _, _, cx| this.run_clear_data(cx)));
                    let back = Button::new("clear-back", "Back")
                        .kind(ButtonKind::Secondary)
                        .size(ButtonSize::Small)
                        .disabled(busy)
                        .on_click(cx.listener(|this, _, _, cx| {
                            this.clear_data_step = ClearDataStep::Select;
                            cx.notify();
                        }));
                    let cancel = Button::new("clear-cancel", "Cancel")
                        .kind(ButtonKind::Secondary)
                        .size(ButtonSize::Small)
                        .disabled(busy)
                        .on_click(cx.listener(|this, _, _, cx| this.close_clear_data(cx)));
                    (
                        "Are you sure?".into(),
                        div()
                            .flex()
                            .flex_col()
                            .gap_3()
                            .child(div().text_sm().text_color(theme.text_muted).child(format!(
                                "This will permanently delete: {summary}. This action cannot be undone."
                            )))
                            .into_any_element(),
                        div().flex().justify_end().gap_2().child(cancel).child(back).child(delete).into_any_element(),
                    )
                },
                ClearDataStep::Result => {
                    let close = Button::new("clear-close", "Close")
                        .kind(ButtonKind::Primary)
                        .size(ButtonSize::Small)
                        .on_click(cx.listener(|this, _, _, cx| this.close_clear_data(cx)));
                    let mut lines = div().flex().flex_col().gap_2();
                    for cat in CleanupCategory::ALL.iter().filter(|c| selected.contains(c)) {
                        let done = self.clear_data_results.iter().find(|(c, _)| c == cat).is_some_and(|(_, ok)| *ok);
                        let mark = if done {
                            "✓"
                        } else {
                            "✗"
                        };
                        let status = if done {
                            "cleared"
                        } else {
                            "failed"
                        };
                        lines = lines.child(
                            div().flex().items_center().gap_2().child(div().text_sm().child(mark)).child(
                                div()
                                    .text_sm()
                                    .text_color(if done {
                                        theme.text
                                    } else {
                                        theme.text_muted
                                    })
                                    .child(format!("{} {status}", cat.label())),
                            ),
                        );
                    }
                    (
                        "Done".into(),
                        lines.into_any_element(),
                        div().flex().justify_end().gap_2().child(close).into_any_element(),
                    )
                },
            };

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
                        .child(div().text_color(theme.text).font_weight(FontWeight::MEDIUM).child(title))
                        .child(body)
                        .child(footer),
                )
                .into_any_element(),
        )
    }
}
