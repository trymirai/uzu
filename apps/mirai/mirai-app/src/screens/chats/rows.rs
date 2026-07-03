use gpui::{AnyElement, Context, CursorStyle, FontWeight, IntoElement, SharedString, div, prelude::*, px};

use super::{event::ChatsEvent, util::relative_time, view::ChatsView};
use crate::{
    components::{Icon, IconEl},
    persistence::StoredChat,
    theme::{ActiveTheme, FONT_MONO},
    tokens,
};

impl ChatsView {
    pub(super) fn row(
        &self,
        cx: &mut Context<Self>,
        chat: &StoredChat,
        selection_mode: bool,
        selected: bool,
    ) -> AnyElement {
        let theme = cx.theme().clone();
        let hover_bg = theme.bg_hover;
        let click_id = chat.id.clone();
        let subtitle = relative_time(chat.updated_at);

        let mut checkbox = div()
            .size(px(18.))
            .flex_none()
            .rounded(tokens::radius::SM)
            .border_1()
            .border_color(theme.border)
            .flex()
            .items_center()
            .justify_center();
        if selected {
            checkbox = checkbox.bg(theme.info).child(IconEl::new(Icon::Check, theme.card).size(tokens::icon::XS));
        }

        let mut left = div().flex().flex_1().min_w_0().items_center().gap_3();
        if selection_mode {
            left = left.child(checkbox);
        } else {
            left = left.child(IconEl::new(Icon::Chats, theme.text_muted).size(tokens::icon::LG));
        }
        left = left.child(
            div()
                .min_w_0()
                .overflow_hidden()
                .text_sm()
                .text_color(theme.text)
                .font_weight(FontWeight::MEDIUM)
                .child(chat.title.clone()),
        );

        let timestamp = div().flex_none().font_family(FONT_MONO).text_xs().text_color(theme.text_muted).child(subtitle);

        let border = if selected {
            theme.info
        } else {
            theme.border
        };
        div()
            .pb_2()
            .child(
                div()
                    .id(SharedString::from(chat.id.clone()))
                    .flex()
                    .w_full()
                    .items_center()
                    .justify_between()
                    .gap_3()
                    .min_h(px(46.))
                    .px_3()
                    .py_1p5()
                    .rounded_lg()
                    .border_1()
                    .border_color(border)
                    .bg(theme.card)
                    .cursor(CursorStyle::PointingHand)
                    .hover(move |s| s.bg(hover_bg))
                    .on_click(cx.listener(move |this, _, _, cx| {
                        if this.selection_mode {
                            if !this.selected.remove(&click_id) {
                                this.selected.insert(click_id.clone());
                            }
                            cx.notify();
                        } else {
                            cx.emit(ChatsEvent::Open(click_id.clone()));
                        }
                    }))
                    .child(left)
                    .child(timestamp),
            )
            .into_any_element()
    }
}
