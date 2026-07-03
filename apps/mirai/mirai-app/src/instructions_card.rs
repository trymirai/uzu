use gpui::{
    Context, CursorStyle, ElementId, Entity, FontWeight, IntoElement, Render, SharedString, Window, div, prelude::*,
};

use crate::{
    components::{Icon, IconEl, InputEvent, TextInput},
    persistence,
    theme::ActiveTheme,
};

pub struct InstructionsCard {
    element_id: ElementId,
    input: Entity<TextInput>,
    open: bool,
}

impl InstructionsCard {
    pub fn new(
        cx: &mut Context<Self>,
        element_id: impl Into<ElementId>,
        placeholder: impl Into<SharedString>,
    ) -> Self {
        let input = cx.new(|cx| TextInput::new(cx, placeholder).multiline(false, 3, 6));
        let current = persistence::global_instructions();
        if !current.is_empty() {
            input.update(cx, |input, cx| input.set_text(current, cx));
        }
        cx.subscribe(&input, |_this, _input, event, _cx| match event {
            InputEvent::Submit(text) | InputEvent::Changed(text) => persistence::set_global_instructions(text),
        })
        .detach();
        Self {
            element_id: element_id.into(),
            input,
            open: false,
        }
    }
}

impl Render for InstructionsCard {
    fn render(
        &mut self,
        _window: &mut Window,
        cx: &mut Context<Self>,
    ) -> impl IntoElement {
        let theme = cx.theme().clone();
        let hover = theme.bg_hover;
        let open = self.open;

        let header = div()
            .id(self.element_id.clone())
            .flex()
            .items_center()
            .gap_3()
            .px_4()
            .py_3()
            .cursor(CursorStyle::PointingHand)
            .hover(move |s| s.bg(hover))
            .on_click(cx.listener(|this, _, _, cx| {
                this.open = !this.open;
                if this.open {
                    let current = persistence::global_instructions();
                    this.input.update(cx, |input, cx| input.set_text(&current, cx));
                }
                cx.notify();
            }))
            .child(IconEl::new(Icon::Plus, theme.text).size(crate::tokens::icon::MD).rotate(if open {
                45.
            } else {
                0.
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
                            .child("Add instructions to all chats"),
                    )
                    .child(div().text_xs().text_color(theme.text_muted).child("Tailor the way the model responds")),
            );

        let mut card = div().w_full().rounded_lg().border_1().border_color(theme.border).bg(theme.card).child(header);

        if open {
            card = card.child(
                div().px_4().pb_3().child(
                    div()
                        .w_full()
                        .px_3()
                        .py_2()
                        .rounded_md()
                        .border_1()
                        .border_color(theme.border)
                        .bg(theme.bg)
                        .child(self.input.clone()),
                ),
            );
        }
        card
    }
}
