use gpui::{
    App, ClickEvent, CursorStyle, ElementId, FontWeight, IntoElement, RenderOnce, SharedString, Window, div,
    prelude::*, px,
};

use crate::{theme::ActiveTheme, tokens};

type SegHandler = Box<dyn Fn(&ClickEvent, &mut Window, &mut App) + 'static>;

struct Segment {
    label: SharedString,
    on_click: Option<SegHandler>,
}

#[derive(IntoElement)]
pub struct SegmentedControl {
    id: ElementId,
    segments: Vec<Segment>,
    selected: usize,
}

impl SegmentedControl {
    pub fn new(
        id: impl Into<ElementId>,
        selected: usize,
    ) -> Self {
        Self {
            id: id.into(),
            segments: Vec::new(),
            selected,
        }
    }

    pub fn segment(
        mut self,
        label: impl Into<SharedString>,
        on_click: impl Fn(&ClickEvent, &mut Window, &mut App) + 'static,
    ) -> Self {
        self.segments.push(Segment {
            label: label.into(),
            on_click: Some(Box::new(on_click)),
        });
        self
    }
}

impl RenderOnce for SegmentedControl {
    fn render(
        self,
        _window: &mut Window,
        cx: &mut App,
    ) -> impl IntoElement {
        let theme = cx.theme().clone();
        let selected = self.selected;

        let mut row = div()
            .id(self.id)
            .flex()
            .items_center()
            .gap_1()
            .p(px(2.))
            .rounded_lg()
            .bg(theme.bg_sub)
            .border_1()
            .border_color(theme.border);

        for (i, seg) in self.segments.into_iter().enumerate() {
            let is_sel = i == selected;
            let mut cell = div()
                .id(SharedString::from(format!("seg-{i}")))
                .flex_1()
                .flex()
                .items_center()
                .justify_center()
                .px_3()
                .py_1()
                .rounded_md()
                .text_size(tokens::font::COMPACT)
                .cursor(CursorStyle::PointingHand)
                .child(seg.label);

            if is_sel {
                let label_color = if theme.dark {
                    theme.bg
                } else {
                    theme.text
                };
                cell = cell.bg(gpui::white()).text_color(label_color).font_weight(FontWeight::MEDIUM);
            } else {
                let hover = theme.bg_hover;
                cell = cell.text_color(theme.text_muted).hover(move |s| s.bg(hover));
            }
            if let Some(handler) = seg.on_click {
                cell = cell.on_click(move |e, w, cx| handler(e, w, cx));
            }
            row = row.child(cell);
        }
        row
    }
}
