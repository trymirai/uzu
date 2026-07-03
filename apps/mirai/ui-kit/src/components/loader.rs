use std::time::Duration;

use gpui::{
    Animation, AnimationExt, App, IntoElement, RenderOnce, SharedString, Styled, Transformation, Window, div,
    percentage, prelude::*, px, svg,
};

use crate::{components::icon::Icon, theme::ActiveTheme};

#[derive(IntoElement)]
pub struct Loader {
    size: f32,
    label: Option<SharedString>,
}

impl Default for Loader {
    fn default() -> Self {
        Self {
            size: 16.0,
            label: None,
        }
    }
}

impl Loader {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn size(
        mut self,
        size: f32,
    ) -> Self {
        self.size = size;
        self
    }

    pub fn label(
        mut self,
        label: impl Into<SharedString>,
    ) -> Self {
        self.label = Some(label.into());
        self
    }
}

impl RenderOnce for Loader {
    fn render(
        self,
        _window: &mut Window,
        cx: &mut App,
    ) -> impl IntoElement {
        let color = cx.theme().text_muted;
        let size = self.size;

        let spinner = svg().path(Icon::Spinner.path()).size(px(size)).text_color(color).with_animation(
            "loader-spin",
            Animation::new(Duration::from_secs(1)).repeat(),
            |el, delta| el.with_transformation(Transformation::rotate(percentage(delta))),
        );

        let mut row = div().flex().items_center().gap_2().child(spinner);
        if let Some(label) = self.label {
            row = row.child(div().text_sm().text_color(color).child(label));
        }
        row
    }
}
