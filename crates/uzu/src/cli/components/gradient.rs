use iocraft::{prelude::*, taffy::style::Style};

use crate::cli::helpers::ColorRgb;

#[with_layout_style_props]
#[non_exhaustive]
#[derive(Default, Props)]
pub struct GradientProps<'a> {
    pub children: Vec<AnyElement<'a>>,
    pub from_color: Option<Color>,
    pub to_color: Option<Color>,
    pub fill_factor: Option<f32>,
}

#[derive(Default)]
pub struct Gradient {
    from_color: (u8, u8, u8),
    to_color: (u8, u8, u8),
    fill_factor: Option<f32>,
}

impl Component for Gradient {
    type Props<'a> = GradientProps<'a>;

    fn new(_props: &Self::Props<'_>) -> Self {
        Self::default()
    }

    fn update(
        &mut self,
        props: &mut Self::Props<'_>,
        _hooks: Hooks,
        updater: &mut ComponentUpdater,
    ) {
        self.from_color = props.from_color.map(ColorRgb::to_rgb).unwrap_or((0, 0, 0));
        self.to_color = props.to_color.map(ColorRgb::to_rgb).unwrap_or((0, 0, 0));
        self.fill_factor = props.fill_factor;

        let style: Style = props.layout_style().into();
        updater.set_layout_style(style);
        updater.update_children(props.children.iter_mut(), None);
    }

    fn draw(
        &mut self,
        drawer: &mut ComponentDrawer<'_>,
    ) {
        let layout = drawer.layout();
        let width = layout.size.width as usize;
        let height = layout.size.height as usize;
        if width == 0 || height == 0 {
            return;
        }

        let fill_columns =
            self.fill_factor.map(|factor| (factor.clamp(0.0, 1.0) * width as f32).round() as usize).unwrap_or(width);

        let mut canvas = drawer.canvas();
        for column in 0..fill_columns {
            let factor = column as f32 / (width.saturating_sub(1).max(1) as f32);
            let red = (self.from_color.0 as f32 * (1.0 - factor) + self.to_color.0 as f32 * factor).round() as u8;
            let green = (self.from_color.1 as f32 * (1.0 - factor) + self.to_color.1 as f32 * factor).round() as u8;
            let blue: u8 = (self.from_color.2 as f32 * (1.0 - factor) + self.to_color.2 as f32 * factor).round() as u8;
            canvas.set_background_color(
                column as isize,
                0,
                1,
                height,
                Color::Rgb {
                    r: red,
                    g: green,
                    b: blue,
                },
            );
        }
    }
}
