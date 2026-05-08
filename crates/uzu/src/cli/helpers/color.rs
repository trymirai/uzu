use iocraft::prelude::*;

pub trait ColorRgb {
    fn to_rgb(self) -> (u8, u8, u8);

    fn darker(
        self,
        factor: f32,
    ) -> Color
    where
        Self: Sized,
    {
        let (red, green, blue) = self.to_rgb();
        let factor = factor.clamp(0.0, 1.0);
        Color::Rgb {
            r: (red as f32 * factor).round() as u8,
            g: (green as f32 * factor).round() as u8,
            b: (blue as f32 * factor).round() as u8,
        }
    }
}

impl ColorRgb for Color {
    fn to_rgb(self) -> (u8, u8, u8) {
        match self {
            Color::Rgb {
                r,
                g,
                b,
            } => (r, g, b),
            Color::Black => (0, 0, 0),
            Color::DarkGrey => (85, 85, 85),
            Color::DarkRed => (170, 0, 0),
            Color::DarkGreen => (0, 170, 0),
            Color::DarkYellow => (170, 85, 0),
            Color::DarkBlue => (0, 0, 170),
            Color::DarkMagenta => (170, 0, 170),
            Color::DarkCyan => (0, 170, 170),
            Color::Grey => (170, 170, 170),
            Color::Red => (255, 85, 85),
            Color::Green => (85, 255, 85),
            Color::Yellow => (255, 255, 85),
            Color::Blue => (85, 85, 255),
            Color::Magenta => (255, 85, 255),
            Color::Cyan => (85, 255, 255),
            Color::White => (255, 255, 255),
            _ => (0, 0, 0),
        }
    }
}
