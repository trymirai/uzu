use std::collections::VecDeque;

use ratatui::{Frame, layout::Rect};

use crate::{
    color::{level_for, level_style},
    state::{accent, accent_rgb, background, background_rgb, foreground_rgb},
    widgets::panel_owned,
};

const BOX_GLYPH: char = '■';
const COLUMN_STRIDE: usize = 2;

pub(crate) fn render_chart(
    frame: &mut Frame,
    area: Rect,
    title: String,
    history: &VecDeque<f64>,
    ceiling: f64,
) {
    let block = panel_owned(title);
    let inner = block.inner(area);
    frame.render_widget(block, area);
    if inner.width == 0 || inner.height == 0 {
        return;
    }

    let box_columns = (inner.width as usize).div_ceil(COLUMN_STRIDE);
    let rows = inner.height as usize;
    let ceiling = ceiling.max(1e-9);
    let accent = accent();
    let accent_rgb = accent_rgb();
    let foreground_rgb = foreground_rgb();
    let background = background();
    let background_rgb = background_rgb();
    let off_style = level_style(0, accent_rgb, foreground_rgb, background_rgb, accent).bg(background);

    let shown: Vec<f64> = history.iter().rev().take(box_columns).rev().copied().collect();
    let offset = box_columns - shown.len();

    let buffer = frame.buffer_mut();
    for column in 0..box_columns {
        let fraction = match column.checked_sub(offset).and_then(|index| shown.get(index)) {
            Some(value) => (value / ceiling).clamp(0.0, 1.0),
            None => 0.0,
        };
        let filled_rows = (fraction * rows as f64).round() as usize;
        let data_style =
            level_style(level_for(fraction), accent_rgb, foreground_rgb, background_rgb, accent).bg(background);
        let x = inner.left() + (column * COLUMN_STRIDE) as u16;
        for row in 0..rows {
            let style = if row < filled_rows { data_style } else { off_style };
            let y = inner.bottom() - 1 - row as u16;
            if let Some(cell) = buffer.cell_mut((x, y)) {
                cell.set_char(BOX_GLYPH).set_style(style);
            }
        }
    }
}
