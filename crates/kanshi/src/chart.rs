use std::collections::VecDeque;

use ratatui::{Frame, layout::Rect};

use crate::{
    state::{accent, background},
    widgets::panel_owned,
};

const BRAILLE_DOTS: [[u16; 4]; 2] = [[0x01, 0x02, 0x04, 0x40], [0x08, 0x10, 0x20, 0x80]];

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

    let columns = inner.width as usize;
    let shown: Vec<f64> = history.iter().rev().take(columns).rev().copied().collect();
    if shown.is_empty() {
        return;
    }
    let offset = columns - shown.len();
    let ceiling = ceiling.max(1e-9);
    let cell_rows = inner.height as usize;
    let sub_rows = cell_rows * 4;
    let last = shown.len() - 1;
    let color = accent();
    let background = background();

    let fill_at = |sub_x: usize| -> Option<usize> {
        let position = sub_x as f64 / 2.0 - offset as f64;
        if position < 0.0 {
            return None;
        }
        let low = (position.floor() as usize).min(last);
        let high = (low + 1).min(last);
        let blend = position - low as f64;
        let value = shown[low] * (1.0 - blend) + shown[high] * blend;
        Some(((value / ceiling).clamp(0.0, 1.0) * sub_rows as f64).round() as usize)
    };

    let buffer = frame.buffer_mut();
    for cell_x in 0..columns {
        let fills = [fill_at(cell_x * 2), fill_at(cell_x * 2 + 1)];
        if fills.iter().all(Option::is_none) {
            continue;
        }
        let x = inner.left() + cell_x as u16;
        for row in 0..cell_rows {
            let mut dots = 0u16;
            for (sub, fill) in fills.iter().enumerate() {
                let Some(fill) = fill else {
                    continue;
                };
                for internal in 0..4 {
                    if row * 4 + (3 - internal) < *fill {
                        dots |= BRAILLE_DOTS[sub][internal];
                    }
                }
            }
            if dots == 0 {
                continue;
            }
            let symbol = char::from_u32(0x2800 + dots as u32).unwrap_or(' ');
            let y = inner.bottom() - 1 - row as u16;
            if let Some(cell) = buffer.cell_mut((x, y)) {
                cell.set_char(symbol).set_fg(color).set_bg(background);
            }
        }
    }
}
