use ratatui::{
    layout::{Constraint, Direction, Layout, Rect},
    style::Style,
    widgets::{Block, Borders},
};

use crate::state::accent;

pub(crate) fn panel(title: &str) -> Block<'_> {
    Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(accent()))
        .title_style(Style::default().fg(accent()))
        .title(title)
}

pub(crate) fn panel_owned(title: String) -> Block<'static> {
    Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(accent()))
        .title_style(Style::default().fg(accent()))
        .title(title)
}

pub(crate) fn split_horizontal<const N: usize>(
    area: Rect,
    constraints: [Constraint; N],
) -> std::rc::Rc<[Rect]> {
    Layout::default().direction(Direction::Horizontal).constraints(constraints).split(area)
}

pub(crate) fn split_vertical<const N: usize>(
    area: Rect,
    constraints: [Constraint; N],
) -> std::rc::Rc<[Rect]> {
    Layout::default().direction(Direction::Vertical).constraints(constraints).split(area)
}
