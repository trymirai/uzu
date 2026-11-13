use std::collections::HashMap;

use ratatui::{
    layout::{Constraint, Direction, Layout, Rect},
    prelude::Frame,
    style::{Color, Modifier, Style},
    widgets::{Block, Borders, Gauge, List, ListItem, ListState, Paragraph},
};
use shoji::types::model::ModelIdentifier;
use unicode_width::UnicodeWidthStr;
use uzu::storage::DownloadPhase;

use super::{
    app::{App, ModelWithState},
    models::ModelOrganizer,
    sections::Section,
};

pub fn draw(
    frame: &mut Frame,
    app: &mut App,
) {
    // Try to get models snapshot
    let models_snapshot = match app.models.try_lock() {
        Ok(guard) => guard.clone(),
        Err(_) => return, // Skip frame if locked
    };
    app.active_section = app.active_section.first_with_models(Section::next, |section| {
        ModelOrganizer::filter_for_section(&models_snapshot, section).len()
    });

    // Main layout
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Min(0), Constraint::Length(3)])
        .split(frame.area());

    // Three columns for sections
    let main_chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(33), Constraint::Percentage(33), Constraint::Percentage(34)])
        .split(chunks[0]);

    // Render each section
    render_section(frame, main_chunks[0], app, &models_snapshot, Section::Installed);
    render_section(frame, main_chunks[1], app, &models_snapshot, Section::Downloading);
    render_section(frame, main_chunks[2], app, &models_snapshot, Section::Available);

    // Render contextual help
    render_help(frame, chunks[1], app, &models_snapshot);
}

fn render_section(
    frame: &mut Frame,
    area: Rect,
    app: &mut App,
    models: &HashMap<ModelIdentifier, ModelWithState>,
    section: Section,
) {
    let is_active = app.active_section == section;
    let section_models = ModelOrganizer::filter_for_section(models, section);

    // Special rendering for Downloading section with progress
    if section == Section::Downloading && !section_models.is_empty() {
        render_downloading_section(frame, area, app, &section_models, is_active);
        return;
    }

    // Regular list rendering
    let name_width = area.width.saturating_sub(2) as usize;
    let items: Vec<ListItem> = section_models
        .iter()
        .map(|(id, model_with_state)| {
            let (prefix, suffix) = match &model_with_state.state.phase {
                DownloadPhase::Initializing {} => ("", " (Checking…)".to_string()),
                DownloadPhase::Downloaded {} => ("✓ ", String::new()),
                DownloadPhase::NotDownloaded {} => {
                    let size_mb = model_with_state.state.total_bytes as f64 / 1_000_000.0;
                    ("", format!(" ({:.1} MB)", size_mb))
                },
                _ => ("", String::new()),
            };
            let id = truncate_middle(id, name_width.saturating_sub(prefix.width() + suffix.width()));
            ListItem::new(format!("{prefix}{id}{suffix}"))
        })
        .collect();

    let (border_style, highlight_style) = if is_active {
        (
            Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD),
            Style::default().bg(Color::Blue).fg(Color::White).add_modifier(Modifier::BOLD),
        )
    } else {
        (Style::default().fg(Color::Gray), Style::default().bg(Color::DarkGray))
    };

    let list = List::new(items)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title(format!("{} ({})", section.title(), section_models.len()))
                .border_style(border_style),
        )
        .highlight_style(highlight_style);

    let state = app.list_states.get_mut(&section).unwrap();
    update_list_window(state, section_models.len(), area.height.saturating_sub(2) as usize);
    frame.render_stateful_widget(list, area, state);
}

fn render_downloading_section(
    frame: &mut Frame,
    area: Rect,
    app: &mut App,
    section_models: &[(ModelIdentifier, ModelWithState)],
    is_active: bool,
) {
    let border_style = if is_active {
        Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)
    } else {
        Style::default().fg(Color::Gray)
    };

    let block = Block::default()
        .borders(Borders::ALL)
        .title(format!("Downloading ({})", section_models.len()))
        .border_style(border_style);

    let inner = block.inner(area);
    frame.render_widget(block, area);

    // Calculate layout for each model
    let model_height = 3;
    let available_height = inner.height as usize;
    let visible_count = (available_height / model_height).min(section_models.len());

    let state = app.list_states.get_mut(&Section::Downloading).unwrap();
    update_list_window(state, section_models.len(), visible_count);
    let selected = state.selected();
    let scroll_offset = state.offset();

    let constraints: Vec<Constraint> = (0..visible_count).map(|_| Constraint::Length(model_height as u16)).collect();

    if constraints.is_empty() {
        return;
    }

    let model_chunks = Layout::default().direction(Direction::Vertical).constraints(constraints).split(inner);

    // Render visible models
    for (i, chunk) in model_chunks.iter().enumerate() {
        let model_idx = i + scroll_offset;
        if model_idx >= section_models.len() {
            break;
        }

        let (id, model_with_state) = &section_models[model_idx];
        let is_selected = selected == Some(model_idx);

        render_downloading_model(frame, *chunk, id, model_with_state, is_selected);
    }
}

fn render_downloading_model(
    frame: &mut Frame,
    area: Rect,
    id: &ModelIdentifier,
    model_with_state: &ModelWithState,
    is_selected: bool,
) {
    let state = &model_with_state.state;
    let progress = state.progress() as f64;
    let bytes_info = format!(
        "({:.1}/{:.1} MB - {:.1}%)",
        state.downloaded_bytes as f64 / 1_000_000.0,
        state.total_bytes as f64 / 1_000_000.0,
        progress * 100.0
    );
    let (info, gauge_color, selected_border) = match &state.phase {
        DownloadPhase::Initializing {} => (" [CHECKING]".to_string(), Color::Gray, Color::White),
        DownloadPhase::Downloading {} => (format!(" {bytes_info}"), Color::Cyan, Color::Blue),
        DownloadPhase::Paused {} => (format!(" [PAUSED] {bytes_info}"), Color::Yellow, Color::Yellow),
        DownloadPhase::Locked {
            manager_id,
        } => (format!(" [LOCKED BY {manager_id}] {bytes_info}"), Color::Yellow, Color::Yellow),
        DownloadPhase::Error {
            message,
        } => (format!(" [ERROR: {message}]"), Color::Red, Color::Red),
        DownloadPhase::NotDownloaded {} | DownloadPhase::Downloaded {} => (String::new(), Color::Gray, Color::White),
    };
    let indicator = if is_selected {
        "▶ "
    } else {
        ""
    };
    let available_width = area.width.saturating_sub(2) as usize;
    let name = truncate_middle(id, available_width.saturating_sub(indicator.width() + info.width()));
    let label = format!("{indicator}{name}{info}");
    let border_color = if is_selected {
        selected_border
    } else {
        Color::Gray
    };

    let gauge_style = if is_selected {
        Style::default().fg(Color::White).bg(Color::Blue).add_modifier(Modifier::BOLD)
    } else {
        Style::default().fg(gauge_color).bg(Color::Black)
    };

    let gauge = Gauge::default()
        .block(Block::default().borders(Borders::ALL).border_style(Style::default().fg(border_color).add_modifier(
            if is_selected {
                Modifier::BOLD
            } else {
                Modifier::empty()
            },
        )))
        .gauge_style(gauge_style)
        .label(label)
        .ratio(progress.clamp(0.0, 1.0));

    frame.render_widget(gauge, area);
}

fn render_help(
    frame: &mut Frame,
    area: Rect,
    app: &App,
    models: &HashMap<ModelIdentifier, ModelWithState>,
) {
    let helpers = app.get_helpers(models);
    let help_text = helpers.join("  |  ");

    let help = Paragraph::new(help_text)
        .block(Block::default().borders(Borders::ALL).title("Help"))
        .style(Style::default().fg(Color::Gray));

    frame.render_widget(help, area);
}

/// Stores the selection and scroll offset `list_window` gives the state's list.
fn update_list_window(
    state: &mut ListState,
    count: usize,
    visible: usize,
) {
    let (selected, offset) = list_window(state.selected(), state.offset(), count, visible);
    state.select(selected);
    *state.offset_mut() = offset;
}

/// The selection and scroll offset of `count` rows with `visible` on screen: the first row is selected when none is,
/// the selection stays in range and on screen, and the list never scrolls past its last row.
fn list_window(
    selected: Option<usize>,
    offset: usize,
    count: usize,
    visible: usize,
) -> (Option<usize>, usize) {
    if count == 0 {
        return (None, 0);
    }
    let visible = visible.max(1);
    let selected = selected.unwrap_or(0).min(count - 1);
    let offset = offset.min(selected).max((selected + 1).saturating_sub(visible)).min(count.saturating_sub(visible));
    (Some(selected), offset)
}

/// `text` cut in the middle with an ellipsis to fit `max_width` columns: model IDs differ at their ends.
fn truncate_middle(
    text: &str,
    max_width: usize,
) -> String {
    let characters: Vec<char> = text.chars().collect();
    if characters.len() <= max_width {
        return text.to_string();
    }
    let kept = max_width.saturating_sub(1);
    let head = kept.div_ceil(2);
    let tail = kept - head;
    let head: String = characters[..head].iter().collect();
    let tail: String = characters[characters.len() - tail..].iter().collect();
    format!("{head}…{tail}")
}

#[cfg(test)]
#[path = "../../unit/storage/ui_test.rs"]
mod tests;
