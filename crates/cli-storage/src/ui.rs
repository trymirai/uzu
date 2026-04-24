use ratatui::{
    layout::{Constraint, Direction, Layout, Rect},
    prelude::Frame,
    style::{Color, Modifier, Style},
    widgets::{Block, Borders, Gauge, List, ListItem, Paragraph},
};

use crate::{
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
    models: &std::collections::HashMap<String, ModelWithState>,
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
    let items: Vec<ListItem> = section_models
        .iter()
        .map(|(id, model_with_state)| match &model_with_state.state.phase {
            uzu::storage::types::DownloadPhase::Downloaded {} => ListItem::new(format!("✓ {}", id)),
            uzu::storage::types::DownloadPhase::NotDownloaded {} => {
                let size_mb = model_with_state.state.total_bytes as f64 / 1_000_000.0;
                ListItem::new(format!("{} ({:.1} MB)", id, size_mb))
            },
            _ => ListItem::new(id.clone()),
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

    // Update list state
    let state = app.list_states.get_mut(&section).unwrap();

    if section_models.is_empty() {
        state.select(None);
    } else if state.selected().is_none() {
        state.select(Some(0));
    } else if let Some(selected) = state.selected() {
        if selected >= section_models.len() {
            state.select(Some(section_models.len().saturating_sub(1)));
        }
    }

    frame.render_stateful_widget(list, area, state);
}

fn render_downloading_section(
    frame: &mut Frame,
    area: Rect,
    app: &mut App,
    section_models: &[(String, ModelWithState)],
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
    let selected = state.selected().unwrap_or(0);

    // Scroll to keep selection visible
    let scroll_offset = if selected >= visible_count {
        selected - visible_count + 1
    } else {
        0
    };

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
        let is_selected = model_idx == selected;

        render_downloading_model(frame, *chunk, id, model_with_state, is_selected);
    }
}

fn render_downloading_model(
    frame: &mut Frame,
    area: Rect,
    id: &str,
    model_with_state: &ModelWithState,
    is_selected: bool,
) {
    let progress = model_with_state.state.progress() as f64;
    let downloaded_mb = model_with_state.state.downloaded_bytes as f64 / 1_000_000.0;
    let total_mb = model_with_state.state.total_bytes as f64 / 1_000_000.0;

    // Calculate available width (subtract borders: 2 chars)
    let available_width = area.width.saturating_sub(2) as usize;

    let (base_label, gauge_color, border_color) = match &model_with_state.state.phase {
        uzu::storage::types::DownloadPhase::Downloading {} => {
            let progress_info = format!(" ({:.1}/{:.1} MB - {:.1}%)", downloaded_mb, total_mb, progress * 100.0);
            let name = truncate_name_for_label(id, &progress_info, available_width, is_selected);
            (
                format!("{}{}", name, progress_info),
                Color::Cyan,
                if is_selected {
                    Color::Blue
                } else {
                    Color::Gray
                },
            )
        },
        uzu::storage::types::DownloadPhase::Paused {} => {
            let progress_info =
                format!(" [PAUSED] ({:.1}/{:.1} MB - {:.1}%)", downloaded_mb, total_mb, progress * 100.0);
            let name = truncate_name_for_label(id, &progress_info, available_width, is_selected);
            (
                format!("{}{}", name, progress_info),
                Color::Yellow,
                if is_selected {
                    Color::Yellow
                } else {
                    Color::Gray
                },
            )
        },
        uzu::storage::types::DownloadPhase::Error {
            message: err,
        } => {
            let error_info = format!(" [ERROR: {}]", err);
            let name = truncate_name_for_label(id, &error_info, available_width, is_selected);
            (
                format!("{}{}", name, error_info),
                Color::Red,
                if is_selected {
                    Color::Red
                } else {
                    Color::Gray
                },
            )
        },
        _ => (
            truncate_text(id, available_width),
            Color::Gray,
            if is_selected {
                Color::White
            } else {
                Color::Gray
            },
        ),
    };

    let label = if is_selected {
        format!("▶ {}", base_label)
    } else {
        base_label
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
        .ratio(progress.min(1.0).max(0.0));

    frame.render_widget(gauge, area);
}

fn render_help(
    frame: &mut Frame,
    area: Rect,
    app: &App,
    models: &std::collections::HashMap<String, ModelWithState>,
) {
    let helpers = app.get_helpers(models);
    let help_text = helpers.join("  |  ");

    let help = Paragraph::new(help_text)
        .block(Block::default().borders(Borders::ALL).title("Help"))
        .style(Style::default().fg(Color::Gray));

    frame.render_widget(help, area);
}

/// Truncate name to fit with progress info
fn truncate_name_for_label(
    name: &str,
    progress_info: &str,
    available_width: usize,
    is_selected: bool,
) -> String {
    // Account for selection indicator "▶ " if selected
    let indicator_len = if is_selected {
        2
    } else {
        0
    };
    let progress_len = progress_info.len();

    // Calculate max name length
    let max_name_len = available_width.saturating_sub(indicator_len).saturating_sub(progress_len);

    truncate_text(name, max_name_len)
}

/// Truncate text to fit within max_len, adding "..." if needed
fn truncate_text(
    text: &str,
    max_len: usize,
) -> String {
    if text.len() <= max_len {
        return text.to_string();
    }

    if max_len <= 3 {
        return text.chars().take(max_len).collect();
    }

    // Reserve 3 chars for "..."
    let truncate_at = max_len.saturating_sub(3);
    format!("{}...", text.chars().take(truncate_at).collect::<String>())
}
