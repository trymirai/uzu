use iocraft::prelude::*;
use unicode_segmentation::UnicodeSegmentation;
use unicode_width::UnicodeWidthStr;

use crate::cli::helpers::{SYMBOL_CURSOR, SYMBOL_NEW_LINE};

#[derive(Debug)]
pub struct RenderedText {
    pub original_text: String,
    pub position: usize,
}

impl RenderedText {
    pub fn new() -> Self {
        Self {
            original_text: String::new(),
            position: 0,
        }
    }

    pub fn segments(
        &self,
        maximal_width: usize,
    ) -> Vec<Vec<MixedTextContent>> {
        let cursor_at_end = self.position == graphemes_length(&self.original_text);
        let mut text = self.original_text.clone();
        if cursor_at_end {
            text.push_str(SYMBOL_CURSOR);
        }
        let cursor_position = self.position;

        let mut lines: Vec<Vec<MixedTextContent>> = vec![];
        let mut current_line: Vec<MixedTextContent> = vec![];
        let mut current_segment = String::new();
        let mut current_width = 0;

        let flush_segment = |line: &mut Vec<MixedTextContent>, segment: &mut String| {
            if !segment.is_empty() {
                line.push(MixedTextContent::new(segment.clone()));
                segment.clear();
            }
        };

        for (grapheme_index, (_, grapheme)) in text.grapheme_indices(true).enumerate() {
            let is_cursor = grapheme_index == cursor_position;
            let width = grapheme.width();

            if grapheme == SYMBOL_NEW_LINE {
                flush_segment(&mut current_line, &mut current_segment);
                if is_cursor {
                    current_line.push(MixedTextContent::new(SYMBOL_CURSOR.to_string()));
                }
                lines.push(current_line.clone());
                current_line.clear();
                current_width = 0;
                continue;
            }

            if current_width + width > maximal_width {
                flush_segment(&mut current_line, &mut current_segment);
                lines.push(current_line.clone());
                current_line.clear();
                current_width = 0;
            }

            if is_cursor {
                flush_segment(&mut current_line, &mut current_segment);
                let mut content = MixedTextContent::new(grapheme.to_string());
                if !cursor_at_end {
                    content.invert = true;
                }
                current_line.push(content);
            } else {
                current_segment.push_str(grapheme);
            }
            current_width += width;
        }

        flush_segment(&mut current_line, &mut current_segment);
        if !current_line.is_empty() {
            lines.push(current_line);
        }

        lines
    }
}

impl RenderedText {
    pub fn reset(&mut self) {
        self.original_text = String::new();
        self.position = 0;
    }

    pub fn add_character(
        &mut self,
        character: char,
    ) {
        let position_byte_index = self
            .original_text
            .grapheme_indices(true)
            .nth(self.position)
            .map(|(byte_index, _)| byte_index)
            .unwrap_or(self.original_text.len());

        let mut next_text = self.original_text.clone();
        next_text.insert(position_byte_index, character);

        let previous_length = graphemes_length(&self.original_text);
        let next_length = graphemes_length(&next_text);

        self.original_text = next_text;
        self.position += next_length.saturating_sub(previous_length);
    }

    pub fn remove_character(&mut self) {
        if self.position == 0 {
            return;
        }

        let position_byte_index = self
            .original_text
            .grapheme_indices(true)
            .nth(self.position)
            .map(|(byte_index, _)| byte_index)
            .unwrap_or(self.original_text.len());

        let previous_grapheme_byte_index = self.original_text[..position_byte_index]
            .grapheme_indices(true)
            .last()
            .map(|(byte_index, _)| byte_index)
            .unwrap_or(0);

        let mut next_text = self.original_text.clone();
        next_text.replace_range(previous_grapheme_byte_index..position_byte_index, "");

        let previous_length = graphemes_length(&self.original_text);
        let next_length = graphemes_length(&next_text);

        self.original_text = next_text;
        self.position -= previous_length.saturating_sub(next_length);
    }

    pub fn move_position_left(&mut self) {
        if self.position > 0 {
            self.position -= 1;
        }
    }

    pub fn move_position_right(&mut self) {
        if self.position < graphemes_length(&self.original_text) {
            self.position += 1;
        }
    }

    pub fn move_position_to_start(&mut self) {
        self.position = 0;
    }

    pub fn move_position_to_end(&mut self) {
        self.position = graphemes_length(&self.original_text);
    }

    pub fn move_position_up(
        &mut self,
        maximal_width: usize,
    ) {
        let (row, column) = grapheme_visual_position(&self.original_text, self.position, maximal_width);
        if row == 0 {
            return;
        }
        self.position = grapheme_at_visual_position(&self.original_text, row - 1, column, maximal_width);
    }

    pub fn move_position_down(
        &mut self,
        maximal_width: usize,
    ) {
        let (row, column) = grapheme_visual_position(&self.original_text, self.position, maximal_width);
        self.position = grapheme_at_visual_position(&self.original_text, row + 1, column, maximal_width);
    }
}

fn graphemes_length(text: &str) -> usize {
    text.grapheme_indices(true).count()
}

pub fn grapheme_at_visual_position(
    text: &String,
    target_row: usize,
    target_col: usize,
    maximal_width: usize,
) -> usize {
    let mut row = 0usize;
    let mut column = 0usize;
    let mut result = 0;

    for (grapheme_index, grapheme) in text.grapheme_indices(true) {
        let width = grapheme.width();
        if grapheme != SYMBOL_NEW_LINE && column + width > maximal_width {
            row += 1;
            column = 0;
        }

        if row > target_row || (row == target_row && column > target_col) {
            return result;
        }
        result = grapheme_index;

        if grapheme == "\n" {
            row += 1;
            column = 0;
        } else {
            column += width;
        }
    }

    if column + 1 > maximal_width {
        row += 1;
        column = 0;
    }
    if row > target_row || (row == target_row && column > target_col) {
        return result;
    }
    graphemes_length(text)
}

pub fn grapheme_visual_position(
    text: &String,
    position: usize,
    maximal_width: usize,
) -> (usize, usize) {
    let mut row = 0usize;
    let mut column = 0usize;

    for (grapheme_index, grapheme) in text.grapheme_indices(true) {
        let width = grapheme.width();

        if grapheme != "\n" && column + width > maximal_width {
            row += 1;
            column = 0;
        }

        if grapheme_index == position {
            return (row, column);
        }

        if grapheme == "\n" {
            row += 1;
            column = 0;
        } else {
            column += width;
        }
    }

    if column + 1 > maximal_width {
        return (row + 1, 0);
    }
    (row, column)
}
