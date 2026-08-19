use std::io::{IsTerminal, Write, stdout};

use comfy_table::{
    CellAlignment, ContentArrangement, Table, modifiers::UTF8_ROUND_CORNERS, presets::UTF8_FULL_CONDENSED,
};

use super::Cell;

#[derive(Clone, Copy)]
pub enum Slot {
    Pending,
    Unsupported,
    Micros(f64),
}

impl Slot {
    fn render(self) -> String {
        match self {
            Slot::Pending => "·".to_owned(),
            Slot::Unsupported => "—".to_owned(),
            Slot::Micros(micros) => format!("{micros:.1}"),
        }
    }
}

pub struct Block {
    title: String,
    columns: Vec<&'static str>,
    rows: Vec<(Cell, Vec<Slot>)>,
    drawn_lines: usize,
    live: bool,
}

impl Block {
    pub fn new(
        title: String,
        columns: Vec<&'static str>,
        cells: &[Cell],
    ) -> Self {
        let rows = cells.iter().map(|cell| (*cell, vec![Slot::Pending; columns.len()])).collect();
        let mut block = Self {
            title,
            columns,
            rows,
            drawn_lines: 0,
            live: stdout().is_terminal(),
        };
        if block.live {
            block.draw();
        }
        block
    }

    pub fn set(
        &mut self,
        row: usize,
        column: usize,
        slot: Slot,
    ) {
        self.rows[row].1[column] = slot;
        if self.live {
            self.draw();
        }
    }

    pub fn finish(&mut self) {
        if !self.live {
            self.draw();
        }
        self.drawn_lines = 0;
    }

    fn draw(&mut self) {
        let rendered = self.render();
        let mut out = stdout().lock();
        if self.drawn_lines > 0 {
            let _ = write!(out, "\x1b[{}A\x1b[J", self.drawn_lines);
        }
        let _ = writeln!(out, "{rendered}");
        let _ = out.flush();
        self.drawn_lines = rendered.lines().count();
    }

    fn render(&self) -> String {
        let mut table = Table::new();
        let mut header = vec!["layer".to_owned(), "M".to_owned(), "K".to_owned(), "N".to_owned()];
        header.extend(self.columns.iter().map(|column| (*column).to_owned()));

        table
            .load_preset(UTF8_FULL_CONDENSED)
            .apply_modifier(UTF8_ROUND_CORNERS)
            .set_content_arrangement(ContentArrangement::Dynamic)
            .set_header(header);

        for (cell, slots) in &self.rows {
            let mut row = vec![cell.layer.to_owned(), cell.m.to_string(), cell.k.to_string(), cell.n.to_string()];
            row.extend(slots.iter().map(|slot| slot.render()));
            table.add_row(row);
        }

        for index in 1..4 + self.columns.len() {
            if let Some(column) = table.column_mut(index) {
                column.set_cell_alignment(CellAlignment::Right);
            }
        }

        format!("{}  (microseconds per matmul, fastest sample)\n{table}", self.title)
    }
}
