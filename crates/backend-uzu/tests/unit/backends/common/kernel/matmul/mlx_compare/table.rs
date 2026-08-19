use std::io::{IsTerminal, Write, stdout};

use comfy_table::{
    Cell as TableCell, CellAlignment, Color, ContentArrangement, Table, modifiers::UTF8_ROUND_CORNERS,
    presets::UTF8_FULL_CONDENSED,
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

    fn micros(self) -> Option<f64> {
        match self {
            Slot::Micros(micros) => Some(micros),
            _ => None,
        }
    }
}

fn extremes(slots: &[Slot]) -> Option<(f64, f64)> {
    let mut measured = slots.iter().filter_map(|slot| slot.micros());
    let first = measured.next()?;
    let (low, high) = measured.fold((first, first), |(low, high), micros| (low.min(micros), high.max(micros)));
    (high > low).then_some((low, high))
}

fn paint(
    slot: Slot,
    extremes: Option<(f64, f64)>,
) -> TableCell {
    let cell = TableCell::new(slot.render());
    match (slot.micros(), extremes) {
        (Some(micros), Some((low, _))) if micros == low => cell.fg(Color::Green),
        (Some(micros), Some((_, high))) if micros == high => cell.fg(Color::Red),
        _ => cell,
    }
}

pub struct Block {
    title: String,
    columns: Vec<String>,
    rows: Vec<(Cell, Vec<Slot>)>,
    drawn_lines: usize,
    live: bool,
}

impl Block {
    pub fn new(
        title: String,
        columns: Vec<String>,
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
        let mut header = vec!["M".to_owned()];
        header.extend(self.columns.iter().cloned());

        table
            .load_preset(UTF8_FULL_CONDENSED)
            .apply_modifier(UTF8_ROUND_CORNERS)
            .set_content_arrangement(ContentArrangement::Dynamic)
            .set_header(header);

        for (cell, slots) in &self.rows {
            let extremes = extremes(slots);
            let mut row = vec![TableCell::new(cell.m)];
            row.extend(slots.iter().map(|slot| paint(*slot, extremes)));
            table.add_row(row);
        }

        for index in 0..=self.columns.len() {
            if let Some(column) = table.column_mut(index) {
                column.set_cell_alignment(CellAlignment::Right);
            }
        }

        format!("{}  (microseconds per matmul, fastest sample)\n{table}", self.title)
    }
}
