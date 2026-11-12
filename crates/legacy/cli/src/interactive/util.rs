use iocraft::prelude::*;

pub fn cycle<T: Copy + PartialEq>(
    values: &[T],
    current: T,
    delta: isize,
) -> T {
    let length = values.len() as isize;
    let index = values.iter().position(|value| *value == current).map(|index| index as isize).unwrap_or(0);
    let next = (index + delta).rem_euclid(length) as usize;
    values[next]
}

pub fn use_terminal_width(hooks: &mut Hooks) -> u16 {
    let (width, _) = hooks.use_terminal_size();
    crossterm::terminal::size().map_or(width, |(width, _)| width)
}

pub fn types_text(modifiers: KeyModifiers) -> bool {
    !modifiers.contains(KeyModifiers::CONTROL) || modifiers.contains(KeyModifiers::ALT)
}

#[cfg(test)]
#[path = "../../unit/interactive/util_test.rs"]
mod tests;
