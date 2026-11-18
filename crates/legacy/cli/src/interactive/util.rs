use iocraft::KeyModifiers;

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

/// Whether a key pressed with `modifiers` types its character: Ctrl makes it a shortcut, but the Windows console reports
/// AltGr as Ctrl+Alt.
pub fn types_text(modifiers: KeyModifiers) -> bool {
    !modifiers.contains(KeyModifiers::CONTROL) || (cfg!(windows) && modifiers.contains(KeyModifiers::ALT))
}
