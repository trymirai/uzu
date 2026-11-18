use super::*;

fn window(
    selected: Option<usize>,
    offset: usize,
    count: usize,
    visible: usize,
) -> (Option<usize>, usize) {
    let mut state = ListState::default().with_selected(selected).with_offset(offset);
    update_list_window(&mut state, count, visible);
    (state.selected(), state.offset())
}

/// A list selects its first row until one is chosen, so actions apply to the row drawn as selected, and keeps the
/// selection in range and on screen.
#[test]
fn list_window_selects_and_clamps() {
    assert_eq!(window(None, 0, 0, 5), (None, 0));
    assert_eq!(window(None, 0, 3, 5), (Some(0), 0));
    assert_eq!(window(Some(7), 0, 3, 5), (Some(2), 0));
    assert_eq!(window(Some(9), 0, 25, 5), (Some(9), 5));
    assert_eq!(window(Some(9), 8, 25, 5), (Some(9), 8));
    assert_eq!(window(Some(2), 8, 25, 5), (Some(2), 2));
}

/// A window that grows past the list's end scrolls back, so no row hides above empty space.
#[test]
fn list_window_never_scrolls_past_the_end() {
    assert_eq!(window(Some(24), 20, 25, 5), (Some(24), 20));
    assert_eq!(window(Some(24), 20, 25, 48), (Some(24), 0));
    assert_eq!(window(Some(24), 20, 25, 10), (Some(24), 15));
}
