use super::*;

/// A list selects its first row until one is chosen, so actions apply to the row drawn as selected, and keeps the
/// selection in range and on screen.
#[test]
fn list_window_selects_and_clamps() {
    assert_eq!(list_window(None, 0, 0, 5), (None, 0));
    assert_eq!(list_window(None, 0, 3, 5), (Some(0), 0));
    assert_eq!(list_window(Some(7), 0, 3, 5), (Some(2), 0));
    assert_eq!(list_window(Some(9), 0, 25, 5), (Some(9), 5));
    assert_eq!(list_window(Some(9), 8, 25, 5), (Some(9), 8));
    assert_eq!(list_window(Some(2), 8, 25, 5), (Some(2), 2));
}

/// A window that grows past the list's end scrolls back, so no row hides above empty space.
#[test]
fn list_window_never_scrolls_past_the_end() {
    assert_eq!(list_window(Some(24), 20, 25, 5), (Some(24), 20));
    assert_eq!(list_window(Some(24), 20, 25, 48), (Some(24), 0));
    assert_eq!(list_window(Some(24), 20, 25, 10), (Some(24), 15));
}

/// Model IDs keep both ends, so the 4-bit and 8-bit variants of a model stay apart.
#[test]
fn ids_truncate_in_the_middle() {
    assert_eq!(truncate_middle("alibaba:qwen3.5:0.8b:mirai:mirai-m:4", 40), "alibaba:qwen3.5:0.8b:mirai:mirai-m:4");
    assert_eq!(truncate_middle("alibaba:qwen3.5:0.8b:mirai:mirai-m:4", 20), "alibaba:qw…mirai-m:4");
    assert_eq!(truncate_middle("alibaba:qwen3.5:0.8b:mirai:mirai-l:8", 20), "alibaba:qw…mirai-l:8");
    assert_eq!(truncate_middle("alibaba:qwen3.5:0.8b:mirai:mirai-m:4", 20).chars().count(), 20);
}
