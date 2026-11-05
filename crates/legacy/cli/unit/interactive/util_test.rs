use super::*;

/// AltGr characters, which arrive as Ctrl+Alt on Windows, are text; Ctrl alone, with or without Shift, is a shortcut.
#[test]
fn altgr_types_text_and_ctrl_makes_shortcuts() {
    for modifiers in
        [KeyModifiers::NONE, KeyModifiers::SHIFT, KeyModifiers::ALT, KeyModifiers::CONTROL | KeyModifiers::ALT]
    {
        assert!(types_text(modifiers), "{modifiers:?}");
    }
    for modifiers in [KeyModifiers::CONTROL, KeyModifiers::CONTROL | KeyModifiers::SHIFT] {
        assert!(!types_text(modifiers), "{modifiers:?}");
    }
}
