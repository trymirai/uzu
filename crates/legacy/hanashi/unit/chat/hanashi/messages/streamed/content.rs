use super::*;

#[test]
fn preserves_reasoning_whitespace_but_normalizes_text() {
    let blocks = Content::Sections(vec![
        Section::Reasoning {
            value: Some(" think\n\n".to_string()),
        },
        Section::Text {
            value: Some(" answer \n".to_string()),
        },
    ])
    .blocks(&ChatRole::Assistant {});

    assert_eq!(
        blocks,
        vec![
            ChatContentBlock::Reasoning {
                value: " think\n\n".to_string(),
            },
            ChatContentBlock::Text {
                value: "answer".to_string(),
            },
        ]
    );
}

#[test]
fn removes_qwen_template_separators_from_visible_text() {
    let blocks = Content::Sections(vec![
        Section::Text {
            value: Some("\n".to_string()),
        },
        Section::Reasoning {
            value: Some("\nLet me think.\n".to_string()),
        },
        Section::Text {
            value: Some("\n\nHello!".to_string()),
        },
    ])
    .blocks(&ChatRole::Assistant {});

    assert_eq!(
        blocks,
        vec![
            ChatContentBlock::Reasoning {
                value: "\nLet me think.\n".to_string(),
            },
            ChatContentBlock::Text {
                value: "Hello!".to_string(),
            },
        ]
    );
}

#[test]
fn drops_whitespace_only_text_sections() {
    let blocks = Content::Sections(vec![
        Section::Reasoning {
            value: Some(" \n".to_string()),
        },
        Section::Text {
            value: Some("\t".to_string()),
        },
    ])
    .blocks(&ChatRole::Assistant {});

    assert!(blocks.is_empty());
}
