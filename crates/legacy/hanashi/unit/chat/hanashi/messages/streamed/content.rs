use super::*;

#[test]
fn preserves_text_section_whitespace_for_rerendering() {
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
                value: " answer \n".to_string(),
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
