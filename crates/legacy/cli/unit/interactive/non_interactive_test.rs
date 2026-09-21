use shoji::types::session::chat::ChatReplyStats;

use super::*;
use crate::interactive::components::Theme;

#[test]
fn completed_transcript_renders_without_a_terminal_or_ansi_sequences() {
    let messages = [
        ChatMessage::user().with_text("Do not echo this prompt".to_string()),
        ChatMessage::assistant().with_reasoning("Let me think".to_string()).with_text("Hello, 世界!".to_string()),
    ];
    let theme = Theme::default();
    let mut output = Vec::new();
    chat_transcript_component(
        build_transcript(&messages, 0),
        Some(ChatReplyStats {
            duration: 1.25,
            ..Default::default()
        }),
        theme.subtitle_color,
        theme.overlay_color(),
        theme.padding(),
        false,
    )
    .write(&mut output)
    .unwrap();
    let output = String::from_utf8(output).unwrap();
    assert!(output.contains("Let me think"));
    assert!(output.contains("Hello, 世界!"));
    assert!(output.contains("duration: 1.25 s"));
    assert!(!output.contains("Do not echo this prompt"));
    assert!(!output.contains('\u{1b}'));

    let mut output = std::io::BufWriter::new(Vec::new());
    let mut transcript = PlainTranscript::default();
    transcript.write(&mut output, vec![TranscriptItem::Thinking("Let me".to_string())], false).unwrap();
    assert_eq!(output.get_ref(), b"Let me");
    transcript
        .write(
            &mut output,
            vec![TranscriptItem::Thinking("Let me think".to_string()), TranscriptItem::Text("Hello, 世".to_string())],
            false,
        )
        .unwrap();
    assert_eq!(output.get_ref(), "Let me think\nHello, 世".as_bytes());
    transcript.write(&mut output, build_transcript(&messages, 0), true).unwrap();
    assert_eq!(output.get_ref(), "Let me think\nHello, 世界!".as_bytes());

    let mut output = std::io::BufWriter::new(Vec::new());
    let mut transcript = PlainTranscript::default();
    let emoji = "🙂".as_bytes();
    for end in 1..emoji.len() {
        let partial = format!("Hello, {}", String::from_utf8_lossy(&emoji[..end]));
        transcript.write(&mut output, vec![TranscriptItem::Text(partial)], false).unwrap();
        assert_eq!(output.get_ref(), b"Hello, ");
    }
    transcript.write(&mut output, vec![TranscriptItem::Text("Hello, 🙂".to_string())], false).unwrap();
    assert_eq!(output.get_ref(), "Hello, 🙂".as_bytes());

    let error = transcript.write(&mut output, vec![TranscriptItem::Text("Changed prefix".to_string())], false);
    assert_eq!(error.unwrap_err().kind(), std::io::ErrorKind::InvalidData);
    assert_eq!(output.get_ref(), "Hello, 🙂".as_bytes());
    assert_eq!(transcript.write(&mut output, Vec::new(), true).unwrap_err().kind(), std::io::ErrorKind::InvalidData);
    assert_eq!(output.get_ref(), "Hello, 🙂".as_bytes());

    transcript.write(&mut output, vec![TranscriptItem::Text("Hello, 🙂�".to_string())], false).unwrap();
    assert_eq!(output.get_ref(), "Hello, 🙂".as_bytes());
    transcript.write(&mut output, vec![TranscriptItem::Text("Hello, 🙂�".to_string())], true).unwrap();
    assert_eq!(output.get_ref(), "Hello, 🙂�".as_bytes());
}

#[test]
fn byte_by_byte_transcripts_only_append() {
    let cases = [
        ("ascii", "Plain text with punctuation."),
        ("newlines", "first\nsecond\n"),
        ("tabs", "one\ttwo\tthree"),
        ("surrounding_spaces", "  padded text  "),
        ("unicode_punctuation", "“quotes”—an ellipsis…"),
        ("two_byte_boundaries", "\u{80}\u{7ff}"),
        ("three_byte_boundaries", "\u{800}\u{d7ff}\u{e000}\u{ffff}"),
        ("four_byte_boundaries", "\u{10000}\u{10ffff}"),
        ("precomposed_accents", "café déjà vu"),
        ("combining_accent", "Cafe\u{301}"),
        ("stacked_combining_marks", "a\u{301}\u{323}\u{304}"),
        ("leading_combining_mark", "\u{301}accent"),
        ("hangul_syllables", "안녕하세요"),
        ("hangul_jamo", "\u{1100}\u{1161}\u{11a8}"),
        ("chinese", "你好，世界"),
        ("japanese", "こんにちは世界"),
        ("devanagari", "नमस्ते दुनिया"),
        ("arabic", "مرحبا بالعالم"),
        ("hebrew", "שלום עולם"),
        ("thai", "สวัสดีโลก"),
        ("georgian", "გამარჯობა"),
        ("cyrillic", "Привет, мир"),
        ("greek", "Καλημέρα κόσμε"),
        ("math_symbols", "∑ π √∞ ≠ ≤ ≥"),
        ("supplementary_math", "𝔘𝕫𝕦 𝟘𝟙𝟚"),
        ("musical_symbols", "𝄞𝄢"),
        ("single_emoji", "🙂"),
        ("adjacent_emoji", "🙂🙃😉"),
        ("emoji_in_text", "A🙂B🚀C"),
        ("skin_tone", "👍🏽"),
        ("different_skin_tones", "👋🏻👋🏿"),
        ("text_variation_selector", "\u{2764}\u{fe0e}"),
        ("emoji_variation_selector", "❤️"),
        ("digit_keycap", "1\u{fe0f}\u{20e3}"),
        ("symbol_keycaps", "#\u{20e3}*\u{fe0f}\u{20e3}"),
        ("country_flag", "🇬🇪"),
        ("adjacent_flags", "🇺🇸🇯🇵🇺🇳"),
        ("single_regional_indicator", "🇬"),
        ("profession_zwj", "👩🏽‍💻"),
        ("family_zwj", "👨‍👩‍👧‍👦"),
        ("rainbow_flag_zwj", "🏳️‍🌈"),
        ("pirate_flag_zwj", "🏴‍☠️"),
        ("couple_zwj", "👩‍❤️‍💋‍👨"),
        ("gender_and_skin_tone", "🏃🏽‍♀️"),
        ("head_shaking_zwj", "🙂‍↔️"),
        ("tag_sequence_flag", "\u{1f3f4}\u{e0067}\u{e0062}\u{e0065}\u{e006e}\u{e0067}\u{e007f}"),
        ("literal_replacement_inside", "before � after"),
        ("literal_replacement_at_end", "literal �"),
        ("replacements_around_emoji", "��🙂��"),
        ("mixed_multiline", "你好 👩🏽‍💻\nCafe\u{301} 🇬🇪\nDone ✅"),
        ("space_before_nonbreaking_space", "a \u{a0}z"),
        ("newline_before_em_space", "a\n\u{2003}z"),
        ("adjacent_unicode_spaces", "a\u{2003}\u{3000}z"),
        ("trailing_unicode_space", "a \u{a0}"),
        ("replacement_between_spaces", "a � \u{2003}z"),
        ("leading_unicode_spaces", " \u{a0}\u{2003}🙂"),
        ("only_unicode_spaces", "\u{a0} \u{2003}"),
    ];

    for (name, text) in cases {
        for reasoning in [false, true] {
            let message = |text: String| {
                if reasoning {
                    ChatMessage::assistant().with_reasoning(text)
                } else {
                    ChatMessage::assistant().with_reasoning("Thinking".to_string()).with_text(text)
                }
            };
            let expected = if reasoning {
                text.trim().to_string()
            } else {
                format!("Thinking\n{text}")
            };
            let mut output = std::io::BufWriter::new(Vec::new());
            let mut transcript = PlainTranscript::default();
            let mut previous = Vec::new();

            for end in 0..=text.len() {
                let at = format!("{name}, reasoning={reasoning}, byte={end}");
                let bytes = &text.as_bytes()[..end];
                let partial = String::from_utf8_lossy(bytes).into_owned();
                let items = build_transcript(&[message(partial)], 0);
                transcript.write(&mut output, items.clone(), false).unwrap_or_else(|error| panic!("{at}: {error}"));

                let rendered = output.get_ref();
                assert!(rendered.starts_with(&previous), "output rewound: {at}");
                assert!(expected.as_bytes().starts_with(rendered), "output diverged from final text: {at}");
                assert!(std::str::from_utf8(rendered).is_ok(), "partial UTF-8 was emitted: {at}");
                assert!(!rendered.contains(&0x1b), "escape code was emitted: {at}");

                if let Ok(valid) = std::str::from_utf8(bytes) {
                    let visible = if reasoning {
                        valid.trim()
                    } else {
                        valid
                    };
                    if !visible.ends_with('\u{fffd}') {
                        let expected_partial = if reasoning {
                            visible.to_string()
                        } else if visible.is_empty() {
                            "Thinking".to_string()
                        } else {
                            format!("Thinking\n{visible}")
                        };
                        assert_eq!(rendered, expected_partial.as_bytes(), "complete text was not flushed: {at}");
                    }
                }

                previous = rendered.clone();
                transcript.write(&mut output, items, false).unwrap_or_else(|error| panic!("{at}: {error}"));
                assert_eq!(output.get_ref(), &previous, "repeated update duplicated output: {at}");
            }

            let final_items = build_transcript(&[message(text.to_string())], 0);
            transcript.write(&mut output, final_items, true).unwrap();
            assert!(output.get_ref().starts_with(&previous), "completion rewound: {name}, reasoning={reasoning}");
            assert_eq!(output.get_ref(), expected.as_bytes(), "incomplete output: {name}, reasoning={reasoning}");
        }
    }
}

#[test]
fn reasoning_and_text_transitions_only_append() {
    let cases: &[(&str, &[(bool, &str)], &str)] = &[
        ("reasoning_then_text", &[(true, "Think"), (false, "Answer")], "Think\nAnswer"),
        ("text_then_reasoning", &[(false, "Before"), (true, "Think"), (false, "After")], "Before\nThink\nAfter"),
        ("alternating", &[(true, "One"), (false, "Answer"), (true, "Two"), (false, "Tail")], "One\nAnswer\nTwo\nTail"),
        ("identical_content", &[(false, "same"), (true, "same"), (false, " more")], "same\nsame\n more"),
        (
            "resumed_reasoning_emoji",
            &[(true, "Think"), (false, "Answer"), (true, "🙂"), (false, "Done")],
            "Think\nAnswer\n🙂\nDone",
        ),
        ("resumed_text_emoji", &[(false, "Hello"), (true, "Think"), (false, "🙂")], "Hello\nThink\n🙂"),
        (
            "resumed_reasoning_whitespace",
            &[(true, "One "), (false, "Answer"), (true, "\u{a0}Two"), (false, "Tail")],
            "One\nAnswer\n \u{a0}Two\nTail",
        ),
        ("initial_empty_reasoning", &[(true, " \n"), (false, "Answer"), (true, "Think")], "Answer\nThink"),
        ("reasoning_ends_with_replacement", &[(true, "Think�"), (false, "Answer")], "Think�\nAnswer"),
        ("text_ends_with_replacement", &[(false, "Answer�"), (true, "Think")], "Answer�\nThink"),
    ];

    for &(name, parts, expected) in cases {
        let mut reasoning = Vec::new();
        let mut text = Vec::new();
        let mut output = std::io::BufWriter::new(Vec::new());
        let mut transcript = PlainTranscript::default();
        let mut previous = Vec::new();
        let message = |reasoning: &[u8], text: &[u8]| {
            ChatMessage::assistant()
                .with_reasoning(String::from_utf8_lossy(reasoning).into_owned())
                .with_text(String::from_utf8_lossy(text).into_owned())
        };
        for &(is_thinking, part) in parts {
            for &byte in part.as_bytes() {
                if is_thinking {
                    reasoning.push(byte);
                } else {
                    text.push(byte);
                }
                let items = build_transcript(&[message(&reasoning, &text)], 0);
                transcript.write(&mut output, items.clone(), false).unwrap_or_else(|error| panic!("{name}: {error}"));
                let rendered = output.get_ref();
                assert!(rendered.starts_with(&previous), "output rewound: {name}");
                assert!(expected.as_bytes().starts_with(rendered), "output diverged: {name}: {rendered:?}");
                assert!(std::str::from_utf8(rendered).is_ok(), "partial UTF-8 was emitted: {name}");
                assert!(!rendered.contains(&0x1b), "escape code was emitted: {name}");
                previous = rendered.clone();
                transcript.write(&mut output, items, false).unwrap();
                assert_eq!(output.get_ref(), &previous, "repeated update duplicated output: {name}");
            }
        }
        transcript.write(&mut output, build_transcript(&[message(&reasoning, &text)], 0), true).unwrap();
        assert_eq!(output.get_ref(), expected.as_bytes(), "{name}");
    }
}

#[test]
#[ignore = "exhaustive Unicode and malformed UTF-8 sweep"]
fn exhaustive_byte_streams_only_append() {
    let mut bytes = "a � ".as_bytes().to_vec();
    let prefix_len = bytes.len();
    let mut encoded = [0; 4];
    for codepoint in 0..=0x10ffff {
        if let Some(ch) = char::from_u32(codepoint) {
            bytes.truncate(prefix_len);
            bytes.extend_from_slice(ch.encode_utf8(&mut encoded).as_bytes());
            assert_decoded_bytes_only_append(&bytes);
        }
    }
    eprintln!("All 1,112,064 Unicode scalar values passed");

    for pair in 0..=u16::MAX {
        bytes.truncate(prefix_len);
        bytes.extend_from_slice(&pair.to_be_bytes());
        assert_decoded_bytes_only_append(&bytes);
    }
    eprintln!("All 65,536 byte pairs passed");

    let boundaries = [
        b' ', b'a', 0x7f, 0x80, 0x8f, 0x90, 0x9f, 0xa0, 0xbf, 0xc0, 0xc1, 0xc2, 0xdf, 0xe0, 0xe1, 0xec, 0xed, 0xee,
        0xef, 0xf0, 0xf1, 0xf3, 0xf4, 0xf5, 0xff,
    ];
    for a in boundaries {
        for b in boundaries {
            for c in boundaries {
                for d in boundaries {
                    bytes.truncate(prefix_len);
                    bytes.extend_from_slice(&[a, b, c, d]);
                    assert_decoded_bytes_only_append(&bytes);
                }
            }
        }
    }
    eprintln!("All 390,625 four-byte combinations of UTF-8 boundary bytes passed");
}

fn assert_decoded_bytes_only_append(bytes: &[u8]) {
    let decoded = String::from_utf8_lossy(bytes);
    for reasoning in [false, true] {
        let message = |text: String| {
            if reasoning {
                ChatMessage::assistant().with_reasoning(text)
            } else {
                ChatMessage::assistant().with_text(text)
            }
        };
        let expected = if reasoning {
            decoded.trim()
        } else {
            &decoded
        };
        let mut output = std::io::BufWriter::new(Vec::new());
        let mut transcript = PlainTranscript::default();
        for end in 0..=bytes.len() {
            let partial = String::from_utf8_lossy(&bytes[..end]).into_owned();
            let items = build_transcript(&[message(partial)], 0);
            let previous_len = output.get_ref().len();
            transcript
                .write(&mut output, items, false)
                .unwrap_or_else(|error| panic!("{bytes:02x?}, reasoning={reasoning}, byte={end}: {error}"));
            let rendered = std::str::from_utf8(output.get_ref()).unwrap();
            assert!(rendered.len() >= previous_len, "output rewound: {bytes:02x?}, byte={end}");
            assert!(
                expected.starts_with(rendered),
                "output diverged: {bytes:02x?}, reasoning={reasoning}, byte={end}, rendered={rendered:?}, expected={expected:?}"
            );
        }
        let items = build_transcript(&[message(decoded.to_string())], 0);
        transcript.write(&mut output, items, true).unwrap();
        assert_eq!(output.get_ref(), expected.as_bytes(), "{bytes:02x?}, reasoning={reasoning}");
    }
}
