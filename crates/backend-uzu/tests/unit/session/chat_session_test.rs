use crate::session::{
    chat_session::{find_stop_match, finish_reason_with_stop},
    helpers::OutputParser,
    types::FinishReason,
};

const REASONING_REGEX: &str = r"(?s)(?:<think>)?(?P<chain_of_thought>.*?)(?:</think>\s*(?P<response>.*))?\Z";

fn stops(values: &[&str]) -> Vec<String> {
    values.iter().map(|value| value.to_string()).collect()
}

#[test]
fn test_chat_session_stop_matches_substring_in_middle() {
    assert_eq!(find_stop_match("hello END world", &stops(&["END"])), Some(6));
}

#[test]
fn test_chat_session_stop_returns_earliest_of_several() {
    assert_eq!(find_stop_match("abc END xy STOP", &stops(&["STOP", "END"])), Some(4));
}

#[test]
fn test_chat_session_stop_no_match_returns_none() {
    assert_eq!(find_stop_match("nothing to see here", &stops(&["END"])), None);
}

#[test]
fn test_chat_session_stop_empty_list_returns_none() {
    assert_eq!(find_stop_match("text END", &[]), None);
}

#[test]
fn test_chat_session_stop_empty_strings_are_ignored() {
    assert_eq!(find_stop_match("text END", &stops(&["", "END"])), Some(5));
}

#[test]
fn test_chat_session_stop_offset_is_a_char_boundary_for_multibyte_text() {
    let text = "café END";
    let offset = find_stop_match(text, &stops(&["END"])).unwrap();
    assert_eq!(offset, 6);
    assert_eq!(&text[..offset], "café ");
}

#[test]
fn test_chat_session_stop_does_not_match_inside_reasoning() {
    let parser = OutputParser::new(Some(REASONING_REGEX.to_string())).unwrap();
    let response = parser.parse("<think>plan STOP here</think>visible answer".to_string()).parsed.response.unwrap();
    assert_eq!(response, "visible answer");
    assert_eq!(find_stop_match(&response, &stops(&["STOP"])), None);
}

#[test]
fn test_chat_session_stop_matches_within_response_after_reasoning() {
    let parser = OutputParser::new(Some(REASONING_REGEX.to_string())).unwrap();
    let response = parser.parse("<think>thinking</think>keep this STOP drop this".to_string()).parsed.response.unwrap();
    assert_eq!(find_stop_match(&response, &stops(&["STOP"])), Some(10));
}

#[test]
fn test_chat_session_stop_no_response_while_reasoning_is_open() {
    let parser = OutputParser::new(Some(REASONING_REGEX.to_string())).unwrap();
    let parsed = parser.parse("<think>still thinking, STOP appears".to_string());
    assert!(parsed.parsed.response.is_none());
}

#[test]
fn test_chat_session_stop_overrides_length_finish_reason() {
    assert!(matches!(finish_reason_with_stop(Some(FinishReason::Length), true), Some(FinishReason::Stop)));
}

#[test]
fn test_chat_session_stop_overrides_context_limit_finish_reason() {
    assert!(matches!(finish_reason_with_stop(Some(FinishReason::ContextLimitReached), true), Some(FinishReason::Stop)));
}

#[test]
fn test_chat_session_stop_sets_stop_when_no_other_reason() {
    assert!(matches!(finish_reason_with_stop(None, true), Some(FinishReason::Stop)));
}

#[test]
fn test_chat_session_no_stop_match_preserves_finish_reason() {
    assert!(matches!(finish_reason_with_stop(Some(FinishReason::Length), false), Some(FinishReason::Length)));
    assert!(finish_reason_with_stop(None, false).is_none());
}
