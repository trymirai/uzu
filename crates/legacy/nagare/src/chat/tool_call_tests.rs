use super::*;

#[test]
fn json_candidates_use_only_the_top_level_tool_name() {
    assert_eq!(json_candidate_name(r#"{"arguments":{"name":"wrong"},"name":"right"}"#).as_deref(), Some("right"));
    assert_eq!(json_candidate_name(r#"{"name":"wr\u0069te","arguments":{"name":"wrong""#).as_deref(), Some("write"));
    assert_eq!(json_candidate_name(r#"{"arguments":{"name":"wrong""#), None);
}

#[test]
fn candidates_and_completed_calls_keep_the_same_identifier() {
    use shoji::{
        traits::backend::chat_message::{Output, ToolCallState},
        types::session::chat::ChatContentBlock,
    };
    let mut identifiers = Vec::new();
    let candidate = super::super::build_message(
        &Output {
            tool_calls: vec![ToolCallState::Candidate("<function=write>".into())],
            ..Default::default()
        },
        &mut identifiers,
    );
    let ChatContentBlock::ToolCallCandidate {
        identifier: Some(identifier),
        ..
    } = &candidate.content[0]
    else {
        panic!("candidate must have an identifier");
    };
    let finished = super::super::build_message(
        &Output {
            tool_calls: vec![ToolCallState::Finished(ToolCall {
                identifier: None,
                name: "write".into(),
                arguments: Value::from(serde_json::json!({"path": "a"})),
            })],
            ..Default::default()
        },
        &mut identifiers,
    );
    assert_eq!(finished.tool_calls()[0].identifier.as_ref(), Some(identifier));
}

#[test]
fn xml_stream_preserves_escapes_and_closing_tag_prefixes() {
    let text = "héllo 🙂\n\"quoted\"\\\n</parametric>\n";
    let raw = format!("<function=write>\n<parameter=content>\n{text}\n</parameter>\n</function>");
    let stream = ToolCallArgumentStream::create(Vec::new());
    let mut json = String::new();
    for (index, _) in raw.char_indices() {
        json.push_str(&stream.update(raw[..index].into()).unwrap().arguments);
    }
    json.push_str(&stream.update(raw).unwrap().arguments);
    assert!(json.contains("héllo"), "arguments must appear before completion");
    json.push_str(
        &stream
            .finish(ToolCall {
                identifier: None,
                name: "write".into(),
                arguments: Value::from(serde_json::json!({"content": text})),
            })
            .unwrap()
            .arguments,
    );
    assert_eq!(serde_json::from_str::<serde_json::Value>(&json).unwrap(), serde_json::json!({"content": text}));
}

#[test]
fn atem_stream_preserves_parameter_text_and_closing_tag_prefixes() {
    for text in ["\nhéllo 🙂\n\"quoted\"\\\n</atem:parametric>\n", "{\n  \"a\": 1, \"b\": [true, null]\n}"] {
        let raw = format!(
            "<atem:function_calls>\n<atem:invoke name=\"write\">\n\
             <atem:parameter name=\"content\">{text}</atem:parameter>\n\
             </atem:invoke>\n</atem:function_calls>"
        );
        let stream = ToolCallArgumentStream::create(Vec::new());
        let mut json = String::new();
        for (index, _) in raw.char_indices() {
            json.push_str(&stream.update(raw[..index].into()).unwrap().arguments);
            if index == raw.find("</atem:parameter>").unwrap() {
                let quoted = serde_json::to_string(text).unwrap();
                assert_eq!(json, format!("{{\"content\":{}", &quoted[..quoted.len() - 1]));
            }
        }
        json.push_str(&stream.update(raw).unwrap().arguments);
        json.push_str(
            &stream
                .finish(ToolCall {
                    identifier: None,
                    name: "write".into(),
                    arguments: Value::from(serde_json::json!({"content": text})),
                })
                .unwrap()
                .arguments,
        );
        assert_eq!(serde_json::from_str::<serde_json::Value>(&json).unwrap(), serde_json::json!({"content": text}));
    }
}

#[test]
fn declared_types_stop_recursive_union_expansion() {
    for keyword in ["anyOf", "oneOf"] {
        let mut branches = vec![serde_json::json!({"type": "object"})];
        branches.extend(std::iter::repeat_n(serde_json::json!({"$ref": "#"}), 32));
        let schema = serde_json::json!({keyword: branches});

        assert_eq!(declared_types(&schema, &schema, 8), ["object"]);
    }
}

#[test]
fn declared_types_reuse_shared_union_nodes() {
    let mut definitions = serde_json::Map::new();
    definitions.insert("0".to_string(), serde_json::json!({"type": ["object", "object", "null"]}));
    for level in 1..=6 {
        let branches = vec![serde_json::json!({"$ref": format!("#/$defs/{}", level - 1)}); 32];
        definitions.insert(level.to_string(), serde_json::json!({"anyOf": branches}));
    }
    let schema = serde_json::json!({"$ref": "#/$defs/6", "$defs": definitions});

    assert_eq!(declared_types(&schema, &schema, 8), ["null", "object"]);
}

#[test]
fn declared_types_revisit_shared_nodes_with_more_depth() {
    let mut nested = serde_json::json!({"$ref": "#/$defs/Shared"});
    for _ in 0..6 {
        nested = serde_json::json!({"anyOf": [nested]});
    }
    let schema = serde_json::json!({
        "$defs": {"Shared": {"anyOf": [{"type": "string"}]}},
        "anyOf": [{"type": "object"}, {"$ref": "#/$defs/Shared"}, nested]
    });

    assert_eq!(declared_types(&schema, &schema, 8), ["object", "string"]);
}

#[test]
fn declared_types_discard_partial_results_when_work_is_exhausted() {
    let leaves = vec![serde_json::json!({"type": "object"}); 32];
    let mut branches = vec![serde_json::json!({"type": "string"})];
    branches.extend(std::iter::repeat_n(serde_json::json!({"anyOf": leaves}), 32));
    let mut kinds = vec!["object"; MAX_SCHEMA_TRAVERSAL_WORK];
    kinds.push("string");

    for property in [serde_json::json!({"anyOf": branches}), serde_json::json!({"type": kinds})] {
        let tools = serde_json::json!([{"function": {
            "name": "read",
            "parameters": {"properties": {"options": property}}
        }}]);
        let parameters = Value::from(tools[0]["function"]["parameters"].clone());
        let types = ToolParameterTypes::from_parameters([("read", &parameters)]);
        assert!(types.declared("read", "options").is_none());

        let call = ToolCall {
            identifier: None,
            name: "read".to_string(),
            arguments: Value {
                json: serde_json::json!({"options": "{\"a\": 1}"}).to_string(),
            },
        };
        assert_eq!(coerce_tool_call(&call, &types).arguments.json, call.arguments.json);
        assert_eq!(serialize_param_value("{\"a\": 1}", types.declared("read", "options")), r#""{\"a\": 1}""#);
    }
}
