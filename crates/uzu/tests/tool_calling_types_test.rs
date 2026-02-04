use uzu::tool_calling::*;

#[test]
fn test_tool_calling_value_from_primitives() {
    let value: Value = 42i64.into();
    assert_eq!(value.as_i64(), Some(42));

    let value: Value = 3.14f64.into();
    assert_eq!(value.as_f64(), Some(3.14));

    let value: Value = "Hello".into();
    assert_eq!(value.as_str(), Some("Hello"));

    let value: Value = true.into();
    assert_eq!(value.as_bool(), Some(true));
}

#[test]
fn test_tool_calling_value_json_roundtrip() {
    let json = serde_json::json!({"text": "Hello", "number": 42});
    let value: Value = json.clone().into();
    let json_restored: serde_json::Value = value.into();
    assert_eq!(json, json_restored);
}

#[test]
fn test_tool_calling_tool_call_result_success() {
    let result = ToolCallResult::success(
        "call_1".to_string(),
        "test_tool".to_string(),
        Value::String("Ok".to_string()),
    );
    assert_eq!(result.tool_call_id, "call_1");
    assert_eq!(result.name, "test_tool");
    assert!(result.is_success());
    assert_eq!(
        result.content,
        ToolCallResultContent::Success(Value::String("Ok".to_string()))
    );

    let json = serde_json::to_value(&result).unwrap();
    assert_eq!(json["content"], "Ok");
}

#[test]
fn test_tool_calling_tool_call_result_failure() {
    let result = ToolCallResult::failure(
        "call_2".to_string(),
        "test_tool".to_string(),
        "Something went wrong".to_string(),
    );
    assert_eq!(result.tool_call_id, "call_2");
    assert_eq!(result.name, "test_tool");
    assert!(!result.is_success());
    assert_eq!(
        result.content,
        ToolCallResultContent::Failure("Something went wrong".to_string())
    );

    let json = serde_json::to_value(&result).unwrap();
    assert_eq!(json["content"]["error"], "Something went wrong");
}

#[test]
fn test_tool_calling_tool_deserialization() {
    let json = serde_json::json!({
        "type": "function",
        "function": {
            "name": "test_func",
            "description": "A test function",
            "parameters": {
                "type": "object",
                "properties": {
                    "arg1": {"type": "string"}
                },
                "required": ["arg1"]
            }
        }
    });

    let tool: Tool = serde_json::from_value(json).unwrap();

    match tool {
        Tool::Function {
            function,
        } => {
            assert_eq!(function.name, "test_func");
            assert_eq!(function.description, "A test function");
        },
    }
}

#[test]
fn test_tool_calling_tool_call_with_value() {
    let arguments = Value::Object(
        [
            ("location".to_string(), Value::String("Tokyo, Japan".to_string())),
            ("unit".to_string(), Value::String("celsius".to_string())),
        ]
        .into_iter()
        .collect(),
    );

    let tool_call = ToolCall::new("get_temperature".to_string(), arguments);

    assert!(!tool_call.id.is_empty());
    assert_eq!(tool_call.name, "get_temperature");
    assert!(tool_call.arguments.as_object().is_some());
}

#[test]
fn test_tool_calling_tool_call_creation() {
    let tool_call = ToolCall::new(
        "my_function".to_string(),
        Value::from(serde_json::json!({"arg": "value"})),
    );

    assert!(!tool_call.id.is_empty());
    assert_eq!(tool_call.name, "my_function");
    assert_eq!(tool_call.arguments["arg"], "value");
}

#[test]
fn test_tool_calling_tool_from_parameters() {
    let tool = ToolFunction::from_parameters(
        "get_weather".to_string(),
        "Get weather information".to_string(),
        vec![
            ToolParameter::required(
                "location".to_string(),
                ToolParameterType::string(),
                "The city name".to_string(),
            ),
            ToolParameter::optional(
                "units".to_string(),
                ToolParameterType::string(),
                "Temperature units".to_string(),
            )
            .with_enum(vec!["celsius".into(), "fahrenheit".into()]),
        ],
    )
    .into_tool();

    let json = serde_json::to_value(&tool).unwrap();

    assert_eq!(json["type"], "function");
    assert_eq!(json["function"]["name"], "get_weather");
    assert_eq!(
        json["function"]["parameters"]["properties"]["location"]["type"],
        "string"
    );
    assert_eq!(
        json["function"]["parameters"]["properties"]["units"]["enum"],
        serde_json::json!(["celsius", "fahrenheit"])
    );
    assert_eq!(
        json["function"]["parameters"]["required"],
        serde_json::json!(["location"])
    );
}

#[test]
fn test_tool_calling_tool_parameter_types() {
    assert_eq!(
        ToolParameterType::string().to_schema(),
        serde_json::json!({"type": "string"})
    );

    assert_eq!(
        ToolParameterType::bool().to_schema(),
        serde_json::json!({"type": "boolean"})
    );

    assert_eq!(
        ToolParameterType::int().to_schema(),
        serde_json::json!({"type": "integer"})
    );

    assert_eq!(
        ToolParameterType::double().to_schema(),
        serde_json::json!({"type": "number"})
    );

    let array_type = ToolParameterType::array(ToolParameterType::string());
    let schema = array_type.to_schema();
    assert_eq!(schema["type"], "array");
    assert_eq!(schema["items"]["type"], "string");
}
