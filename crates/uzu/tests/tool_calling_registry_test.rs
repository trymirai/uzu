use std::sync::Arc;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use uzu::tool_calling::*;

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "lowercase")]
enum TemperatureUnit {
    Celsius,
    Fahrenheit,
}

#[tool(description = "Get the current temperature at a location.")]
#[allow(unused_variables)]
fn get_current_temperature(
    /// The location to get the temperature for, in the format "City, Country"
    location: String,
    /// The unit to return the temperature in.
    unit: TemperatureUnit,
) -> f64 {
    match unit {
        TemperatureUnit::Celsius => 30.0,
        TemperatureUnit::Fahrenheit => 86.0,
    }
}

#[test]
fn test_tool_calling_register_and_call_macro_tool() {
    let mut registry = ToolRegistry::new();
    registry.register(GetCurrentTemperatureToolImplementation);

    assert!(registry.contains("get_current_temperature"));

    let tool_call = ToolCall::new(
        "get_current_temperature".to_string(),
        Value::from(serde_json::json!({
            "location": "London, UK",
            "unit": "fahrenheit"
        })),
    );

    let result = registry.execute(&tool_call).unwrap();

    assert!(result.is_success());
    assert_eq!(
        result.content,
        ToolCallResultContent::Success(Value::Double(86.0))
    );
}

#[test]
fn test_tool_calling_execute_tool_call() {
    let mut registry = ToolRegistry::new();
    registry.register(GetCurrentTemperatureToolImplementation);

    let tool_call = ToolCall::new(
        "get_current_temperature".to_string(),
        Value::from(serde_json::json!({
            "location": "Paris, FR",
            "unit": "celsius"
        })),
    );

    let result = registry.execute(&tool_call).unwrap();

    assert_eq!(result.tool_call_id, tool_call.id);
    assert_eq!(result.name, "get_current_temperature");
    assert_eq!(
        result.content,
        ToolCallResultContent::Success(Value::Double(30.0))
    );
}

#[test]
fn test_tool_calling_list_registered_tools() {
    let mut registry = ToolRegistry::new();
    registry.register(GetCurrentTemperatureToolImplementation);

    let tools = registry.tools();
    assert_eq!(tools.len(), 1);

    let json = serde_json::to_value(&tools).unwrap();
    assert!(json.is_array());
    assert_eq!(json[0]["type"], "function");
}

#[derive(Debug, Deserialize, JsonSchema)]
struct GreetParameters {
    /// The name of the person to greet
    name: String,
}

#[test]
fn test_tool_calling_register_lambda_tool() {
    let mut registry = ToolRegistry::new();

    registry.register_lambda(
        "greet".to_string(),
        "Greet a person by name".to_string(),
        |parameters: GreetParameters| -> Result<String, std::convert::Infallible> {
            Ok(format!("Hello, {}!", parameters.name))
        },
    );

    assert!(registry.contains("greet"));

    let tool_call = ToolCall::new(
        "greet".to_string(),
        Value::from(serde_json::json!({
            "name": "Alice"
        })),
    );

    let result = registry.execute(&tool_call).unwrap();

    assert!(result.is_success());
    assert_eq!(
        result.content,
        ToolCallResultContent::Success(Value::String(
            "Hello, Alice!".to_string()
        ))
    );

    let tools = registry.tools();
    let tool_json = serde_json::to_value(&tools[0]).unwrap();
    assert_eq!(tool_json["type"], "function");
    assert_eq!(tool_json["function"]["name"], "greet");
}

#[test]
fn test_tool_calling_register_lambda_with_error() {
    let mut registry = ToolRegistry::new();

    registry.register_lambda(
        "divide".to_string(),
        "Divide two numbers".to_string(),
        |parameters: DivideParameters| -> Result<f64, String> {
            if parameters.divisor == 0.0 {
                return Err("Cannot divide by zero".to_string());
            }
            Ok(parameters.dividend / parameters.divisor)
        },
    );

    let tool_call = ToolCall::new(
        "divide".to_string(),
        Value::from(serde_json::json!({"dividend": 10.0, "divisor": 2.0})),
    );
    let result = registry.execute(&tool_call).unwrap();
    assert!(result.is_success());
    assert_eq!(
        result.content,
        ToolCallResultContent::Success(Value::Double(5.0))
    );

    let tool_call = ToolCall::new(
        "divide".to_string(),
        Value::from(serde_json::json!({"dividend": 10.0, "divisor": 0.0})),
    );
    let result = registry.execute(&tool_call).unwrap();
    assert!(!result.is_success());
    assert_eq!(
        result.content,
        ToolCallResultContent::Failure("Cannot divide by zero".to_string())
    );
}

#[derive(Debug, Deserialize, JsonSchema)]
struct DivideParameters {
    dividend: f64,
    divisor: f64,
}

#[test]
fn test_tool_calling_register_handler() {
    let mut registry = ToolRegistry::new();

    let tool = ToolFunction::from_parameters(
        "echo".to_string(),
        "Echo the input message".to_string(),
        vec![ToolParameter::required(
            "message".to_string(),
            ToolParameterType::string(),
            "The message to echo".to_string(),
        )],
    )
    .into_tool();

    let handler: ToolHandler = Arc::new(|tool_call: &ToolCall| {
        if let Some(arguments) = tool_call.arguments.as_object() {
            if let Some(Value::String(message)) = arguments.get("message") {
                return Ok(Value::String(format!("Echo: {}", message)));
            }
        }
        Err("Missing message parameter".to_string())
    });

    registry.register_handler(tool, handler);

    assert!(registry.contains("echo"));

    let tool_call = ToolCall::new(
        "echo".to_string(),
        Value::Object(
            [("message".to_string(), Value::String("Hello!".to_string()))]
                .into_iter()
                .collect(),
        ),
    );

    let result = registry.execute(&tool_call).unwrap();
    assert_eq!(result.tool_call_id, tool_call.id);
    assert_eq!(
        result.content,
        ToolCallResultContent::Success(Value::String(
            "Echo: Hello!".to_string()
        ))
    );
}

#[test]
fn test_tool_calling_tool_not_found() {
    let registry = ToolRegistry::new();

    let tool_call = ToolCall::new("nonexistent".to_string(), Value::Null);
    let result = registry.execute(&tool_call);

    assert!(result.is_err());
    match result {
        Err(ToolError::NotFound {
            name,
        }) => assert_eq!(name, "nonexistent"),
        _ => panic!("Expected NotFound error"),
    }
}
