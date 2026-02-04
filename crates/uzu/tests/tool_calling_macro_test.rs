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
fn test_tool_calling_generates_tool_definition() {
    let tool_implementation = GetCurrentTemperatureToolImplementation;
    let tool = tool_implementation.tool();

    match &tool {
        Tool::Function {
            function,
        } => {
            assert_eq!(function.name, "get_current_temperature");
            assert_eq!(
                function.description,
                "Get the current temperature at a location."
            );
            assert!(function.parameters.properties.contains_key("location"));
            assert!(function.parameters.properties.contains_key("unit"));
            assert!(
                function.parameters.required.contains(&"location".to_string())
            );
            assert!(function.parameters.required.contains(&"unit".to_string()));
        },
    }
}

#[test]
fn test_tool_calling_serializes_to_correct_format() {
    let tool_implementation = GetCurrentTemperatureToolImplementation;
    let tool = tool_implementation.tool();

    let json = serde_json::to_value(&tool).unwrap();

    assert_eq!(json["type"], "function");
    assert_eq!(json["function"]["name"], "get_current_temperature");
    assert_eq!(
        json["function"]["description"],
        "Get the current temperature at a location."
    );
    assert!(json["function"]["parameters"].is_object());
}

#[test]
fn test_tool_calling_call_executes_function() {
    let tool_implementation = GetCurrentTemperatureToolImplementation;

    let tool_call = ToolCall::new(
        "get_current_temperature".to_string(),
        Value::from(serde_json::json!({
            "location": "New York, US",
            "unit": "celsius"
        })),
    );

    let result = tool_implementation.call(&tool_call).unwrap();

    assert!(result.is_success());
    assert_eq!(
        result.content,
        ToolCallResultContent::Success(Value::Double(30.0))
    );
}

#[test]
fn test_tool_calling_includes_parameter_descriptions() {
    let tool_implementation = GetCurrentTemperatureToolImplementation;
    let tool = tool_implementation.tool();

    let json = serde_json::to_value(&tool).unwrap();

    assert_eq!(
        json["function"]["parameters"]["properties"]["location"]["description"],
        "The location to get the temperature for, in the format \"City, Country\""
    );
    assert_eq!(
        json["function"]["parameters"]["properties"]["unit"]["description"],
        "The unit to return the temperature in."
    );
}

#[tool(description = "Divide two numbers.")]
fn divide(
    /// The dividend
    dividend: f64,
    /// The divisor
    divisor: f64,
) -> Result<f64, String> {
    if divisor == 0.0 {
        return Err("Cannot divide by zero".to_string());
    }
    Ok(dividend / divisor)
}

#[test]
fn test_tool_calling_fallible_function_success() {
    let tool_implementation = DivideToolImplementation;

    let tool_call = ToolCall::new(
        "divide".to_string(),
        Value::from(serde_json::json!({
            "dividend": 10.0,
            "divisor": 2.0
        })),
    );

    let result = tool_implementation.call(&tool_call).unwrap();

    assert!(result.is_success());
    assert_eq!(
        result.content,
        ToolCallResultContent::Success(Value::Double(5.0))
    );
}

#[test]
fn test_tool_calling_fallible_function_error() {
    let tool_implementation = DivideToolImplementation;

    let tool_call = ToolCall::new(
        "divide".to_string(),
        Value::from(serde_json::json!({
            "dividend": 10.0,
            "divisor": 0.0
        })),
    );

    let result = tool_implementation.call(&tool_call).unwrap();

    assert!(!result.is_success());
    assert_eq!(
        result.content,
        ToolCallResultContent::Failure("Cannot divide by zero".to_string())
    );
}
