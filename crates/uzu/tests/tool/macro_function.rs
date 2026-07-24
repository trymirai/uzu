#![cfg(not(target_family = "wasm"))]

use nagare::tool::{func_def::ToolFunctionDefinition, schema::UzuToolSchema, uzu_tool_function};
use serde::{Deserialize, Serialize};

/// The current weather at the requested location.
#[derive(Serialize, UzuToolSchema)]
struct Weather {
    /// Temperature in degrees Celsius.
    temperature: f64,
    /// Optional human-readable summary.
    summary: Option<String>,
}

/// Get the current weather for the given geographic coordinates.
#[uzu_tool_function]
fn get_weather(
    /// Latitude in decimal degrees.
    latitude: f64,
    /// Longitude in decimal degrees.
    longitude: f64,
) -> Result<Weather, Box<dyn std::error::Error + Send + Sync>> {
    Ok(Weather {
        temperature: latitude + longitude,
        summary: None,
    })
}

/// An async tool without parameters.
#[uzu_tool_function]
async fn current_time() -> String {
    "17:03".to_string()
}

#[tokio::test]
async fn check_generated_definition() {
    let definition: ToolFunctionDefinition = get_weather.into();
    assert_eq!(definition.name(), "get_weather");
    assert_eq!(definition.description(), "Get the current weather for the given geographic coordinates.");

    let parameters: serde_json::Value = serde_json::from_str(&definition.parameters().as_ref().unwrap().json).unwrap();
    assert_eq!(
        parameters,
        serde_json::json!({
            "type": "object",
            "properties": {
                "latitude": { "type": "number", "description": "Latitude in decimal degrees." },
                "longitude": { "type": "number", "description": "Longitude in decimal degrees." }
            },
            "required": ["latitude", "longitude"]
        })
    );

    let return_definition: serde_json::Value =
        serde_json::from_str(&definition.return_definition().as_ref().unwrap().json).unwrap();
    assert_eq!(
        return_definition,
        serde_json::json!({
            "type": "object",
            "properties": {
                "temperature": { "type": "number", "description": "Temperature in degrees Celsius." },
                "summary": { "type": "string", "description": "Optional human-readable summary." }
            },
            "required": ["temperature"],
            "description": "The current weather at the requested location."
        })
    );

    let result = definition.execute(serde_json::json!({ "latitude": 10.5, "longitude": 20.25 }).into()).await.unwrap();
    let result: serde_json::Value = result.try_into().unwrap();
    assert_eq!(result, serde_json::json!({ "temperature": 30.75, "summary": null }));

    let error = definition.execute(serde_json::json!({ "latitude": "oops" }).into()).await.unwrap_err();
    assert!(error.to_string().contains("latitude"), "unexpected error: {error}");

    let time: ToolFunctionDefinition = current_time.into();
    assert_eq!(time.name(), "current_time");
    assert!(time.parameters().is_none());
    let time_schema: serde_json::Value =
        serde_json::from_str(&time.return_definition().as_ref().unwrap().json).unwrap();
    assert_eq!(time_schema, serde_json::json!({ "type": "string" }));
    let result: serde_json::Value = time.execute(serde_json::json!({}).into()).await.unwrap().try_into().unwrap();
    assert_eq!(result, serde_json::json!("17:03"));
}

/// A geographic coordinate.
#[derive(Serialize, Deserialize, UzuToolSchema)]
struct Coordinate {
    /// Latitude in decimal degrees.
    latitude: f64,
    /// Longitude in decimal degrees.
    longitude: f64,
}

/// A weather forecast request.
#[derive(Deserialize, UzuToolSchema)]
struct ForecastRequest {
    /// Location to forecast.
    location: Coordinate,
    /// Number of days ahead.
    days: u32,
}

/// The forecast for a single day.
#[derive(Serialize, UzuToolSchema)]
struct DayForecast {
    /// Temperature in degrees Celsius.
    temperature: f64,
    /// Chance of rain from 0 to 1.
    rain_probability: f64,
}

/// A forecast for the requested location.
#[derive(Serialize, UzuToolSchema)]
struct Forecast {
    location: Coordinate,
    /// Forecasts per day.
    days: Vec<DayForecast>,
}

/// Get the weather forecast for a location.
#[uzu_tool_function]
fn get_forecast(request: ForecastRequest) -> Forecast {
    Forecast {
        location: request.location,
        days: (0..request.days)
            .map(|day| DayForecast {
                temperature: 20.0 + day as f64,
                rain_probability: 0.25,
            })
            .collect(),
    }
}

#[tokio::test]
async fn structured_input_and_output() {
    let definition: ToolFunctionDefinition = get_forecast.into();
    assert_eq!(definition.name(), "get_forecast");
    assert_eq!(definition.description(), "Get the weather forecast for a location.");

    let coordinate_properties = serde_json::json!({
        "latitude": { "type": "number", "description": "Latitude in decimal degrees." },
        "longitude": { "type": "number", "description": "Longitude in decimal degrees." }
    });

    let parameters: serde_json::Value = serde_json::from_str(&definition.parameters().as_ref().unwrap().json).unwrap();
    assert_eq!(
        parameters,
        serde_json::json!({
            "type": "object",
            "properties": {
                "request": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "object",
                            "properties": coordinate_properties,
                            "required": ["latitude", "longitude"],
                            // The field doc overrides the struct-level description of `Coordinate`.
                            "description": "Location to forecast."
                        },
                        "days": { "type": "integer", "description": "Number of days ahead." }
                    },
                    "required": ["location", "days"],
                    "description": "A weather forecast request."
                }
            },
            "required": ["request"]
        })
    );

    let return_definition: serde_json::Value =
        serde_json::from_str(&definition.return_definition().as_ref().unwrap().json).unwrap();
    assert_eq!(
        return_definition,
        serde_json::json!({
            "type": "object",
            "properties": {
                "location": {
                    "type": "object",
                    "properties": coordinate_properties,
                    "required": ["latitude", "longitude"],
                    // No field doc, so the struct-level description of `Coordinate` is kept.
                    "description": "A geographic coordinate."
                },
                "days": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "temperature": { "type": "number", "description": "Temperature in degrees Celsius." },
                            "rain_probability": { "type": "number", "description": "Chance of rain from 0 to 1." }
                        },
                        "required": ["temperature", "rain_probability"],
                        "description": "The forecast for a single day."
                    },
                    "description": "Forecasts per day."
                }
            },
            "required": ["location", "days"],
            "description": "A forecast for the requested location."
        })
    );

    let result = definition
        .execute(
            serde_json::json!({
                "request": {
                    "location": { "latitude": 59.94, "longitude": 30.31 },
                    "days": 2
                }
            })
            .into(),
        )
        .await
        .unwrap();
    let result: serde_json::Value = result.try_into().unwrap();
    assert_eq!(
        result,
        serde_json::json!({
            "location": { "latitude": 59.94, "longitude": 30.31 },
            "days": [
                { "temperature": 20.0, "rain_probability": 0.25 },
                { "temperature": 21.0, "rain_probability": 0.25 }
            ]
        })
    );

    let error = definition
        .execute(serde_json::json!({ "request": { "location": { "latitude": 59.94 }, "days": 2 } }).into())
        .await
        .unwrap_err();
    assert!(error.to_string().contains("request"), "unexpected error: {error}");
}

#[derive(Serialize, Deserialize, UzuToolSchema)]
#[serde(rename_all = "camelCase")]
struct UserLookup {
    #[serde(rename = "id")]
    user_id: String,
    display_name: String,
    #[serde(default)]
    max_results: u32,
    #[serde(skip)]
    internal_note: String,
}

#[uzu_tool_function]
fn lookup_user(request: UserLookup) -> String {
    let _ = &request.internal_note;
    format!("{}:{}:{}", request.user_id, request.display_name, request.max_results)
}

#[tokio::test]
async fn serde_field_attributes_match_generated_schema() {
    let definition: ToolFunctionDefinition = lookup_user.into();
    let parameters: serde_json::Value = serde_json::from_str(&definition.parameters().as_ref().unwrap().json).unwrap();

    assert_eq!(
        parameters,
        serde_json::json!({
            "type": "object",
            "properties": {
                "request": {
                    "type": "object",
                    "properties": {
                        "id": { "type": "string" },
                        "displayName": { "type": "string" },
                        "maxResults": { "type": "integer" }
                    },
                    "required": ["id", "displayName"]
                }
            },
            "required": ["request"]
        })
    );

    let result: serde_json::Value = definition
        .execute(
            serde_json::json!({
                "request": {
                    "id": "00123",
                    "displayName": "Ada"
                }
            })
            .into(),
        )
        .await
        .unwrap()
        .try_into()
        .unwrap();
    assert_eq!(result, serde_json::json!("00123:Ada:0"));
}

#[derive(Deserialize, UzuToolSchema)]
#[serde(deny_unknown_fields)]
struct StrictRequest {
    query: String,
}

#[uzu_tool_function]
fn strict_lookup(request: StrictRequest) -> String {
    request.query
}

#[tokio::test]
async fn deny_unknown_fields_disallows_additional_schema_properties() {
    let definition: ToolFunctionDefinition = strict_lookup.into();
    let parameters: serde_json::Value = serde_json::from_str(&definition.parameters().as_ref().unwrap().json).unwrap();

    assert_eq!(
        parameters,
        serde_json::json!({
            "type": "object",
            "properties": {
                "request": {
                    "type": "object",
                    "properties": {
                        "query": { "type": "string" }
                    },
                    "required": ["query"],
                    "additionalProperties": false
                }
            },
            "required": ["request"]
        })
    );

    let error = definition
        .execute(serde_json::json!({ "request": { "query": "Ada", "unexpected": true } }).into())
        .await
        .unwrap_err();
    assert!(error.to_string().contains("unknown field"), "unexpected error: {error}");
}

/// Add two integers.
#[uzu_tool_function]
fn add(
    /// The first addend.
    a: i64,
    /// The second addend.
    b: i64,
) -> i64 {
    a + b
}

#[tokio::test]
async fn primitive_input_and_output() {
    let definition: ToolFunctionDefinition = add.into();
    assert_eq!(definition.name(), "add");
    assert_eq!(definition.description(), "Add two integers.");

    let parameters: serde_json::Value = serde_json::from_str(&definition.parameters().as_ref().unwrap().json).unwrap();
    assert_eq!(
        parameters,
        serde_json::json!({
            "type": "object",
            "properties": {
                "a": { "type": "integer", "description": "The first addend." },
                "b": { "type": "integer", "description": "The second addend." }
            },
            "required": ["a", "b"]
        })
    );

    let return_definition: serde_json::Value =
        serde_json::from_str(&definition.return_definition().as_ref().unwrap().json).unwrap();
    assert_eq!(return_definition, serde_json::json!({ "type": "integer" }));

    let result: serde_json::Value =
        definition.execute(serde_json::json!({ "a": 40, "b": 2 }).into()).await.unwrap().try_into().unwrap();
    assert_eq!(result, serde_json::json!(42));
}

/// Reset the session state.
#[uzu_tool_function]
fn reset() {}

#[tokio::test]
async fn void_input_and_output() {
    let definition: ToolFunctionDefinition = reset.into();
    assert_eq!(definition.name(), "reset");
    assert_eq!(definition.description(), "Reset the session state.");
    assert!(definition.parameters().is_none());
    assert!(definition.return_definition().is_none());

    let result: serde_json::Value = definition.execute(serde_json::json!({}).into()).await.unwrap().try_into().unwrap();
    assert_eq!(result, serde_json::Value::Null);
}

/// Clear all stored data.
#[uzu_tool_function]
fn clear_data(
    /// Must be true to confirm the deletion.
    confirm: bool
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    if confirm {
        Ok(())
    } else {
        Err("confirmation required".into())
    }
}

#[tokio::test]
async fn void_result_and_error_propagation() {
    let definition: ToolFunctionDefinition = clear_data.into();
    assert_eq!(definition.name(), "clear_data");

    let parameters: serde_json::Value = serde_json::from_str(&definition.parameters().as_ref().unwrap().json).unwrap();
    assert_eq!(
        parameters,
        serde_json::json!({
            "type": "object",
            "properties": {
                "confirm": { "type": "boolean", "description": "Must be true to confirm the deletion." }
            },
            "required": ["confirm"]
        })
    );
    assert!(definition.return_definition().is_none());

    let result: serde_json::Value =
        definition.execute(serde_json::json!({ "confirm": true }).into()).await.unwrap().try_into().unwrap();
    assert_eq!(result, serde_json::Value::Null);

    let error = definition.execute(serde_json::json!({ "confirm": false }).into()).await.unwrap_err();
    assert_eq!(error.to_string(), "confirmation required");
}
