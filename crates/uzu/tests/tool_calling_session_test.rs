use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
mod common;
use uzu::{
    session::{
        ChatSession,
        config::{DecodingConfig, RunConfig},
        types::{Input, Output},
    },
    tool_calling::*,
};

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

#[derive(Debug, Deserialize, JsonSchema)]
#[allow(unused)]
struct WindSpeedParameters {
    /// The location to get the wind speed for, in the format "City, Country"
    location: String,
}

#[test]
fn test_tool_calling_session() {
    let mut tool_registry = ToolRegistry::new();
    tool_registry.register(GetCurrentTemperatureToolImplementation);
    tool_registry.register_lambda(
        "get_current_wind_speed".to_string(),
        "Get the current wind speed in km/h at a given location.".to_string(),
        |_: WindSpeedParameters| -> Result<f64, String> {
            return Ok(10.0);
        },
    );

    let model_path = common::get_test_model_path();
    let config = DecodingConfig::default();
    let mut session =
        ChatSession::new(model_path, config, Some(tool_registry)).unwrap();
    let input = Input::Text(String::from(
        "What is temperature and wind speed in London?",
    ));
    let output = session
        .run(
            input,
            RunConfig::default().tokens_limit(512),
            Some(|_: Output| {
                return true;
            }),
        )
        .unwrap();
    println!("output: {:?}", output.text.original);
}
