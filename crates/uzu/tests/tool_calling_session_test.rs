use std::process::Command;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
mod common;
use uzu::{
    session::{
        ChatSession,
        config::{DecodingConfig, RunConfig},
        types::{Input, Output, ParsedSection},
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

#[derive(Debug, Serialize, JsonSchema)]
struct BashOutput {
    stdout: String,
    stderr: String,
    exit_code: i32,
}

#[tool(description = "Execute a bash command and return its output.")]
fn bash(
    /// The bash command to execute
    command: String
) -> Result<BashOutput, String> {
    let output = Command::new("/bin/bash")
        .arg("-c")
        .arg(&command)
        .output()
        .map_err(|error| format!("Failed to execute bash: {}", error))?;

    Ok(BashOutput {
        stdout: String::from_utf8_lossy(&output.stdout).into_owned(),
        stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
        exit_code: output.status.code().unwrap_or(-1),
    })
}

#[test]
fn test_tool_calling_session_base() {
    let mut tool_registry = ToolRegistry::new();
    tool_registry.register(GetCurrentTemperatureToolImplementation);
    tool_registry.register_lambda(
        "get_current_wind_speed".to_string(),
        "Get the current wind speed in km/h at a given location.".to_string(),
        |_: WindSpeedParameters| -> Result<f64, String> {
            return Ok(10.0);
        },
    );
    run_with_tools_registry(
        "What is temperature and wind speed in London?".to_string(),
        tool_registry,
    );
}

#[test]
fn test_tool_calling_session_bash() {
    let mut tool_registry = ToolRegistry::new();
    tool_registry.register(BashToolImplementation);
    run_with_tools_registry(
        "Find the Desktop folder and list its contents".to_string(),
        tool_registry,
    );
}

fn print_sections(
    sections: &[ParsedSection],
    prefix: String,
) {
    let names: Vec<&str> = sections
        .iter()
        .map(|section| match section {
            ParsedSection::ChainOfThought(_) => "ChainOfThought",
            ParsedSection::Response(_) => "Response",
            ParsedSection::ToolCallCandidate(_) => "ToolCallCandidate",
            ParsedSection::ToolCall(_) => "ToolCall",
        })
        .collect();
    println!("{}: {}", prefix, names.join(", "));
}

fn run_with_tools_registry(
    prompt: String,
    tool_registry: ToolRegistry,
) {
    let model_path = common::get_test_model_path();
    let config = DecodingConfig::default();
    let mut session =
        ChatSession::new(model_path, config, Some(tool_registry)).unwrap();
    let input = Input::Text(prompt);
    let output = session
        .run(
            input,
            RunConfig::default().tokens_limit(2048),
            Some(|output: Output| {
                print_sections(
                    &output.text.parsed.sections,
                    "Chunk".to_string(),
                );
                return true;
            }),
        )
        .unwrap();
    println!("-------------------------");
    print_sections(&output.text.parsed.sections, "Final".to_string());
    println!("-------------------------");
    println!(
        "Chain of thought: {}",
        output.text.parsed.chain_of_thought().unwrap_or("None".to_string())
    );
    println!("-------------------------");
    println!(
        "Response: {}",
        output.text.parsed.response().unwrap_or("None".to_string())
    );
    println!("-------------------------");
    for tool_call in output.text.parsed.tool_calls() {
        println!("Tool call: {:#?}", tool_call);
        println!("-------------------------");
    }
    for tool_call_candidate in output.text.parsed.tool_call_candidates() {
        println!("Tool call candidate: {}", tool_call_candidate);
        println!("-------------------------");
    }
    println!("Original: {}", output.text.original);
    println!("-------------------------");
}
