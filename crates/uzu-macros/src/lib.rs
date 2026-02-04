//! Procedural macros for the uzu crate.

mod tool_calling;

use proc_macro::TokenStream;
use syn::parse_macro_input;

/// Attribute macro to generate tool calling infrastructure for a function.
///
/// This macro generates:
/// - A parameters struct with serde/schemars derives
/// - A tool struct implementing `ToolFunction` trait
/// - The tool definition with JSON Schema for parameters
///
/// # Example
///
/// ```rust,ignore
/// use uzu::tool_calling::tool;
///
/// #[tool(description = "Get the current temperature at a location.")]
/// fn get_current_temperature(
///     /// The location to get the temperature for, in the format "City, Country"
///     location: String,
///     /// The unit to return the temperature in.
///     unit: TemperatureUnit,
/// ) -> f64 {
///     42.0
/// }
///
/// // This generates:
/// // - `GetCurrentTemperatureToolParams` struct
/// // - `GetCurrentTemperatureTool` struct implementing `ToolFunction`
/// ```
#[proc_macro_attribute]
pub fn tool(attr: TokenStream, item: TokenStream) -> TokenStream {
    let args = parse_macro_input!(attr as tool_calling::tool_attribute::ToolArguments);
    let input_function = parse_macro_input!(item as syn::ItemFn);

    match tool_calling::expand_tool_attribute(args, input_function) {
        Ok(tokens) => tokens.into(),
        Err(err) => err.to_compile_error().into(),
    }
}
