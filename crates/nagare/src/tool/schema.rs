use schemars::generate::SchemaSettings;
pub use schemars::{JsonSchema, Schema};
use serde_json::{Map, Value as JsonValue};
use shoji::types::basic::Value;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SchemaContract {
    /// Schema for arguments accepted by Serde deserialization.
    Input,
    /// Schema for values produced by Serde serialization.
    Output,
}

pub fn tool_schema<T: JsonSchema>(contract: SchemaContract) -> Schema {
    let settings = SchemaSettings::draft07().with(|settings| {
        settings.meta_schema = None;
        settings.inline_subschemas = true;
    });
    let settings = match contract {
        SchemaContract::Input => settings.for_deserialize(),
        SchemaContract::Output => settings.for_serialize(),
    };

    let mut schema = settings.into_generator().into_root_schema_for::<T>();
    schema.remove("title");
    schema
}

pub fn tool_schema_value<T: JsonSchema>(contract: SchemaContract) -> Value {
    schema_value(tool_schema::<T>(contract))
}

/// Builds the outer object schema owned by the function macro.
pub fn object_schema<PK, P, RK, R>(
    properties: P,
    required: R,
) -> Schema
where
    PK: Into<String>,
    P: IntoIterator<Item = (PK, Schema)>,
    RK: Into<String>,
    R: IntoIterator<Item = RK>,
{
    let properties =
        properties.into_iter().map(|(name, schema)| (name.into(), schema.to_value())).collect::<Map<_, _>>();
    let required: Vec<JsonValue> = required.into_iter().map(|name| JsonValue::String(name.into())).collect();

    Schema::try_from(serde_json::json!({
        "type": "object",
        "properties": properties,
        "required": required,
    }))
    .expect("the generated tool parameter schema is an object")
}

pub fn with_description(
    mut schema: Schema,
    description: impl Into<String>,
) -> Schema {
    schema.insert("description".into(), description.into().into());
    schema
}

pub fn schema_value(schema: Schema) -> Value {
    Value::from(schema.to_value())
}
