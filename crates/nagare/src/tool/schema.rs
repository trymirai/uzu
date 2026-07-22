use indexmap::IndexMap;
pub use proc_macros::UzuToolSchema;
use serde::{Deserialize, Serialize};
use shoji::types::basic::Value;

pub trait ToolSchema {
    fn json_schema() -> JsonSchema;
}

macro_rules! impl_tool_schema {
    ($constructor:ident => $($ty:ty),+) => {
        $(
            impl ToolSchema for $ty {
                fn json_schema() -> JsonSchema {
                    JsonSchema::$constructor()
                }
            }
        )+
    };
}

impl_tool_schema!(string => String);
impl_tool_schema!(number => f32, f64);
impl_tool_schema!(integer => i8, i16, i32, i64, isize, u8, u16, u32, u64, usize);
impl_tool_schema!(boolean => bool);

impl<T: ToolSchema> ToolSchema for Vec<T> {
    fn json_schema() -> JsonSchema {
        JsonSchema::array(T::json_schema())
    }
}

impl<T: ToolSchema> ToolSchema for Option<T> {
    fn json_schema() -> JsonSchema {
        T::json_schema()
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct JsonSchema {
    #[serde(flatten)]
    pub schema_type: JsonSchemaType,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum JsonSchemaType {
    String {
        #[serde(rename = "enum", skip_serializing_if = "Option::is_none")]
        enum_values: Option<Vec<String>>,
    },
    Number,
    Integer,
    Boolean,
    Array {
        items: Box<JsonSchema>,
    },
    Object {
        #[serde(default)]
        properties: IndexMap<String, JsonSchema>,
        #[serde(default)]
        required: Vec<String>,
    },
}

impl JsonSchema {
    pub fn string() -> Self {
        Self {
            schema_type: JsonSchemaType::String {
                enum_values: None,
            },
            description: None,
        }
    }

    pub fn number() -> Self {
        Self {
            schema_type: JsonSchemaType::Number,
            description: None,
        }
    }

    pub fn integer() -> Self {
        Self {
            schema_type: JsonSchemaType::Integer,
            description: None,
        }
    }

    pub fn boolean() -> Self {
        Self {
            schema_type: JsonSchemaType::Boolean,
            description: None,
        }
    }

    pub fn array(items: JsonSchema) -> Self {
        Self {
            schema_type: JsonSchemaType::Array {
                items: Box::new(items),
            },
            description: None,
        }
    }

    pub fn object<PK, P, RK, R>(
        properties: P,
        required: R,
    ) -> Self
    where
        PK: Into<String>,
        P: IntoIterator<Item = (PK, JsonSchema)>,
        RK: Into<String>,
        R: IntoIterator<Item = RK>,
    {
        Self {
            schema_type: JsonSchemaType::Object {
                properties: properties.into_iter().map(|(key, schema)| (key.into(), schema)).collect(),
                required: required.into_iter().map(Into::into).collect(),
            },
            description: None,
        }
    }

    pub fn empty_object() -> Self {
        Self {
            schema_type: JsonSchemaType::Object {
                properties: IndexMap::new(),
                required: Vec::new(),
            },
            description: None,
        }
    }

    pub fn with_description(
        mut self,
        description: impl Into<String>,
    ) -> Self {
        self.description = Some(description.into());
        self
    }

    pub fn with_enum_values(
        mut self,
        values: Vec<String>,
    ) -> Self {
        if let JsonSchemaType::String {
            enum_values,
        } = &mut self.schema_type
        {
            *enum_values = Some(values);
        }
        self
    }

    pub fn to_value(&self) -> Value {
        Value {
            json: serde_json::to_string(self).expect("JsonSchema serialization is infallible"),
        }
    }
}
