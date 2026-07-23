#![cfg(not(target_family = "wasm"))]

use std::sync::{
    Arc,
    atomic::{AtomicI64, Ordering},
};

use nagare::tool::{func_def::ToolFunctionDefinition, uzu_tool_closure};

#[tokio::test]
async fn closure_with_captured_state() {
    let counter = Arc::new(AtomicI64::new(0));
    let captured = counter.clone();
    let definition: ToolFunctionDefinition = uzu_tool_closure! {
        /// Add an amount to the running total.
        accumulate: |
            /// Amount to add to the total.
            amount: i64,
        | -> i64 {
            captured.fetch_add(amount, Ordering::SeqCst) + amount
        }
    };

    assert_eq!(definition.name(), "accumulate");
    assert_eq!(definition.description(), "Add an amount to the running total.");

    let parameters: serde_json::Value = serde_json::from_str(&definition.parameters().as_ref().unwrap().json).unwrap();
    assert_eq!(
        parameters,
        serde_json::json!({
            "type": "object",
            "properties": {
                "amount": { "type": "integer", "description": "Amount to add to the total." }
            },
            "required": ["amount"]
        })
    );

    let return_definition: serde_json::Value =
        serde_json::from_str(&definition.return_definition().as_ref().unwrap().json).unwrap();
    assert_eq!(return_definition, serde_json::json!({ "type": "integer" }));

    let result: serde_json::Value =
        definition.execute(serde_json::json!({ "amount": 40 }).into()).await.unwrap().try_into().unwrap();
    assert_eq!(result, serde_json::json!(40));

    let result: serde_json::Value =
        definition.execute(serde_json::json!({ "amount": 2 }).into()).await.unwrap().try_into().unwrap();
    assert_eq!(result, serde_json::json!(42));

    assert_eq!(counter.load(Ordering::SeqCst), 42);

    let error = definition.execute(serde_json::json!({ "amount": "oops" }).into()).await.unwrap_err();
    assert!(error.to_string().contains("amount"), "unexpected error: {error}");
}

#[tokio::test]
async fn async_closure_with_optional_parameter() {
    let greeting = String::from("Hello");
    let definition = uzu_tool_closure! {
        /// Greet the given person.
        greet: async |
            /// Name of the person to greet.
            name: String,
            /// Optional punctuation to end the greeting with.
            punctuation: Option<String>,
        | -> String {
            let punctuation = punctuation.unwrap_or_else(|| "!".to_string());
            format!("{greeting}, {name}{punctuation}")
        }
    };

    assert_eq!(definition.name(), "greet");
    assert_eq!(definition.description(), "Greet the given person.");

    let parameters: serde_json::Value = serde_json::from_str(&definition.parameters().as_ref().unwrap().json).unwrap();
    assert_eq!(
        parameters,
        serde_json::json!({
            "type": "object",
            "properties": {
                "name": { "type": "string", "description": "Name of the person to greet." },
                "punctuation": {
                    "type": "string",
                    "description": "Optional punctuation to end the greeting with."
                }
            },
            "required": ["name"]
        })
    );

    let result: serde_json::Value =
        definition.execute(serde_json::json!({ "name": "Ada" }).into()).await.unwrap().try_into().unwrap();
    assert_eq!(result, serde_json::json!("Hello, Ada!"));

    let result: serde_json::Value = definition
        .execute(serde_json::json!({ "name": "Ada", "punctuation": "?" }).into())
        .await
        .unwrap()
        .try_into()
        .unwrap();
    assert_eq!(result, serde_json::json!("Hello, Ada?"));
}

#[tokio::test]
async fn result_closure_propagates_errors() {
    let definition = uzu_tool_closure! {
        /// Divide the dividend by the divisor.
        divide: |dividend: f64, divisor: f64| -> Result<f64, Box<dyn std::error::Error + Send + Sync>> {
            if divisor == 0.0 {
                Err("division by zero".into())
            } else {
                Ok(dividend / divisor)
            }
        }
    };

    let return_definition: serde_json::Value =
        serde_json::from_str(&definition.return_definition().as_ref().unwrap().json).unwrap();
    assert_eq!(return_definition, serde_json::json!({ "type": "number" }));

    let result: serde_json::Value = definition
        .execute(serde_json::json!({ "dividend": 10.0, "divisor": 4.0 }).into())
        .await
        .unwrap()
        .try_into()
        .unwrap();
    assert_eq!(result, serde_json::json!(2.5));

    let error = definition.execute(serde_json::json!({ "dividend": 1.0, "divisor": 0.0 }).into()).await.unwrap_err();
    assert_eq!(error.to_string(), "division by zero");
}

#[tokio::test]
async fn closure_without_parameters_or_annotation() {
    let definition = uzu_tool_closure! {
        /// Report the answer.
        answer: || 42
    };

    assert_eq!(definition.name(), "answer");
    assert!(definition.parameters().is_none());
    assert!(definition.return_definition().is_none());

    let result: serde_json::Value = definition.execute(serde_json::json!({}).into()).await.unwrap().try_into().unwrap();
    assert_eq!(result, serde_json::json!(42));
}

#[tokio::test]
async fn unit_closure_returns_null() {
    let flag = Arc::new(AtomicI64::new(0));
    let captured = flag.clone();
    let definition = uzu_tool_closure! {
        /// Reset the session state.
        reset: || {
            captured.store(1, Ordering::SeqCst);
        }
    };

    assert!(definition.parameters().is_none());
    assert!(definition.return_definition().is_none());

    let result: serde_json::Value = definition.execute(serde_json::json!({}).into()).await.unwrap().try_into().unwrap();
    assert_eq!(result, serde_json::Value::Null);
    assert_eq!(flag.load(Ordering::SeqCst), 1);
}
