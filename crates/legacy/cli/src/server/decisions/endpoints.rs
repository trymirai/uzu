use std::{sync::Arc, time::Instant};

use rocket::{
    State,
    data::{ByteUnit, Data},
    http::Status,
    post,
    response::status,
    serde::json::Json,
};
use tracing::Instrument;
use uzu::session::decision::DecisionSessionError;

use crate::server::{
    decisions::{DecisionBackend, DecisionState, request::DecisionRequest, response::DecisionResponse},
    error::{OaiError, OaiErrorResponse, strip_backend_prefixes},
    request_info::RequestInfo,
    runner::load_decision_backend,
};

type DecisionResult = Result<Json<DecisionResponse>, status::Custom<Json<OaiErrorResponse>>>;

// curl -s localhost:8000/v1/decisions \
//     -H 'content-type: application/json' \
//     -d '{
//       "state": "The order arrived damaged yesterday. The customer has a receipt and asks for a replacement today.",
//       "questions": {
//         "route": {
//           "type": "choice",
//           "instructions": "Which team should handle this request?",
//           "criteria": {
//             "returns": "Refunds, replacements and damaged deliveries",
//             "billing": "Payments, invoices and charges",
//             "technical": "Product setup and faults"
//           }
//         },
//         "receipt": {
//           "type": "noul",
//           "instructions": "Does the customer have a receipt?"
//         },
//         "urgency": {
//           "type": "score",
//           "instructions": "How urgent is this request?",
//           "criteria": ["Routine", "Soon", "Today"]
//         }
//       }
//     }'

// vLLM Semantic Router serves System One at both paths.
#[post("/systemone", format = "json", data = "<body>")]
#[tracing::instrument(skip_all, parent = &request_info.span)]
pub async fn handle_system_one(
    body: Data<'_>,
    decision: &State<DecisionState>,
    request_info: &RequestInfo,
) -> DecisionResult {
    decide(body, decision, request_info).await
}

#[post("/decisions", format = "json", data = "<body>")]
#[tracing::instrument(skip_all, parent = &request_info.span)]
pub async fn handle_decisions(
    body: Data<'_>,
    decision: &State<DecisionState>,
    request_info: &RequestInfo,
) -> DecisionResult {
    decide(body, decision, request_info).await
}

async fn decide(
    body: Data<'_>,
    decision: &DecisionState,
    request_info: &RequestInfo,
) -> DecisionResult {
    let body = match body.open(ByteUnit::Mebibyte(64)).into_string().await {
        Ok(body) if body.is_complete() => body.into_inner(),
        Ok(_) => {
            return Err(error_response(
                Status::BadRequest,
                Some("body"),
                "request_too_large",
                "request body exceeds 64 MiB",
            ));
        },
        Err(error) => {
            return Err(error_response(
                Status::BadRequest,
                Some("body"),
                "invalid_body",
                &format!("failed to read request body: {error}"),
            ));
        },
    };

    let log_body = serde_json::from_str::<serde_json::Value>(&body)
        .map(|value| value.to_string())
        .unwrap_or_else(|_| body.clone());
    tracing::debug!("--> {} {} body={}", request_info.method, request_info.uri, log_body);

    let request = match serde_json::from_str::<DecisionRequest>(&body) {
        Ok(request) => request,
        Err(error) => {
            tracing::info!("rejected: {error}");
            return Err(error_response(
                Status::BadRequest,
                Some("body"),
                "invalid_request",
                &format!("failed to parse System One request: {error}"),
            ));
        },
    };

    if request.questions.is_empty() {
        tracing::info!("rejected: no questions");
        return Err(error_response(
            Status::BadRequest,
            Some("questions"),
            "invalid_request",
            "questions must map one or more question ids to questions",
        ));
    }

    let (model_name, backend) = {
        let guard = decision.backend.read().await;
        match guard.as_ref() {
            Some((name, backend)) => (name.clone(), backend.clone()),
            None => {
                return Err(error_response(
                    Status::Conflict,
                    None,
                    "model_not_loaded",
                    "no decision model loaded; call POST /v1/models/load first",
                ));
            },
        }
    };
    match &backend {
        DecisionBackend::D1Omni(pipeline) => {
            let (d1_state, questions, media, opts) = match request.into_d1_input() {
                Ok(parsed) => parsed,
                Err(error) => {
                    tracing::info!("rejected: `{}`: {}", error.id, error.message);
                    return Err(error_response(
                        Status::BadRequest,
                        Some(&error.id),
                        "invalid_question",
                        &error.message,
                    ));
                },
            };
            let q_count = questions.len();
            tracing::info!("received d1-omni: {q_count} questions, {} media", media.len());
            let started = Instant::now();
            let pipeline = Arc::clone(pipeline);
            let result = tokio::spawn(
                async move {
                    let guard = pipeline.lock().await;
                    tokio::task::block_in_place(|| {
                        guard.system_one(d1_state.as_ref(), &questions, &media, &opts)
                    })
                }
                .in_current_span(),
            )
            .await;
            let elapsed = started.elapsed().as_secs_f64();
            match result {
                Ok(Ok(output)) => {
                    let response = DecisionResponse::from_d1(model_name, output);
                    tracing::info!(
                        "completed d1-omni in {elapsed:.2}s, {q_count} questions, {} tok",
                        response.usage.input_tokens
                    );
                    Ok(Json(response))
                },
                Ok(Err(error)) => {
                    tracing::info!("failed d1-omni in {elapsed:.2}s: {error}");
                    Err(error_response(Status::BadRequest, None, "d1_error", &error.to_string()))
                },
                Err(error) => {
                    tracing::info!("failed d1-omni in {elapsed:.2}s: {error}");
                    Err(error_response(
                        Status::InternalServerError,
                        None,
                        "backend_error",
                        &error.to_string(),
                    ))
                },
            }
        },
        DecisionBackend::Legacy(session) => {
            let (ids, input) = match request.into_input() {
                Ok(input) => input,
                Err(error) => {
                    tracing::info!("rejected: question `{}`: {}", error.id, error.message);
                    return Err(error_response(
                        Status::BadRequest,
                        Some(&format!("questions.{}", error.id)),
                        "invalid_question",
                        &error.message,
                    ));
                },
            };

            tracing::info!("received: {} questions", ids.len());

            let started = Instant::now();
            let questions = input.questions.clone();
            let session = Arc::clone(session);
            // The lock is held by a task of its own, so a dropped request cannot release it mid-decision.
            let result = tokio::spawn(
                async move {
                    let session = session.lock().await;
                    session.decide(input).await
                }
                .in_current_span(),
            )
            .await;
            let elapsed = started.elapsed().as_secs_f64();

            let error = match result {
                Ok(Ok(output)) => {
                    let response = DecisionResponse::new(model_name, ids, &questions, output);
                    tracing::info!(
                        "completed in {elapsed:.2}s, {} questions, prompts {} tok",
                        questions.len(),
                        response.usage.input_tokens
                    );
                    return Ok(Json(response));
                },
                Ok(Err(DecisionSessionError::InvalidQuestion {
                    index,
                    message,
                })) => {
                    let id = &ids[index as usize];
                    tracing::info!("failed in {elapsed:.2}s: question `{id}`: {message}");
                    return Err(error_response(
                        Status::BadRequest,
                        Some(&format!("questions.{id}")),
                        "invalid_question",
                        &message,
                    ));
                },
                Ok(Err(error)) => error.to_string(),
                Err(error) => error.to_string(),
            };

            tracing::info!("failed in {elapsed:.2}s: {error}");

            Err(error_response(Status::InternalServerError, None, "backend_error", &error))
        },
    }
}

#[derive(serde::Deserialize)]
pub struct DecisionModelLoadRequest {
    pub model: String,
}

#[derive(serde::Serialize)]
pub struct DecisionModelAdminResponse {
    pub status: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
}

#[post("/models/load", format = "json", data = "<body>")]
#[tracing::instrument(skip_all, parent = &request_info.span)]
pub async fn handle_decision_model_load(
    body: Json<DecisionModelLoadRequest>,
    decision: &State<DecisionState>,
    request_info: &RequestInfo,
) -> Result<Json<DecisionModelAdminResponse>, status::Custom<Json<OaiErrorResponse>>> {
    let target = body.model.trim().to_string();
    if target.is_empty() {
        return Err(error_response(
            Status::BadRequest,
            Some("model"),
            "invalid_request",
            "model must not be empty",
        ));
    }
    let _gate = decision.load_gate.lock().await;
    match load_decision_backend(&decision.engine, &target).await {
        Ok(Some((name, backend))) => {
            *decision.backend.write().await = Some((name.clone(), backend));
            Ok(Json(DecisionModelAdminResponse {
                status: "ready".to_string(),
                model: Some(name),
                detail: None,
            }))
        },
        Ok(None) => Err(error_response(
            Status::BadRequest,
            Some("model"),
            "unsupported_model",
            &format!("'{target}' is not a Decision / d1-Omni model"),
        )),
        Err(error) => Err(error_response(
            Status::InternalServerError,
            Some("model"),
            "load_failed",
            &format!("{error:#}"),
        )),
    }
}

#[post("/models/unload")]
#[tracing::instrument(skip_all, parent = &request_info.span)]
pub async fn handle_decision_model_unload(
    decision: &State<DecisionState>,
    request_info: &RequestInfo,
) -> Json<DecisionModelAdminResponse> {
    let _gate = decision.load_gate.lock().await;
    *decision.backend.write().await = None;
    Json(DecisionModelAdminResponse {
        status: "idle".to_string(),
        model: None,
        detail: None,
    })
}

fn error_response(
    status: Status,
    param: Option<&str>,
    code: &str,
    message: &str,
) -> status::Custom<Json<OaiErrorResponse>> {
    let kind = if status.code < 500 {
        "invalid_request_error"
    } else {
        "server_error"
    };
    status::Custom(
        status,
        Json(OaiErrorResponse {
            error: OaiError {
                message: strip_backend_prefixes(message).to_string(),
                kind: kind.to_string(),
                param: param.map(str::to_string),
                code: Some(code.to_string()),
            },
        }),
    )
}
