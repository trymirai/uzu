pub(super) enum ClassificationOutcome {
    Ok(Vec<(String, f64)>),
    Err(String),
}
