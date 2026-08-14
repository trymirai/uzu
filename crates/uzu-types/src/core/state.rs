/// State supplied to an inference operation.
///
/// Models create an initial value with
/// [`InferenceModel::create_empty_state`](crate::core::InferenceModel::create_empty_state)
/// and consume it when [`InferenceModel::infer`](crate::core::InferenceModel::reply) is called.
pub trait InferenceState {}
