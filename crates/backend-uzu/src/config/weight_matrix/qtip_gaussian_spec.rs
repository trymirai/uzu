use monostate::MustBe;
use serde::{Deserialize, Serialize};

use crate::{config::weight_matrix::Layout, utils::strict_serde::DeserializeStrict};

/// QTIP trellis leaf. S packages carry two fields the older physical packages lack: the dtype of the row scales
/// and the axes of the QAT post gains the dense checkpoint folded after rotation (`post_gains.<index>` tensors).
/// Written by hand instead of `uzu_config` so those two fields can be optional.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QtipGaussianSpec {
    #[serde(rename = "type")]
    ty: MustBe!("QtipGaussianSpec"),
    #[serde(deserialize_with = "crate::utils::strict_serde::required")]
    pub layout: Layout,
    #[serde(deserialize_with = "crate::utils::strict_serde::required")]
    pub vector_width: u32,
    #[serde(deserialize_with = "crate::utils::strict_serde::required")]
    pub transition_bits: u32,
    #[serde(deserialize_with = "crate::utils::strict_serde::required")]
    pub restart_columns: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scale_dtype: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub post_gain_axes: Option<Box<[String]>>,
}

impl<'de> DeserializeStrict<'de> for QtipGaussianSpec {}
