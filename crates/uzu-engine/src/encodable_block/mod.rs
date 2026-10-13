//! Encodable blocks follow the argument order used by [`mixer`]:
//! - Constructors: dimensions, data types, configuration and other construction inputs,
//!   parameter trees, then the backend context.
//! - Encode methods: operands, dimensions and operation options, explicit state,
//!   then the command buffer. Method receivers always come first.

pub mod batch_topology;
pub mod classifier;
pub mod convolution;
pub mod decoder;
pub mod dflash;
pub mod embedding;
pub mod embedding_table;
pub mod linear;
pub mod mixer;
pub mod mlp;
pub mod normalization;
pub mod per_layer_embedding;
pub mod prediction_head;
pub mod sampling;
pub mod transformer;
pub mod transformer_layer;
pub mod weaver;
pub mod weaver_layer;
pub mod weight_matrix;
