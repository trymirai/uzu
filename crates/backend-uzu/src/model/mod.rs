//! Storage-parameterised model types. Each primitive (`LinearWeights<S>`,
//! eventually `Linear<S>`, `Norm<S>`, `Conv1d<S>`, `Embedding<S>`,
//! `Attention<S>`, `Block<S>`, `Model<S>`) shares one variant set across
//! parsing (`Schema`), loading (`Owned<B>`), and encoding (`Borrowed<'a, B>`)
//! via the `Storage` trait from `backends::common::storage`.

mod linear;

pub use linear::LinearWeights;
