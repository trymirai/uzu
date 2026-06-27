//! Routers screen: download/select a classification ("router") model and
//! classify input text in a playground. [`vm`] holds the row view-model;
//! [`view`] holds `RoutersView`.

mod view;
mod vm;

pub use view::RoutersView;
