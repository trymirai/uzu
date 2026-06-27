//! Settings screen: an inner sidebar (General / Privacy / About) with a
//! scrollable content panel. [`clear_data`] holds the multi-step clear-data
//! wizard; [`view`] holds `SettingsView`.

mod clear_data;
mod view;

pub use view::SettingsView;
