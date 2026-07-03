use gpui::{App, ClickEvent, Window};

pub(crate) type ClickHandler = Box<dyn Fn(&ClickEvent, &mut Window, &mut App) + 'static>;

pub mod button;
pub mod icon;
pub mod icon_button;
pub mod loader;
pub mod markdown;
pub mod modal;
pub mod segmented_control;
pub mod slider;
pub mod text_input;
pub mod toggle;
pub mod vendor_icon;

pub use button::{Button, ButtonKind, ButtonSize};
pub use icon::{Icon, IconEl};
pub use icon_button::IconButton;
pub use loader::Loader;
pub use modal::ConfirmModal;
pub use segmented_control::SegmentedControl;
pub use slider::Slider;
pub use text_input::{InputEvent, TextInput};
pub use toggle::Toggle;
pub use vendor_icon::VendorIcon;
