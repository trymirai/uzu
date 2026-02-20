use super::Backend;

pub trait Event {
    type Backend: Backend;
}
