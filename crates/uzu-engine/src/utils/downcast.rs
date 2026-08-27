use std::any::{Any, TypeId};

#[inline]
pub fn downcast_ref<U: Any>(value: &(impl Any + ?Sized)) -> Option<&U> {
    if Any::type_id(value) == TypeId::of::<U>() {
        Some(unsafe { &*std::ptr::from_ref(value).cast::<U>() })
    } else {
        None
    }
}
