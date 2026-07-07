/// Lazily built once via `&mut` (used by the single-read `Reading` markers).
pub(super) struct Deferred<T> {
    slot: Option<T>,
    init: fn() -> T,
}

impl<T> Deferred<T> {
    pub(super) const fn new(init: fn() -> T) -> Self {
        Self {
            slot: None,
            init,
        }
    }

    pub(super) fn get(&mut self) -> &mut T {
        let init = self.init;
        self.slot.get_or_insert_with(init)
    }
}
