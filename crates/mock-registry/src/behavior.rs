bitflags::bitflags! {
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
    pub struct Behavior: u8 {
        const CORRUPT_BODY = 1 << 0;
        const THROTTLED = 1 << 1;
    }
}
