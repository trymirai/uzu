use bitflags::bitflags;

bitflags! {
    #[repr(transparent)]
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct Capabilities: u32 {
        /// Resume data can retain a redirected CDN URL that may expire before reuse.
        const CACHES_REDIRECTED_URL_IN_RESUME_DATA = 1 << 0;
    }
}
