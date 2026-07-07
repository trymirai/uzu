use objc2_core_foundation::{CFDictionary, CFRetained};

pub(crate) struct RawEnergySample(pub(super) CFRetained<CFDictionary>);
