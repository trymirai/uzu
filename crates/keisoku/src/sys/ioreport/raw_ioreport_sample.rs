use objc2_core_foundation::{CFDictionary, CFRetained};

pub(crate) struct RawIOReportSample(pub(super) CFRetained<CFDictionary>);
