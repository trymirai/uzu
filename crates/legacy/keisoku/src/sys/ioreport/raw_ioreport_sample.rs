use objc2_core_foundation::{CFDictionary, CFRetained};

pub struct RawIOReportSample(pub CFRetained<CFDictionary>);
