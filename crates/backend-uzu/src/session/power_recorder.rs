use crate::session::types::PowerStats;

// Background power sampler for one generation: drives keisoku's recorder on Apple, no-op elsewhere.
pub(crate) struct PowerRecorder {
    #[cfg(target_vendor = "apple")]
    handle: keisoku::RecorderHandle,
}

impl PowerRecorder {
    pub(crate) fn start() -> Self {
        #[cfg(target_vendor = "apple")]
        {
            Self {
                handle: keisoku::start(keisoku::Config {
                    interval: std::time::Duration::from_millis(100),
                }),
            }
        }

        #[cfg(not(target_vendor = "apple"))]
        {
            Self {}
        }
    }

    pub(crate) fn stop(
        self,
        duration: f64,
    ) -> Option<PowerStats> {
        #[cfg(target_vendor = "apple")]
        {
            PowerStats::from_keisoku_session(&self.handle.stop(), duration)
        }

        #[cfg(not(target_vendor = "apple"))]
        {
            let _ = duration;
            None
        }
    }
}
