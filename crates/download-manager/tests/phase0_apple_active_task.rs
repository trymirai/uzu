#[cfg(target_vendor = "apple")]
#[allow(dead_code)]
mod apple {
    use std::path::PathBuf;

    use objc2::rc::Retained;
    use objc2_foundation::NSURLSessionDownloadTask;

    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    struct ActiveDownloadGeneration(u64);

    struct AppleActiveTask {
        task: Retained<NSURLSessionDownloadTask>,
    }

    enum InitialDownloadLifecycleState {
        Downloading {
            task: Option<AppleActiveTask>,
            generation: ActiveDownloadGeneration,
        },
        Paused {
            resume_artifact_path: PathBuf,
        },
        NotDownloaded,
    }

    struct DownloadTaskActor {
        initial_state: InitialDownloadLifecycleState,
    }

    impl DownloadTaskActor {
        fn new_with_running_apple_task(
            task: AppleActiveTask,
            generation: ActiveDownloadGeneration,
        ) -> Self {
            Self {
                initial_state: InitialDownloadLifecycleState::Downloading {
                    task: Some(task),
                    generation,
                },
            }
        }
    }

    fn assert_send_sync<T: Send + Sync>() {}

    #[test]
    fn test_phase0_apple_active_task_wrapper_is_send_sync() {
        assert_send_sync::<AppleActiveTask>();
        assert_send_sync::<InitialDownloadLifecycleState>();
        assert_send_sync::<DownloadTaskActor>();
    }

    #[test]
    fn test_phase0_apple_actor_constructor_owns_running_task() {
        let constructor: fn(AppleActiveTask, ActiveDownloadGeneration) -> DownloadTaskActor =
            DownloadTaskActor::new_with_running_apple_task;

        let _ = constructor;
    }
}

#[cfg(not(target_vendor = "apple"))]
#[test]
fn test_phase0_apple_active_task_wrapper_is_apple_only() {
    // The real ownership proof is compiled on Apple targets. This placeholder
    // keeps the integration test target present on other platforms.
}
