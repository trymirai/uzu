use std::{path::PathBuf, sync::Arc, time::Duration};

use download_manager::backends::universal::UniversalActiveTask;

#[tokio::test(flavor = "multi_thread")]
async fn drop_aborts_running_task_handles() {
    let sentinel = Arc::new(());
    let sentinel_for_task = Arc::clone(&sentinel);
    let handle = tokio::spawn(async move {
        let _hold = sentinel_for_task;
        std::future::pending::<()>().await;
    });

    let task_handles: Box<[_]> = vec![handle].into_boxed_slice();
    let active_task = UniversalActiveTask::new(task_handles, PathBuf::from("/tmp/uzu-active-task-drop-test"));

    tokio::task::yield_now().await;
    assert_eq!(
        Arc::strong_count(&sentinel),
        2,
        "spawned task should be holding the sentinel before active_task is dropped",
    );

    drop(active_task);

    for _ in 0..100 {
        if Arc::strong_count(&sentinel) == 1 {
            return;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    panic!("dropping UniversalActiveTask did not abort the spawned task within 1s");
}
