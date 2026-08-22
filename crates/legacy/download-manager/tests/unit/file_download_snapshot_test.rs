use super::FileDownloadSnapshot;
use crate::FileDownloadState;

#[test]
fn explicit_unknown_total_is_not_inferred_from_downloaded_bytes() {
    let snapshot = FileDownloadSnapshot::with_total_bytes(FileDownloadState::downloading(7, 0), None, None);

    assert_eq!(snapshot.total_bytes, None);
}
