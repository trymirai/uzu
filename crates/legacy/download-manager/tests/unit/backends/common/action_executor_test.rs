use std::os::unix::fs::symlink;

use uuid::Uuid;

use super::*;
use crate::{lock_manager::DestinationLockLease, reducer::Action};

#[tokio::test]
async fn cleanup_action_rejects_a_late_symlinked_parent() {
    let temporary = tempfile::tempdir().unwrap();
    let outside = temporary.path().join("outside");
    std::fs::create_dir(&outside).unwrap();
    let outside_file = outside.join("resume.part");
    std::fs::write(&outside_file, b"unrelated").unwrap();

    let owned_alias = temporary.path().join("owned");
    symlink(&outside, &owned_alias).unwrap();
    let plan = ActionPlan::from_ordered_actions([Action::DeleteResumeArtifact {
        path: owned_alias.join("resume.part"),
    }]);
    let destination = temporary.path().join("model.bin");
    let lease =
        DestinationLockLease::acquire_for_destination(&destination, "test-manager", Uuid::new_v4()).await.unwrap();

    assert!(apply_actions(&plan, &lease).await.is_err());
    assert_eq!(std::fs::read(&outside_file).unwrap(), b"unrelated");

    lease.release().await.unwrap();
}

#[tokio::test]
async fn integrity_receipt_action_rejects_a_symlinked_file() {
    let temporary = tempfile::tempdir().unwrap();
    let destination = temporary.path().join("model.bin");
    std::fs::write(&destination, b"model").unwrap();
    let outside_receipt = temporary.path().join("outside.json");
    std::fs::write(&outside_receipt, b"unrelated").unwrap();
    let receipt_path = temporary.path().join("integrity.json");
    symlink(&outside_receipt, &receipt_path).unwrap();
    let plan = ActionPlan::from_ordered_actions([Action::SaveIntegrityCache {
        destination: destination.clone(),
        receipt_path,
        file_check: crate::FileCheck::CRC("AAAAAA==".to_string()),
    }]);
    let lease =
        DestinationLockLease::acquire_for_destination(&destination, "test-manager", Uuid::new_v4()).await.unwrap();

    assert!(apply_actions(&plan, &lease).await.is_err());
    assert_eq!(std::fs::read(outside_receipt).unwrap(), b"unrelated");

    lease.release().await.unwrap();
}
