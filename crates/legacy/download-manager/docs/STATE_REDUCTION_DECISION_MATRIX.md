# State Reduction & Reconciliation Decision Matrix

This document provides a comprehensive decision matrix for the state reduction and reconciliation system used in file download management and model storage.

## Overview

The system uses a three-stage pipeline for determining state:

1. **Validation** (`reduce_to_checked_file_state`) - Validates file integrity using CRC
2. **Display State** (`reduce_to_file_download_state`) - Determines user-facing state
3. **Reconciliation** (`reconcile_to_internal_state`) - Performs cleanup and determines internal operational state

## File Download Task State Machine

### Stage 1: Validation (reduce_to_checked_file_state)

Determines if a file on disk is valid, invalid, or missing based on CRC validation.

**Inputs:**

- `file_state`: Exists | Missing
- `crc_file_state`: Exists | Missing
- `expected_crc`: Some(crc) | None

**Output:** `CheckedFileState`

| File State | CRC File | Expected CRC | Action                                 | Result                                      |
| ---------- | -------- | ------------ | -------------------------------------- | ------------------------------------------- |
| Missing    | -        | -            | -                                      | **Missing**                                 |
| Exists     | Exists   | Some(crc)    | Read cached CRC, compare with expected | **Valid** if match, recalculate if mismatch |
| Exists     | Missing  | Some(crc)    | Calculate CRC, save cache              | **Valid** if match, **Invalid** otherwise   |
| Exists     | -        | None         | No validation needed                   | **Valid**                                   |

**CRC Caching Optimization:**

- Valid files get a `.crc` cache file to avoid recalculation on next launch
- Makes subsequent launches ~100x faster for large files
- Cache is invalidated when file changes

---

### Stage 2: Display State (reduce_to_file_download_state)

Maps validation results to user-facing download states.

**Inputs:**

- `checked_state`: Valid | Invalid | Missing
- `resume_state`: Exists | Missing
- `task_state`: None | Running | Suspended | Completed | Canceling
- `expected_bytes`: Option<u64>

**Output:** `FileDownloadState` (for UI display)

#### Valid File Cases

| Checked | Resume | Task | Output                           |
| ------- | ------ | ---- | -------------------------------- |
| Valid   | -      | -    | **Downloaded** (file_size bytes) |

*Any valid file is immediately shown as downloaded, regardless of other state.*

---

#### Invalid File Cases

| Checked | Resume  | Task                | Output            | Bytes                      |
| ------- | ------- | ------------------- | ----------------- | -------------------------- |
| Invalid | Missing | None                | **NotDownloaded** | 0 / expected               |
| Invalid | Exists  | None                | **Paused**        | resume_bytes / expected    |
| Invalid | -       | Running             | **Downloading**   | task_bytes / task_expected |
| Invalid | -       | Suspended           | **Paused**        | task_bytes / task_expected |
| Invalid | -       | Completed/Canceling | **NotDownloaded** | 0 / expected               |

*Invalid files with active tasks show progress from task state.*

---

#### Missing File Cases

| Checked | Resume  | Task      | Output            | Bytes                      |
| ------- | ------- | --------- | ----------------- | -------------------------- |
| Missing | Missing | None      | **NotDownloaded** | 0 / expected               |
| Missing | Exists  | None      | **Paused**        | resume_bytes / expected    |
| Missing | -       | Running   | **Downloading**   | task_bytes / task_expected |
| Missing | -       | Suspended | **Paused**        | task_bytes / task_expected |
| Missing | -       | Completed | **NotDownloaded** | 0 / expected               |
| Missing | -       | Canceling | **NotDownloaded** | 0 / expected               |

*Missing files rely on resume data or active tasks for progress info.*

---

### Stage 3: Reconciliation (reconcile_to_internal_state)

Performs cleanup and determines operational state. This is where side effects happen (file deletion, task cancellation).

**Inputs:**

- `checked_file`: Valid | Invalid | Missing
- `resume_file`: Exists | Missing
- `url_session_task`: None | Some(task)
- `expected_bytes`: Option<u64> (for partial download detection)

**Output:** `InternalDownloadState` (for operations)

#### Valid File Reconciliation (Cases 1-10)

| Checked | Resume  | Task | Actions                                     | Internal State |
| ------- | ------- | ---- | ------------------------------------------- | -------------- |
| Valid   | Missing | None | Keep file & CRC                             | **Downloaded** |
| Valid   | Exists  | None | Delete resume, keep file & CRC              | **Downloaded** |
| Valid   | -       | Any  | Cancel task, delete resume, keep file & CRC | **Downloaded** |

*Valid files always result in Downloaded state with cleanup of unnecessary artifacts.*

---

#### Invalid File Reconciliation (Cases 11-20)

| Checked | Resume  | Task                | File Action                           | Resume Action         | Task Action    | Internal State                                |
| ------- | ------- | ------------------- | ------------------------------------- | --------------------- | -------------- | --------------------------------------------- |
| Invalid | Missing | None                | Check size: partial=keep, full=delete | -                     | -              | **NotDownloaded**                             |
| Invalid | Exists  | None                | Delete                                | Keep                  | -              | **Paused**                                    |
| Invalid | -       | Running             | Delete                                | Delete                | Keep           | **Downloading**                               |
| Invalid | Missing | Suspended           | Delete                                | Produce from task     | Cancel+produce | **Paused** if success, else **NotDownloaded** |
| Invalid | Exists  | Suspended           | Delete                                | Compare & keep better | Cancel+produce | **Paused**                                    |
| Invalid | -       | Completed/Canceling | Delete                                | Delete                | Cancel         | **NotDownloaded**                             |

**Special Case - Partial Download Detection (Case 11):**

- If `file_size < expected_bytes`: File is partial, **preserve it** (don't delete)
- If `file_size >= expected_bytes`: File is corrupted, **delete it**
- This prevents deleting incomplete downloads that can be resumed

---

#### Missing File Reconciliation (Cases 21-30)

| Checked | Resume  | Task          | Resume Action         | Task Action    | Internal State                      |
| ------- | ------- | ------------- | --------------------- | -------------- | ----------------------------------- |
| Missing | Missing | None          | -                     | -              | **NotDownloaded**                   |
| Missing | Exists  | None          | Keep                  | -              | **Paused**                          |
| Missing | -       | Running       | Delete                | Keep           | **Downloading**                     |
| Missing | Missing | Suspended     | Produce from task     | Cancel+produce | **Paused** if success               |
| Missing | Exists  | Suspended     | Compare & keep better | Cancel+produce | **Paused**                          |
| Missing | -       | **Completed** | Delete                | **Keep alive** | **Downloading** (awaiting delegate) |
| Missing | -       | Canceling     | Delete                | Cancel         | **NotDownloaded**                   |

**Critical Case - Background Download Completion (Cases 27-28):**

- When task is `Completed` but file is missing: delegate callback is pending
- **Do NOT cancel task** - keep it alive in Downloading state
- This allows URLSession delegate to move the file to final destination
- Common scenario when download completes while app is closed

---

## Model State Reduction

Models aggregate the states of multiple file download tasks into a single model-level state.

**Inputs:** Array of `FileDownloadState` from all files in the model

**Output:** `ModelDownloadState`

### Model State Priority Order

The reducer checks conditions in this priority order:

| Priority | Condition                                      | Model State                   | Bytes                        |
| -------- | ---------------------------------------------- | ----------------------------- | ---------------------------- |
| 1        | All files Downloaded                           | **Downloaded**                | sum(file_bytes)              |
| 2        | Any file Downloading                           | **Downloading**               | sum(downloaded) / sum(total) |
| 3        | Any file Paused                                | **Paused**                    | sum(downloaded) / sum(total) |
| 4        | Any file Error                                 | **Error**                     | First error message          |
| 5        | Some files Downloaded AND downloaded_bytes > 0 | **Paused** (partial progress) | sum(downloaded) / sum(total) |
| 6        | Default (all NotDownloaded)                    | **NotDownloaded**             | 0 / sum(total)               |

**Key Insight - Priority 5 (Partial Progress Detection):**

- If ANY files are downloaded but NOT ALL files are downloaded
- AND total downloaded bytes > 0
- AND nothing is actively downloading or paused
- THEN treat as **Paused** (resumable partial progress)
- This correctly represents scenarios like: config.json ✓ + tokenizer.json ✓ + model.safetensors ✗

---

## Special Case Handling

### Background Download Completion

**Scenario:** App closed while downloading, URLSession completes in background, app relaunched.

**Detection in `initialize_task_cache()`:**

```rust
if task.state() == Completed && checked_file_state == Valid {
    // Delegate callback was missed, manually trigger completion
    file_task.handle_download_completion().await;
}
```

**Actions:**

1. Detect: Completed task + Valid file (CRC may be missing)
2. Trigger: `handle_download_completion()`
3. Result: CRC validated, cached, state transitions to Downloaded

---

### Partial Download Preservation

**Scenario:** Incomplete file exists without resume data (e.g., interrupted download, app crash).

**Detection in Reconciliation Case 11:**

```rust
if file_size < expected_bytes {
    // Partial download - preserve file
    remove_crc_cache();
    return NotDownloaded;
} else {
    // Corrupted full-size file - delete
    delete_file_and_crc();
    return NotDownloaded;
}
```

**Purpose:** Prevent deleting large partial downloads that might be resumable in future.

---

### Resume Data Comparison

**Scenario:** Both file-based resume data and URLSession task resume data exist.

**Resolution in Cases 16, 26:**

1. Read existing resume data bytes: `existing_bytes`
2. Produce resume data from suspended task: `new_bytes`
3. Compare: `if new_bytes > existing_bytes`
4. Keep the resume data with more progress
5. Ensures we never lose download progress

---

## State Transition Rules

### Internal State Transitions

```
NotDownloaded → Downloading  (via download())
Downloading → Paused         (via pause())
Downloading → Downloaded     (on completion)
Paused → Downloading         (via download())
Downloaded → NotDownloaded   (via cancel())
Any → NotDownloaded          (via cancel())
```

### Invalid Transitions

- `Downloaded → Downloading`: Cannot re-download existing file
- `Downloaded → Paused`: Cannot pause completed download

---

## File Lifecycle Examples

### Example 1: Fresh Download

```
Initial:     file=Missing, crc=Missing, resume=Missing, task=None
↓
Checked:     Missing
Display:     NotDownloaded (0 / 1.1GB)
Internal:    NotDownloaded
↓ User clicks Download
Create:      task=Running
Display:     Downloading (0 / 1.1GB)
Internal:    Downloading
↓ Download progresses
Display:     Downloading (500MB / 1.1GB)
↓ Download completes
Delegate:    Moves file, validates CRC, caches CRC
Checked:     Valid
Display:     Downloaded (1.1GB)
Internal:    Downloaded
```

---

### Example 2: Resume After Interruption

```
Initial:     file=Missing, crc=Missing, resume=Exists(500MB), task=None
↓
Checked:     Missing
Display:     Paused (500MB / 1.1GB)
Internal:    Paused
↓ User clicks Resume
Create:      task=Running (with resume data)
Display:     Downloading (500MB / 1.1GB)
Internal:    Downloading
↓ Download progresses from 500MB
Display:     Downloading (800MB / 1.1GB)
```

---

### Example 3: Background Completion

```
App Closed:  task=Running, downloading...
↓
URLSession:  Completes download, moves file (no delegate callback)
↓
Relaunch:    file=Exists, crc=Missing, task=Completed
↓
Checked:     Valid (calculates CRC)
Detected:    Completed task + Valid file
Trigger:     handle_download_completion()
Result:      Validates CRC, caches CRC
Display:     Downloaded (1.1GB)
Internal:    Downloaded
Cleanup:     Cancel completed task
```

---

### Example 4: Partial File Preservation

```
Initial:     file=Exists(100MB), crc=Missing, resume=Missing, task=None
             expected=1.1GB
↓
Checked:     Invalid (file incomplete, fails CRC)
Reconcile:   Detect: 100MB < 1.1GB (partial)
Action:      PRESERVE file, remove CRC cache
Internal:    NotDownloaded
Display:     NotDownloaded (0 / 1.1GB)
↓ Future download can resume from partial file if URLSession supports it
```

---

### Example 5: Multi-File Model Partial Download

```
Model Files: config.json, tokenizer.json, model.safetensors

File States:
- config.json:     Downloaded (9KB / 9KB)
- tokenizer.json:  Downloaded (11MB / 11MB)
- model.safetensors: NotDownloaded (0 / 1.1GB)

Model Reducer:
- all_downloaded: false
- any_downloading: false
- any_paused: false
- any_downloaded: true
- downloaded_bytes: 11MB
↓
Priority 5 Match: Some downloaded + progress > 0
Result: Paused (11MB / 1.1GB)
```

---

## Summary

The state reduction system provides:

1. **Deterministic State**: Same inputs always produce same outputs
2. **CRC Optimization**: Cache validation results for fast subsequent launches
3. **Partial Progress**: Preserve and resume incomplete downloads
4. **Background Safety**: Handle downloads that complete while app closed
5. **Data Integrity**: Validate files before marking as complete
6. **Progress Preservation**: Never lose download progress through resume data comparison
7. **Clean Semantics**: Clear user-facing states (NotDownloaded, Downloading, Paused, Downloaded, Error)

The system handles edge cases gracefully while maintaining clear separation between validation (pure), display (pure), and reconciliation (side effects).
