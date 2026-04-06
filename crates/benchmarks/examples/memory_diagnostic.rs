use std::{env, path::PathBuf, process::Command};

use uzu::session::{
    ChatSession,
    config::{DecodingConfig, RunConfig},
    parameter::{ContextLength, SamplingMethod, SamplingPolicy},
    types::{Input, Message, Output},
};

#[cfg(target_os = "macos")]
fn print_metal_device_info() {
    use core::mem::transmute;
    use metal::{MTLDevice, MTLDeviceExt};
    use objc2::{ffi::objc_msgSend, runtime::{NSObjectProtocol, Sel}, sel};

    let device = <dyn MTLDevice>::system_default().expect("No Metal device");

    let name = device.name();
    println!("[Metal] Device name: {name}");

    unsafe fn query_bool(
        device: &objc2::runtime::ProtocolObject<dyn MTLDevice>,
        selector: Sel,
    ) -> Option<bool> {
        if device.respondsToSelector(selector) {
            let send: unsafe extern "C" fn(*const objc2::runtime::AnyObject, Sel) -> bool =
                unsafe { transmute(objc_msgSend as *const ()) };
            Some(unsafe {
                send(
                    device as *const _ as *const objc2::runtime::AnyObject,
                    selector,
                )
            })
        } else {
            None
        }
    }

    unsafe fn query_u32(
        device: &objc2::runtime::ProtocolObject<dyn MTLDevice>,
        selector: Sel,
    ) -> Option<u32> {
        if device.respondsToSelector(selector) {
            let send: unsafe extern "C" fn(*const objc2::runtime::AnyObject, Sel) -> u32 =
                unsafe { transmute(objc_msgSend as *const ()) };
            Some(unsafe {
                send(
                    device as *const _ as *const objc2::runtime::AnyObject,
                    selector,
                )
            })
        } else {
            None
        }
    }

    unsafe fn query_u64(
        device: &objc2::runtime::ProtocolObject<dyn MTLDevice>,
        selector: Sel,
    ) -> Option<u64> {
        if device.respondsToSelector(selector) {
            let send: unsafe extern "C" fn(*const objc2::runtime::AnyObject, Sel) -> u64 =
                unsafe { transmute(objc_msgSend as *const ()) };
            Some(unsafe {
                send(
                    device as *const _ as *const objc2::runtime::AnyObject,
                    selector,
                )
            })
        } else {
            None
        }
    }

    unsafe {
        if let Some(cores) = query_u32(&device, sel!(gpuCoreCount)) {
            println!("[Metal] GPU cores: {cores}");
        }
        if let Some(mem) = query_u64(&device, sel!(sharedMemorySize)) {
            println!("[Metal] Shared memory: {:.1} GB", mem as f64 / 1024.0 / 1024.0 / 1024.0);
        }
        match query_bool(&device, sel!(supportsMXU)) {
            Some(v) => println!("[Metal] Supports MXU: {v}"),
            None => println!("[Metal] Supports MXU: N/A (selector not found)"),
        }
        match query_bool(&device, sel!(supportsSIMDGroupMatrix)) {
            Some(v) => println!("[Metal] Supports SIMD Group Matrix: {v}"),
            None => println!("[Metal] Supports SIMD Group Matrix: N/A"),
        }
        match query_bool(&device, sel!(supportsSIMDGroup)) {
            Some(v) => println!("[Metal] Supports SIMD Group: {v}"),
            None => println!("[Metal] Supports SIMD Group: N/A"),
        }
    }
}

const DEFAULT_REPO_ID: &str = "google/gemma-3-1b-it";

#[cfg(target_os = "macos")]
fn get_rss_bytes() -> u64 {
    use std::mem;

    use mach2::{
        kern_return::KERN_SUCCESS,
        mach_types::task_t,
        message::mach_msg_type_number_t,
        task::task_info,
        task_info::{TASK_BASIC_INFO, task_basic_info},
        traps::mach_task_self,
    };

    unsafe {
        let task: task_t = mach_task_self();
        let mut info = task_basic_info {
            virtual_size: 0,
            resident_size: 0,
            user_time: mem::zeroed(),
            system_time: mem::zeroed(),
            policy: 0,
            suspend_count: 0,
        };
        let mut count: mach_msg_type_number_t =
            (mem::size_of::<task_basic_info>() / mem::size_of::<u32>()) as u32;
        let result =
            task_info(task, TASK_BASIC_INFO, &mut info as *mut _ as *mut i32, &mut count);
        if result == KERN_SUCCESS {
            info.resident_size as u64
        } else {
            0
        }
    }
}

fn mb(bytes: u64) -> f64 {
    bytes as f64 / 1024.0 / 1024.0
}

fn print_rss(label: &str) -> u64 {
    let rss = get_rss_bytes();
    println!("[RSS] {label}: {:.1} MB", mb(rss));
    rss
}

fn make_input() -> Input {
    Input::Messages(vec![Message::user(
        "Briefly explain what a neural network is in two sentences.".to_string(),
    )])
}

/// Download a model via `uv run tools/helpers/main.py download-model <repo_id>`.
/// Returns the local model path: `<workspace>/models/<version>/<model_name>`.
fn download_model(repo_id: &str) -> PathBuf {
    let workspace_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf();
    let helpers_dir = workspace_root.join("tools/helpers");

    println!("Downloading model {repo_id} via uv ...");
    let status = Command::new("uv")
        .args(["run", "--project", helpers_dir.to_str().unwrap(), "python3", helpers_dir.join("main.py").to_str().unwrap(), "download-model", repo_id])
        .current_dir(&workspace_root)
        .stdout(std::process::Stdio::inherit())
        .stderr(std::process::Stdio::inherit())
        .status()
        .expect("Failed to run uv — is it installed?");
    assert!(status.success(), "Model download failed");

    // Model name is the last part of repo_id (e.g. "gemma-3-1b-it")
    let model_name = repo_id.split('/').last().unwrap().to_lowercase();
    let version = uzu::VERSION;
    let model_path = workspace_root.join("models").join(version).join(&model_name);
    assert!(
        model_path.exists(),
        "Expected model path does not exist: {model_path:?}"
    );
    println!("Model path: {model_path:?}");
    model_path
}

fn main() {
    let repo_id = env::args().nth(1).unwrap_or_else(|| DEFAULT_REPO_ID.to_string());
    let num_iterations: usize = env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(3);

    println!("=== Memory Diagnostic ===");
    println!("Repo: {repo_id}");
    println!("Iterations: {num_iterations}");
    println!();

    print_metal_device_info();
    println!();

    let model_path = download_model(&repo_id);

    let rss_baseline = print_rss("baseline (before anything)");

    for iteration in 0..num_iterations {
        println!("\n--- Iteration {}/{num_iterations} ---", iteration + 1);

        let rss_before_load = print_rss("before model load");

        let decoding_config =
            DecodingConfig::default().with_context_length(ContextLength::Custom(2048));
        let mut session = ChatSession::new(model_path.clone(), decoding_config)
            .expect("Failed to create session");

        let rss_after_load = print_rss("after model load");

        // Warmup
        let warmup_config = RunConfig::default().tokens_limit(1);
        session
            .run(make_input(), warmup_config, Some(|_: Output| true))
            .expect("Warmup failed");

        let rss_after_warmup = print_rss("after warmup");

        // Run 5 inference passes
        for pass in 1..=5 {
            let run_config = RunConfig::default()
                .tokens_limit(64)
                .sampling_policy(SamplingPolicy::Custom {
                    value: SamplingMethod::Greedy,
                });
            let output = session
                .run(make_input(), run_config, Some(|_: Output| true))
                .expect("Inference failed");
            let _rss = print_rss(&format!("after inference pass {pass}"));
            println!(
                "  generated {} tokens, output: {:?}",
                output.stats.total_stats.tokens_count_output,
                &output.text.original[..output.text.original.len().min(80)]
            );
        }

        let rss_final = print_rss("before drop");

        drop(session);

        let rss_after_drop = print_rss("after drop");

        println!("\n  Summary (iteration {}):", iteration + 1);
        println!("    Baseline:      {:.1} MB", mb(rss_baseline));
        println!(
            "    Before load:   {:.1} MB (+{:.1} MB from baseline)",
            mb(rss_before_load),
            mb(rss_before_load - rss_baseline)
        );
        println!(
            "    After load:    {:.1} MB (+{:.1} MB from before load)",
            mb(rss_after_load),
            mb(rss_after_load.saturating_sub(rss_before_load))
        );
        println!(
            "    After warmup:  {:.1} MB (+{:.1} MB from after load)",
            mb(rss_after_warmup),
            mb(rss_after_warmup.saturating_sub(rss_after_load))
        );
        println!(
            "    Final:         {:.1} MB (+{:.1} MB from after warmup)",
            mb(rss_final),
            mb(rss_final.saturating_sub(rss_after_warmup))
        );
        println!(
            "    After drop:    {:.1} MB ({:.1} MB freed)",
            mb(rss_after_drop),
            mb(rss_final.saturating_sub(rss_after_drop))
        );
    }

    println!("\n=== Done ===");
}
