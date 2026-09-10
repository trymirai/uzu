use serde_json::json;

use super::*;

fn task() -> BenchTask {
    serde_json::from_value(json!({
        "identifier": "smoke", "repo_id": "local", "number_of_runs": 1,
        "tokens_limit": 4096, "greedy": false,
        "messages": [{"role": "user", "content": "Hello"}]
    }))
    .unwrap()
}

#[test]
fn bench_forward_counts_preserve_zero_and_missing_data() {
    assert_eq!(sum_forward_passes([Some(2), Some(5)]), Some(7));
    assert_eq!(sum_forward_passes([Some(0)]), Some(0));
    assert_eq!(sum_forward_passes([Some(u32::MAX), Some(1)]), Some(u64::from(u32::MAX) + 1));
    assert_eq!(sum_forward_passes([Some(2), None]), None);
    assert_eq!(sum_forward_passes([]), None);
}

#[test]
fn bench_result_json_saves_decode_count() {
    let mut encoded = json!({
        "task": task(), "device": {"os_name": null, "cpu_name": null, "memory_total": 0},
        "engine_version": "test", "timestamp": 0, "data_type": "bfloat16",
        "memory_used": null, "tokens_count_input": 10, "tokens_count_output": 21,
        "time_to_first_token": 0.1, "prompt_tokens_per_second": 100.0,
        "generate_tokens_per_second": null, "input_energy": null, "output_energy": null,
        "joules_per_token": null, "text": "test"
    });
    encoded["data_type"] = serde_json::to_value(DataType::BF16).unwrap();
    let mut result: BenchResult = serde_json::from_value(encoded).unwrap();
    assert_eq!(result.num_decode_forward_passes, None);
    result.num_decode_forward_passes = Some(5);
    let encoded = serde_json::to_value(result).unwrap();
    assert_eq!(encoded["num_decode_forward_passes"], 5);
    assert!(encoded.get("num_prefill_forward_passes").is_none());
}
