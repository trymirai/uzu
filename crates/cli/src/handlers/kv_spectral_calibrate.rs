use std::{fs, path::Path};

use benchmarks::types::Task;
use uzu::{
    KvSpectralCalibration, average_group_variances, dims_for_cumulative_variance,
    session::{
        ChatSession,
        config::{DecodingConfig, RunConfig},
        parameter::{SamplingMethod, SamplingPolicy},
        types::Input,
    },
};

fn build_run_config(task: &Task) -> RunConfig {
    let mut run_config = RunConfig::default().tokens_limit(task.tokens_limit.max(1));
    if task.greedy {
        run_config = run_config.sampling_policy(SamplingPolicy::Custom {
            value: SamplingMethod::Greedy,
        });
    }
    run_config
}

pub fn handle_kv_spectral_calibrate(
    model_path: String,
    task_path: String,
    output_path: String,
) -> Result<(), Box<dyn std::error::Error>> {
    let task: Task = serde_json::from_str(&fs::read_to_string(task_path)?)?;
    let input = Input::Messages(task.messages.clone());
    let mut session = ChatSession::new(model_path.into(), DecodingConfig::default())?;
    let (output, snapshot) = session.run_capture_prefill_kv_debug(input, build_run_config(&task))?;
    let calibration = KvSpectralCalibration::from_snapshot(&snapshot);
    calibration.save(Path::new(&output_path));

    println!("Prompt tokens: {}", output.stats.total_stats.tokens_count_input);
    println!("Saved spectral calibration to {}", output_path);

    for layer in &calibration.layers {
        let key_average = average_group_variances(&layer.key_variances, layer.num_groups, layer.head_dim);
        let value_average = average_group_variances(&layer.value_variances, layer.num_groups, layer.head_dim);
        let key_rank95 = dims_for_cumulative_variance(&key_average, 0.95);
        let value_rank95 = dims_for_cumulative_variance(&value_average, 0.95);
        println!("Layer {}: key_rank95={} value_rank95={}", layer.layer_index, key_rank95, value_rank95);
    }

    Ok(())
}
