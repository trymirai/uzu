pub fn calculate_metric(data: &[f64]) -> String {
    let mean = mean(data);
    let standard_deviation = standard_deviation(data);
    if let (Some(mean), Some(standard_deviation)) = (mean, standard_deviation) {
        format!("{mean:.3} ± {standard_deviation:.3}")
    } else if let Some(mean) = mean {
        format!("{mean:.3}")
    } else {
        "-".to_string()
    }
}

pub fn mean(data: &[f64]) -> Option<f64> {
    if data.is_empty() {
        return None;
    }

    Some(data.iter().sum::<f64>() / data.len() as f64)
}

pub fn standard_deviation(data: &[f64]) -> Option<f64> {
    let samples_count = data.len();
    if samples_count < 2 {
        return None;
    }

    let mean = mean(data)?;

    let variance = data
        .iter()
        .map(|value| {
            let difference = value - mean;
            difference * difference
        })
        .sum::<f64>()
        / (samples_count as f64 - 1.0);

    Some(variance.sqrt())
}
