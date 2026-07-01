use std::{fs, path::PathBuf, time::Duration};

use proc_macros::uzu_test;

use super::{blocks::rms_norm, measurement::Measurement};
use crate::{backends::metal::Metal, tests::helpers::create_context};

const MEASUREMENT_WINDOW: Duration = Duration::from_secs(2);

#[uzu_test]
#[ignore = "GPU profiling; run via `cargo profile`"]
fn profile_blocks() {
    let context = create_context::<Metal>();

    let mut lines = vec![format!("{},{}", rms_norm::Parameters::csv_header(), Measurement::csv_header())];
    for parameters in rms_norm::parameter_grid() {
        let kernel = rms_norm::build(&context, &parameters);
        let mut buffers = rms_norm::make_buffers(&context, &parameters);
        let measurement = Measurement::measure(&context, MEASUREMENT_WINDOW, |encoder| {
            rms_norm::encode(&kernel, &mut buffers, &parameters, encoder);
        });
        eprintln!("{parameters:?}: {} samples", measurement.power_samples.len());
        lines.extend(measurement.csv_rows(&parameters.csv_fields()));
    }

    let output_path = output_directory().join(format!("{}.csv", rms_norm::NAME));
    fs::write(&output_path, lines.join("\n")).unwrap();
    eprintln!("wrote {} rows to {}", lines.len() - 1, output_path.display());
}

fn output_directory() -> PathBuf {
    std::env::var_os("UZU_PROFILE_OUTPUT_DIRECTORY").map(PathBuf::from).unwrap_or_else(std::env::temp_dir)
}
