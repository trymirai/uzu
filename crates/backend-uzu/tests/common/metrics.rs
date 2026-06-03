use std::{thread::sleep, time::Duration};

const COOL_GPU_TEMP: f32 = 60.0;

fn get_gpu_temp() -> f32 {
    #[cfg(target_os = "macos")]
    {
        use macmon::Sampler;
        let mut sample = Sampler::new().unwrap();
        let metrics = sample.get_metrics(0).unwrap();
        metrics.temp.gpu_temp_avg
    }

    #[cfg(not(target_os = "macos"))]
    0.0f32
}

pub fn wait_gpu_cooldown() {
    let mut gpu_temp = get_gpu_temp();
    if gpu_temp <= COOL_GPU_TEMP {
        return;
    }

    println!("Waiting GPU for cooldown...");
    while gpu_temp > COOL_GPU_TEMP {
        sleep(Duration::from_millis(10));
        gpu_temp = get_gpu_temp();
    }
}
