const COOL_GPU_TEMP: f32 = 60.0;

fn get_gpu_temp() -> f32 {
    #[cfg(metal_backend)]
    {
        use macmon::Sampler;
        let mut sample = Sampler::new().unwrap();
        let metrics = sample.get_metrics(0).unwrap();
        metrics.temp.gpu_temp_avg
    }

    #[cfg(not(metal_backend))]
    unimplemented!()
}

pub fn wait_gpu_cooldown() {
    let mut gpu_temp = get_gpu_temp();
    if gpu_temp <= COOL_GPU_TEMP {
        return;
    }

    println!("Waiting GPU for cooldown...");
    while gpu_temp > COOL_GPU_TEMP {
        std::thread::sleep(std::time::Duration::from_millis(10));
        gpu_temp = get_gpu_temp();
    }
}
