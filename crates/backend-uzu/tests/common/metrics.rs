use macmon::Sampler;

const COOL_GPU_TEMP: f32 = 60.0;

fn get_gpu_temp() -> Result<f32, Box<dyn std::error::Error>> {
    #[cfg(metal_backend)]
    {
        let mut sampler = Sampler::new()?;
        let metrics = sampler.get_metrics(100)?;
        Ok(metrics.temp.gpu_temp_avg)
    }

    #[cfg(not(metal_backend))]
    Err(Box::new(std::io::Error::new("Not implemented yet")))
}

pub fn wait_gpu_cool_down() {
    let mut temp = get_gpu_temp().unwrap();
    if temp <= COOL_GPU_TEMP {
        return;
    }

    println!("Waiting for GPU cool down...");
    while temp > COOL_GPU_TEMP {
        std::thread::sleep(std::time::Duration::from_secs(1));
        temp = get_gpu_temp().unwrap();
    }
}
