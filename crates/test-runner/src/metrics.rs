use std::{thread::sleep, time::Duration};

const COOL_GPU_TEMP: f32 = 60.0;

#[cfg(target_vendor = "apple")]
fn get_gpu_temp() -> f32 {
    let (sum, count) = keisoku::thermal_sensors()
        .iter()
        .filter(|sensor| sensor.component == keisoku::Component::Gpu)
        .fold((0.0f32, 0u32), |(sum, count), sensor| (sum + sensor.value as f32, count + 1));
    if count == 0 {
        0.0
    } else {
        sum / count as f32
    }
}

#[cfg(not(target_vendor = "apple"))]
fn get_gpu_temp() -> f32 {
    0.0
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
