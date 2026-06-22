use objc2_core_foundation::CFDictionary;
use serde::Serialize;

use crate::cf::{IoServiceIterator, dictionary_data, registry_properties};

#[derive(Debug, Default, Clone, Serialize)]
pub struct SocInfo {
    pub chip_name: String,
    pub mac_model: String,
    pub memory_gigabytes: u8,
    pub ecpu_cores: u8,
    pub pcpu_cores: u8,
    pub ecpu_label: String,
    pub pcpu_label: String,
    pub gpu_cores: u8,
    pub ecpu_frequencies: Vec<u32>,
    pub pcpu_frequencies: Vec<u32>,
    pub gpu_frequencies: Vec<u32>,
}

fn dvfs_frequencies(
    dictionary: &CFDictionary,
    key: &str,
) -> Option<Vec<u32>> {
    let bytes = dictionary_data(dictionary, key)?;
    if bytes.len() < 8 {
        return None;
    }
    let frequencies =
        bytes.chunks_exact(8).map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]])).collect();
    Some(frequencies)
}

fn to_megahertz(
    frequencies: Vec<u32>,
    scale: u32,
) -> Vec<u32> {
    frequencies.iter().map(|frequency| frequency / scale).collect()
}

fn cluster_voltage_state_keys(dictionary: &CFDictionary) -> Option<(String, String)> {
    let data = dictionary_data(dictionary, "acc-clusters")?;
    if data.len() < 8 {
        return None;
    }
    let mut clusters: Vec<(u8, String)> =
        data.chunks_exact(8).map(|cluster| (cluster[1], format!("voltage-states{}-sram", cluster[0]))).collect();
    clusters.sort_by_key(|cluster| cluster.0);
    if clusters.len() < 2 {
        return None;
    }
    let ecpu_key = clusters[clusters.len() - 2].1.clone();
    let pcpu_key = clusters[clusters.len() - 1].1.clone();
    Some((ecpu_key, pcpu_key))
}

fn cpu_frequencies(
    dictionary: &CFDictionary,
    key: &str,
    is_efficiency_cluster: bool,
    scale: u32,
) -> Option<Vec<u32>> {
    if let Some(frequencies) = dvfs_frequencies(dictionary, key) {
        return Some(to_megahertz(frequencies, scale));
    }
    let (ecpu_key, pcpu_key) = cluster_voltage_state_keys(dictionary)?;
    let key = if is_efficiency_cluster {
        ecpu_key
    } else {
        pcpu_key
    };
    Some(to_megahertz(dvfs_frequencies(dictionary, &key)?, scale))
}

fn parse_cpu_cores(processors: &str) -> (u64, u64, bool) {
    let counts: Vec<u64> =
        processors.strip_prefix("proc ").unwrap_or("").split(':').map(|count| count.parse().unwrap_or(0)).collect();
    match counts.len() {
        4 => {
            let (efficiency_cores, media_cores) = (counts[2], counts[3]);
            if media_cores > 0 {
                (media_cores, counts[1], true)
            } else {
                (efficiency_cores, counts[1], false)
            }
        },
        3 => (counts[2], counts[1], false),
        _ => (0, 0, false),
    }
}

fn run_system_profiler() -> Option<serde_json::Value> {
    let output = std::process::Command::new("system_profiler")
        .args(["SPHardwareDataType", "SPDisplaysDataType", "-json"])
        .output()
        .ok()?;
    serde_json::from_slice(&output.stdout).ok()
}

impl SocInfo {
    pub fn new() -> Option<Self> {
        let profiler = run_system_profiler()?;
        let hardware = &profiler["SPHardwareDataType"][0];

        let chip_name = hardware["chip_type"].as_str().unwrap_or("Unknown chip").to_string();
        let mac_model = hardware["machine_model"].as_str().unwrap_or("Unknown model").to_string();
        let memory_gigabytes = hardware["physical_memory"]
            .as_str()
            .and_then(|memory| memory.strip_suffix(" GB"))
            .and_then(|gigabytes| gigabytes.parse::<u64>().ok())
            .unwrap_or(0) as u8;
        let (ecpu_cores, pcpu_cores, has_media_cluster) =
            parse_cpu_cores(hardware["number_processors"].as_str().unwrap_or(""));
        let gpu_cores = profiler["SPDisplaysDataType"][0]["sppci_cores"]
            .as_str()
            .and_then(|cores| cores.parse::<u64>().ok())
            .unwrap_or(0) as u8;

        let before_m4 = chip_name.contains("M1") || chip_name.contains("M2") || chip_name.contains("M3");
        let cpu_scale: u32 = if before_m4 {
            1000 * 1000
        } else {
            1000
        };
        let gpu_scale: u32 = 1000 * 1000;

        let mut soc_info = SocInfo {
            chip_name,
            mac_model,
            memory_gigabytes,
            ecpu_cores: ecpu_cores as u8,
            pcpu_cores: pcpu_cores as u8,
            ecpu_label: if has_media_cluster {
                "P".into()
            } else {
                "E".into()
            },
            pcpu_label: if has_media_cluster {
                "S".into()
            } else {
                "P".into()
            },
            gpu_cores,
            ..Default::default()
        };

        for (entry, name) in IoServiceIterator::new("AppleARMIODevice")? {
            if name != "pmgr" {
                continue;
            }
            if let Some(properties) = registry_properties(entry) {
                if let Some(frequencies) = cpu_frequencies(&properties, "voltage-states1-sram", true, cpu_scale) {
                    soc_info.ecpu_frequencies = frequencies;
                }
                if let Some(frequencies) = cpu_frequencies(&properties, "voltage-states5-sram", false, cpu_scale) {
                    soc_info.pcpu_frequencies = frequencies;
                }
                if let Some(frequencies) = dvfs_frequencies(&properties, "voltage-states9") {
                    soc_info.gpu_frequencies = to_megahertz(frequencies, gpu_scale);
                }
            }
        }

        Some(soc_info)
    }
}
