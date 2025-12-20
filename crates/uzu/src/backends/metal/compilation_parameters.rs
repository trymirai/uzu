use mpsgraph::{
    CompilationDescriptor, ComputeDevice, Optimization, OptimizationProfile,
};
use objc2::rc::Retained;

#[derive(Debug, Clone, Copy)]
pub enum BlockDevice {
    Ane,
    Gpu,
}

pub fn make_compilation_descriptor(
    device: BlockDevice,
    optimization_level: Optimization,
    optimization_profile: OptimizationProfile,
    preform_placement_analysis: bool,
) -> Retained<CompilationDescriptor> {
    let desc = CompilationDescriptor::new();
    desc.set_optimization_level(optimization_level);
    desc.set_optimization_profile(optimization_profile);
    desc.set_print_ane_placement_analysis(preform_placement_analysis);
    match device {
        BlockDevice::Ane => {
            desc.set_preferred_device(ComputeDevice::ANE);
            desc.set_allowed_compute_devices(
                ComputeDevice::ANE | ComputeDevice::GPU,
            );
        },
        BlockDevice::Gpu => {
            desc.set_preferred_device(ComputeDevice::GPU);
            desc.set_allowed_compute_devices(ComputeDevice::GPU);
        },
    }
    desc
}

fn preferred_block_device() -> BlockDevice {
    match std::env::var("UZU_MPSGRAPH_DEVICE")
        .unwrap_or_default()
        .trim()
        .to_ascii_lowercase()
        .as_str()
    {
        "ane" => BlockDevice::Ane,
        "gpu" => BlockDevice::Gpu,
        _ => BlockDevice::Gpu,
    }
}

pub struct CompilationConfig {
    pub descriptor_general: Retained<CompilationDescriptor>,
    pub descriptor_mlp: Retained<CompilationDescriptor>,
}

impl CompilationConfig {
    fn new(
        descriptor_general: Retained<CompilationDescriptor>,
        descriptor_mlp: Retained<CompilationDescriptor>,
    ) -> Self {
        Self {
            descriptor_general,
            descriptor_mlp,
        }
    }

    pub fn default() -> Self {
        let optimization_level = Optimization::Level1;
        let optimization_profile = OptimizationProfile::Performance;
        let perform_placement_analysis = false;
        let preferred_device = preferred_block_device();

        Self::new(
            make_compilation_descriptor(
                preferred_device,
                optimization_level,
                optimization_profile,
                perform_placement_analysis,
            ),
            make_compilation_descriptor(
                preferred_device,
                optimization_level,
                optimization_profile,
                perform_placement_analysis,
            ),
        )
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use super::{BlockDevice, preferred_block_device};

    static ENV_MUTEX: Mutex<()> = Mutex::new(());

    #[test]
    fn defaults_to_gpu_without_env() {
        let _guard = ENV_MUTEX.lock().unwrap();
        unsafe {
            std::env::remove_var("UZU_MPSGRAPH_DEVICE");
        }
        assert!(matches!(preferred_block_device(), BlockDevice::Gpu));
    }

    #[test]
    fn honors_ane_preference() {
        let _guard = ENV_MUTEX.lock().unwrap();
        unsafe {
            std::env::set_var("UZU_MPSGRAPH_DEVICE", "ane");
        }
        assert!(matches!(preferred_block_device(), BlockDevice::Ane));
        unsafe {
            std::env::remove_var("UZU_MPSGRAPH_DEVICE");
        }
    }
}
