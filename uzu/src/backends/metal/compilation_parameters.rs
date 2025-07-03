use mpsgraph::{
    CompilationDescriptor, Optimization, OptimizationProfile,
    device::MPSGraphComputeDevice,
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
            desc.set_preferred_device(MPSGraphComputeDevice::ANE);
            desc.set_allowed_compute_devices(
                MPSGraphComputeDevice::ANE | MPSGraphComputeDevice::GPU,
            );
        },
        BlockDevice::Gpu => {
            desc.set_preferred_device(MPSGraphComputeDevice::GPU);
            desc.set_allowed_compute_devices(MPSGraphComputeDevice::GPU);
        },
    }
    desc
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

        Self::new(
            make_compilation_descriptor(
                BlockDevice::Gpu,
                optimization_level,
                optimization_profile,
                perform_placement_analysis,
            ),
            make_compilation_descriptor(
                BlockDevice::Gpu,
                optimization_level,
                optimization_profile,
                perform_placement_analysis,
            ),
        )
    }
}
