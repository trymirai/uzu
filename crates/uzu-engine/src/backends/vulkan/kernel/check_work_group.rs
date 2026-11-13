use crate::backends::vulkan::{VkKernelError, VkPhysicalDevice};

pub fn check_work_group(
    physical_device: &VkPhysicalDevice,
    kernel: &'static str,
    size: [u32; 3],
    subgroup_size: Option<u32>,
) -> Result<(), VkKernelError> {
    let limits = &physical_device.properties.limits;
    let work_group_error = || VkKernelError::WorkGroupSize {
        size,
        limit: limits.max_compute_work_group_size,
        invocations: limits.max_compute_work_group_invocations,
    };
    if size.contains(&0) || size.iter().zip(limits.max_compute_work_group_size).any(|(&size, limit)| size > limit) {
        return Err(work_group_error());
    }
    let invocations = size.iter().product::<u32>();
    if invocations > limits.max_compute_work_group_invocations {
        return Err(work_group_error());
    }
    let subgroups = physical_device.subgroup_properties.max_compute_workgroup_subgroups;
    if let Some(subgroup_size) = subgroup_size
        && (!size[0].is_multiple_of(subgroup_size) || invocations > subgroup_size * subgroups)
    {
        return Err(VkKernelError::KernelPrecondition {
            kernel,
            condition: "a workgroup of whole subgroups of its required size, within maxComputeWorkgroupSubgroups",
        });
    }
    Ok(())
}
