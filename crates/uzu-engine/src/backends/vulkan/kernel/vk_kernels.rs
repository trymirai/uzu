include!(concat!(env!("OUT_DIR"), "/vulkan.rs"));

use crate::backends::vulkan::{VkKernelError, VkPhysicalDevice};

/// Fails unless a workgroup of `size` is within the device's limits and, with a required subgroup size, made of whole
/// subgroups of it within maxComputeWorkgroupSubgroups; the generated kernels check theirs before creating a pipeline.
pub fn check_work_group(
    physical_device: &VkPhysicalDevice,
    kernel: &'static str,
    size: [u32; 3],
    subgroup_size: Option<u32>,
) -> Result<(), VkKernelError> {
    let limits = physical_device.limits();
    let invocations = size.into_iter().try_fold(1u32, u32::checked_mul);
    if size.contains(&0)
        || size.iter().zip(limits.max_compute_work_group_size).any(|(&size, limit)| size > limit)
        || invocations.is_none_or(|count| count > limits.max_compute_work_group_invocations)
    {
        return Err(VkKernelError::WorkGroupSize {
            size,
            limit: limits.max_compute_work_group_size,
            invocations: limits.max_compute_work_group_invocations,
        });
    }
    let subgroups = physical_device.subgroup_properties.max_compute_workgroup_subgroups;
    if let Some(subgroup_size) = subgroup_size
        && (!size[0].is_multiple_of(subgroup_size)
            || invocations.is_none_or(|count| count > subgroup_size.saturating_mul(subgroups)))
    {
        return Err(VkKernelError::KernelPrecondition {
            kernel,
            condition: "a workgroup of whole subgroups of its required size, within maxComputeWorkgroupSubgroups",
        });
    }
    Ok(())
}
