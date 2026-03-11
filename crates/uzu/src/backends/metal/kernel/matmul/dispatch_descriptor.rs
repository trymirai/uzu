use super::gemm_mpp;
use crate::{
    DataType,
    backends::{
        common::kernel::matmul::{MatmulArguments, MatmulDispatchDescriptor, MatmulError, gemv},
        metal::{
            Metal,
            context::{DeviceClass, DeviceGeneration, MetalContext},
        },
    },
};

fn default_gemv_max_batch(
    device_generation: DeviceGeneration,
    device_class: DeviceClass,
) -> i32 {
    match (device_generation, device_class) {
        (DeviceGeneration::Gen18, DeviceClass::Max | DeviceClass::Ultra) => 4,
        (DeviceGeneration::Gen18, _) => 4,
        (_, DeviceClass::Phone) => 4,
        (DeviceGeneration::Gen17, DeviceClass::Max | DeviceClass::Ultra) => 8,
        (DeviceGeneration::Gen17, _) => 16,
        (DeviceGeneration::Gen15 | DeviceGeneration::Gen16, DeviceClass::Max | DeviceClass::Ultra) => 16,
        (DeviceGeneration::Gen15 | DeviceGeneration::Gen16, _) => 32,
        (DeviceGeneration::Gen14, DeviceClass::Max | DeviceClass::Ultra) => 32,
        (DeviceGeneration::Gen14, _) => 64,
        (DeviceGeneration::Gen13, DeviceClass::Max | DeviceClass::Ultra) => 16,
        (DeviceGeneration::Gen13, _) => 32,
        _ => 16,
    }
}

fn gemv_max_batch(context: &MetalContext) -> i32 {
    if let Ok(val) = std::env::var("UZU_GEMV_MAX_BATCH") {
        if let Ok(n) = val.parse() {
            return n;
        }
    }

    default_gemv_max_batch(context.device_generation(), context.device_class())
}

pub fn choose_dispatch_descriptor(
    context: &MetalContext,
    data_type: DataType,
    arguments: &MatmulArguments<Metal>,
) -> Result<MatmulDispatchDescriptor, MatmulError<Metal>> {
    let gemv_max_batch = gemv_max_batch(context);

    if let Some(descriptor) = gemv::DispatchDescriptor::try_new::<Metal>(data_type, arguments, gemv_max_batch)? {
        return Ok(MatmulDispatchDescriptor::Gemv(descriptor));
    }

    Ok(MatmulDispatchDescriptor::GemmMpp(gemm_mpp::DispatchDescriptor::new(data_type, arguments)?))
}

#[cfg(test)]
mod tests {
    use super::default_gemv_max_batch;
    use crate::backends::metal::context::{DeviceClass, DeviceGeneration};

    #[test]
    fn returns_measured_and_predicted_gemv_cutoffs() {
        let cases = [
            ((DeviceGeneration::Gen18, DeviceClass::Ultra), 4),
            ((DeviceGeneration::Gen18, DeviceClass::Base), 4),
            ((DeviceGeneration::Unknown(0), DeviceClass::Phone), 4),
            ((DeviceGeneration::Gen17, DeviceClass::Max), 8),
            ((DeviceGeneration::Gen17, DeviceClass::Base), 16),
            ((DeviceGeneration::Gen16, DeviceClass::Max), 16),
            ((DeviceGeneration::Gen15, DeviceClass::Base), 32),
            ((DeviceGeneration::Gen14, DeviceClass::Max), 32),
            ((DeviceGeneration::Gen14, DeviceClass::Base), 64),
            ((DeviceGeneration::Gen13, DeviceClass::Max), 16),
            ((DeviceGeneration::Gen13, DeviceClass::Base), 32),
            ((DeviceGeneration::Unknown(7), DeviceClass::Unknown('x')), 16),
        ];

        for ((device_generation, device_class), expected_gemv_max_batch) in cases {
            assert_eq!(default_gemv_max_batch(device_generation, device_class), expected_gemv_max_batch,);
        }
    }
}
