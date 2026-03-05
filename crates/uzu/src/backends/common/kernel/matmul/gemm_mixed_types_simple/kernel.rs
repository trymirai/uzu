use std::ops::DerefMut;

use super::{super::matmul_arguments::MatmulArguments, dispatch_descriptor::DispatchDescriptor};
use crate::{
    DataType,
    backends::common::{
        Backend, Kernels,
        kernel::{
            matmul::MatmulError,
            MixedTypesSimpleGemmI8Bf16Bf16Kernel, MixedTypesSimpleGemmI8F16F16Kernel,
            MixedTypesSimpleGemmI8F32F32Kernel, MixedTypesSimpleGemmI8I8I32Kernel,
        },
    },
};

#[derive(Debug, Clone, Copy)]
enum MixedTypesSimpleRoute {
    I8I8I32,
    I8Bf16Bf16,
    I8F16F16,
    I8F32F32,
}

fn route_for_combo(
    a_dtype: DataType,
    b_dtype: DataType,
    output_dtype: DataType,
) -> Option<MixedTypesSimpleRoute> {
    match (a_dtype, b_dtype, output_dtype) {
        (DataType::I8, DataType::I8, DataType::I32) => Some(MixedTypesSimpleRoute::I8I8I32),
        (DataType::I8, DataType::BF16, DataType::BF16) => Some(MixedTypesSimpleRoute::I8Bf16Bf16),
        (DataType::I8, DataType::F16, DataType::F16) => Some(MixedTypesSimpleRoute::I8F16F16),
        (DataType::I8, DataType::F32, DataType::F32) => Some(MixedTypesSimpleRoute::I8F32F32),
        _ => None,
    }
}

pub fn supports_combo(
    a_dtype: DataType,
    b_dtype: DataType,
    output_dtype: DataType,
) -> bool {
    route_for_combo(a_dtype, b_dtype, output_dtype).is_some()
}

pub struct GemmMixedTypesSimpleKernel<B: Backend> {
    route: MixedTypesSimpleRoute,
    i8_i8_i32: Option<<B::Kernels as Kernels>::MixedTypesSimpleGemmI8I8I32Kernel>,
    i8_bf16_bf16: Option<<B::Kernels as Kernels>::MixedTypesSimpleGemmI8Bf16Bf16Kernel>,
    i8_f16_f16: Option<<B::Kernels as Kernels>::MixedTypesSimpleGemmI8F16F16Kernel>,
    i8_f32_f32: Option<<B::Kernels as Kernels>::MixedTypesSimpleGemmI8F32F32Kernel>,
}

impl<B: Backend> GemmMixedTypesSimpleKernel<B> {
    pub fn new(
        a_dtype: DataType,
        b_dtype: DataType,
        output_dtype: DataType,
    ) -> Result<Self, MatmulError<B>> {
        let route = route_for_combo(a_dtype, b_dtype, output_dtype)
            .ok_or(MatmulError::UnsupportedDataType(output_dtype))?;

        Ok(Self {
            route,
            i8_i8_i32: None,
            i8_bf16_bf16: None,
            i8_f16_f16: None,
            i8_f32_f32: None,
        })
    }

    pub fn encode(
        &mut self,
        context: &B::Context,
        arguments: &mut MatmulArguments<B>,
        dispatch_descriptor: &DispatchDescriptor,
        encoder: &mut B::ComputeEncoder,
    ) -> Result<(), MatmulError<B>> {
        let group_count_x = u32::try_from(dispatch_descriptor.threadgroups.x)
            .map_err(|_| MatmulError::<B>::ThreadgroupOverflow(dispatch_descriptor.threadgroups.x))?;
        let group_count_y = u32::try_from(dispatch_descriptor.threadgroups.y)
            .map_err(|_| MatmulError::<B>::ThreadgroupOverflow(dispatch_descriptor.threadgroups.y))?;
        let group_count_z = u32::try_from(dispatch_descriptor.threadgroups.z)
            .map_err(|_| MatmulError::<B>::ThreadgroupOverflow(dispatch_descriptor.threadgroups.z))?;

        match self.route {
            MixedTypesSimpleRoute::I8I8I32 => {
                if self.i8_i8_i32.is_none() {
                    self.i8_i8_i32 = Some(
                        <B::Kernels as Kernels>::MixedTypesSimpleGemmI8I8I32Kernel::new(context)
                            .map_err(MatmulError::BackendError)?,
                    );
                }
                let p = self.i8_i8_i32.as_ref().unwrap();
                p.encode(
                    (arguments.a, arguments.a_offset as usize),
                    arguments.b,
                    arguments.d.deref_mut(),
                    std::slice::from_ref(&dispatch_descriptor.params),
                    group_count_x,
                    group_count_y,
                    group_count_z,
                    encoder,
                );
                Ok(())
            },
            MixedTypesSimpleRoute::I8Bf16Bf16 => {
                if self.i8_bf16_bf16.is_none() {
                    self.i8_bf16_bf16 = Some(
                        <B::Kernels as Kernels>::MixedTypesSimpleGemmI8Bf16Bf16Kernel::new(context)
                            .map_err(MatmulError::BackendError)?,
                    );
                }
                let p = self.i8_bf16_bf16.as_ref().unwrap();
                p.encode(
                    (arguments.a, arguments.a_offset as usize),
                    arguments.b,
                    arguments.d.deref_mut(),
                    std::slice::from_ref(&dispatch_descriptor.params),
                    group_count_x,
                    group_count_y,
                    group_count_z,
                    encoder,
                );
                Ok(())
            },
            MixedTypesSimpleRoute::I8F16F16 => {
                if self.i8_f16_f16.is_none() {
                    self.i8_f16_f16 = Some(
                        <B::Kernels as Kernels>::MixedTypesSimpleGemmI8F16F16Kernel::new(context)
                            .map_err(MatmulError::BackendError)?,
                    );
                }
                let p = self.i8_f16_f16.as_ref().unwrap();
                p.encode(
                    (arguments.a, arguments.a_offset as usize),
                    arguments.b,
                    arguments.d.deref_mut(),
                    std::slice::from_ref(&dispatch_descriptor.params),
                    group_count_x,
                    group_count_y,
                    group_count_z,
                    encoder,
                );
                Ok(())
            },
            MixedTypesSimpleRoute::I8F32F32 => {
                if self.i8_f32_f32.is_none() {
                    self.i8_f32_f32 = Some(
                        <B::Kernels as Kernels>::MixedTypesSimpleGemmI8F32F32Kernel::new(context)
                            .map_err(MatmulError::BackendError)?,
                    );
                }
                let p = self.i8_f32_f32.as_ref().unwrap();
                p.encode(
                    (arguments.a, arguments.a_offset as usize),
                    arguments.b,
                    arguments.d.deref_mut(),
                    std::slice::from_ref(&dispatch_descriptor.params),
                    group_count_x,
                    group_count_y,
                    group_count_z,
                    encoder,
                );
                Ok(())
            },
        }
    }
}
