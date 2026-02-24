use super::{super::matmul_arguments::MatmulArguments, dispatch_descriptor::DispatchDescriptor};
use crate::{
    DataType,
    backends::common::{
        Backend, Kernels,
        kernel::{ScalarIntGemmI8Bf16Bf16Kernel, ScalarIntGemmI8I8I32Kernel},
    },
};

#[derive(Debug, Clone, Copy)]
enum ScalarIntRoute {
    I8I8I32,
    I8Bf16Bf16,
}

fn route_for_combo(
    a_dtype: DataType,
    b_dtype: DataType,
    output_dtype: DataType,
) -> Option<ScalarIntRoute> {
    match (a_dtype, b_dtype, output_dtype) {
        (DataType::I8, DataType::I8, DataType::I32) => Some(ScalarIntRoute::I8I8I32),
        (DataType::I8, DataType::BF16, DataType::BF16) => Some(ScalarIntRoute::I8Bf16Bf16),
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

pub struct GemmScalarIntKernel<B: Backend> {
    route: ScalarIntRoute,
    i8_i8_i32: Option<<B::Kernels as Kernels>::ScalarIntGemmI8I8I32Kernel>,
    i8_bf16_bf16: Option<<B::Kernels as Kernels>::ScalarIntGemmI8Bf16Bf16Kernel>,
}

impl<B: Backend> GemmScalarIntKernel<B>
where
    B::Error: From<String>,
{
    pub fn new(
        a_dtype: DataType,
        b_dtype: DataType,
        output_dtype: DataType,
    ) -> Result<Self, B::Error> {
        let route = route_for_combo(a_dtype, b_dtype, output_dtype).ok_or_else(|| {
            B::Error::from(format!(
                "ScalarIntGemm: unsupported combo {:?} * {:?} -> {:?}",
                a_dtype, b_dtype, output_dtype
            ))
        })?;

        Ok(Self {
            route,
            i8_i8_i32: None,
            i8_bf16_bf16: None,
        })
    }

    pub fn encode(
        &mut self,
        context: &B::Context,
        arguments: &MatmulArguments<B>,
        dispatch_descriptor: &DispatchDescriptor,
        encoder: &B::ComputeEncoder,
    ) -> Result<(), B::Error> {
        let group_count_x = u32::try_from(dispatch_descriptor.threadgroups.x).map_err(|_| {
            B::Error::from(format!("ScalarIntGemm group count x overflows u32: {}", dispatch_descriptor.threadgroups.x))
        })?;
        let group_count_y = u32::try_from(dispatch_descriptor.threadgroups.y).map_err(|_| {
            B::Error::from(format!("ScalarIntGemm group count y overflows u32: {}", dispatch_descriptor.threadgroups.y))
        })?;
        let group_count_z = u32::try_from(dispatch_descriptor.threadgroups.z).map_err(|_| {
            B::Error::from(format!("ScalarIntGemm group count z overflows u32: {}", dispatch_descriptor.threadgroups.z))
        })?;

        match self.route {
            ScalarIntRoute::I8I8I32 => {
                if self.i8_i8_i32.is_none() {
                    self.i8_i8_i32 = Some(<B::Kernels as Kernels>::ScalarIntGemmI8I8I32Kernel::new(context)?);
                }
                let p = self.i8_i8_i32.as_ref().unwrap();
                p.encode(
                    (arguments.a, arguments.a_offset as usize),
                    arguments.b,
                    arguments.d,
                    std::slice::from_ref(&dispatch_descriptor.params),
                    group_count_x,
                    group_count_y,
                    group_count_z,
                    encoder,
                );
                Ok(())
            },
            ScalarIntRoute::I8Bf16Bf16 => {
                if self.i8_bf16_bf16.is_none() {
                    self.i8_bf16_bf16 = Some(<B::Kernels as Kernels>::ScalarIntGemmI8Bf16Bf16Kernel::new(context)?);
                }
                let p = self.i8_bf16_bf16.as_ref().unwrap();
                p.encode(
                    (arguments.a, arguments.a_offset as usize),
                    arguments.b,
                    arguments.d,
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
