use super::{super::matmul_arguments::MatmulArguments, dispatch_descriptor::DispatchDescriptor};
use crate::{
    DataType,
    backends::common::{
        Backend, Kernels,
        kernel::{
            ScalarIntGemmI8Bf16Bf16Kernel, ScalarIntGemmI8I8I32Kernel,
        },
    },
};

pub struct GemmScalarIntKernel<B: Backend> {
    a_dtype: DataType,
    b_dtype: DataType,
    i8_i8_i32: Option<<B::Kernels as Kernels>::ScalarIntGemmI8I8I32Kernel>,
    i8_bf16_bf16: Option<<B::Kernels as Kernels>::ScalarIntGemmI8Bf16Bf16Kernel>,
}

impl<B: Backend> GemmScalarIntKernel<B>
where
    B::Error: From<String>,
{
    pub fn new(a_dtype: DataType, b_dtype: DataType) -> Result<Self, B::Error> {
        Ok(Self {
            a_dtype,
            b_dtype,
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

        match (self.a_dtype, self.b_dtype) {
            (DataType::I8, DataType::I8) => {
                if self.i8_i8_i32.is_none() {
                    self.i8_i8_i32 = Some(<B::Kernels as Kernels>::ScalarIntGemmI8I8I32Kernel::new(context)?);
                }
                let p = self.i8_i8_i32.as_ref().unwrap();
                p.encode(
                    (arguments.a, arguments.a_offset as usize),
                    arguments.b, arguments.d,
                    std::slice::from_ref(&dispatch_descriptor.params),
                    group_count_x, group_count_y, group_count_z, encoder,
                );
                Ok(())
            },
            (DataType::I8, DataType::BF16) => {
                if self.i8_bf16_bf16.is_none() {
                    self.i8_bf16_bf16 = Some(<B::Kernels as Kernels>::ScalarIntGemmI8Bf16Bf16Kernel::new(context)?);
                }
                let p = self.i8_bf16_bf16.as_ref().unwrap();
                p.encode(
                    (arguments.a, arguments.a_offset as usize),
                    arguments.b, arguments.d,
                    std::slice::from_ref(&dispatch_descriptor.params),
                    group_count_x, group_count_y, group_count_z, encoder,
                );
                Ok(())
            },
            _ => Err(B::Error::from(format!(
                "ScalarIntGemm: unsupported combo {:?} * {:?}", self.a_dtype, self.b_dtype
            ))),
        }
    }
}
