use crate::{
    backends::common::{Backend, Context, CpuBuffer, GlobalBufferCpuGpu},
    data_type::DataTypeRepr,
};

pub fn create_buffer_with_data<B: Backend, T: DataTypeRepr>(
    context: &B::Context,
    data: &[T],
) -> Result<B::GlobalGpuBuffer, B::Error> {
    let mut buffer = context.create_buffer_cpu_gpu(size_of_val(data))?;
    buffer.copyin(data);
    buffer.handoff()
}
