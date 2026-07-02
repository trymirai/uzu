use std::{mem::size_of, rc::Rc};

use rand::{RngExt, SeedableRng, rngs::SmallRng};

use crate::{
    array::{ArrayContextExt, ArrayElement},
    backends::common::{
        Allocation, AllocationType, AsBufferRangeMut, Backend, Context, DenseBuffer, Encoder, SparseBuffer,
        SparseBufferExt,
    },
    data_type::DataType,
    dispatch_dtype,
};

pub fn allocation_size_bytes<T>(elements_count: usize) -> usize {
    elements_count * size_of::<T>()
}

pub fn alloc_allocation<B: Backend, T>(
    context: &B::Context,
    elements_count: usize,
) -> Allocation<B> {
    context
        .create_allocation(allocation_size_bytes::<T>(elements_count), AllocationType::Global)
        .expect("Failed to create allocation")
}

pub fn alloc_allocation_with_data<B: Backend, T: ArrayElement>(
    context: &B::Context,
    data: &[T],
) -> Allocation<B> {
    let mut allocation = context
        .create_allocation(allocation_size_bytes::<T>(data.len()), AllocationType::Global)
        .expect("Failed to create allocation");
    write_allocation(&mut allocation, data);
    allocation
}

pub fn allocation_to_vec<B: Backend, T: ArrayElement>(allocation: &Allocation<B>) -> Vec<T> {
    allocation.copyout()
}

pub fn allocation_prefix_to_vec<B: Backend, T: ArrayElement>(
    allocation: &Allocation<B>,
    elements_count: usize,
) -> Vec<T> {
    let mut values = allocation_to_vec::<B, T>(allocation);
    values.truncate(elements_count);
    values
}

pub fn write_allocation<B: Backend, T: ArrayElement>(
    allocation: &mut Allocation<B>,
    data: &[T],
) {
    let bytes = bytemuck::cast_slice(data);
    let buffer_range = allocation.as_buffer_range_mut();
    let range = buffer_range.range();
    assert!(bytes.len() <= range.len(), "source data is larger than destination allocation");
    let destination = unsafe {
        std::slice::from_raw_parts_mut(
            (buffer_range.buffer().cpu_ptr().as_ptr() as *mut u8).add(range.start),
            range.len(),
        )
    };
    destination[..bytes.len()].copy_from_slice(bytes);
}

pub fn seed_from_label(label: &str) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    label.hash(&mut hasher);
    hasher.finish()
}

pub fn random_buffer<B: Backend>(
    context: &B::Context,
    shape: &[usize],
    data_type: DataType,
    seed: u64,
) -> Allocation<B> {
    let element_count: usize = shape.iter().product();
    dispatch_dtype!(|(Element: data_type)| {
        let mut random = SmallRng::seed_from_u64(seed);
        let data: Vec<Element> = (0..element_count)
            .map(|_| <Element as num_traits::NumCast>::from(random.random_range(-1.0f32..1.0f32)).unwrap())
            .collect();
        context.create_array_from(shape, &data).into_allocation()
    })
}

pub fn zeroed_buffer<B: Backend>(
    context: &B::Context,
    shape: &[usize],
    data_type: DataType,
) -> Allocation<B> {
    let element_count: usize = shape.iter().product();
    macro_rules! zeros {
        ($element:ty) => {
            context.create_array_from(shape, &vec![<$element>::default(); element_count]).into_allocation()
        };
    }
    match data_type {
        DataType::U8 => zeros!(u8),
        DataType::I8 => zeros!(i8),
        DataType::U16 => zeros!(u16),
        DataType::I16 => zeros!(i16),
        DataType::U32 => zeros!(u32),
        DataType::I32 => zeros!(i32),
        DataType::U64 => zeros!(u64),
        DataType::I64 => zeros!(i64),
        _ => random_buffer::<B>(context, shape, data_type, 0),
    }
}

pub fn measurement_buffer<B: Backend>(
    context: &B::Context,
    shape: &[usize],
    data_type: DataType,
    seed: u64,
) -> Allocation<B> {
    match data_type {
        DataType::F16 | DataType::BF16 | DataType::F32 | DataType::F64 => {
            random_buffer::<B>(context, shape, data_type, seed)
        },
        _ => zeroed_buffer::<B>(context, shape, data_type),
    }
}

pub fn create_context<B: Backend>() -> Rc<<B as Backend>::Context> {
    B::Context::new().unwrap_or_else(|_| panic!("Failed to create context for {}", std::any::type_name::<B>()))
}

pub fn submit_encoder<B: Backend>(encoder: Encoder<B>) {
    encoder.end_encoding().submit().wait_until_completed().unwrap();
}

pub fn sparse_buffer_create<B: Backend>(
    context: &B::Context,
    capacity: usize,
) -> B::SparseBuffer {
    context.create_sparse_buffer(capacity).expect("Failed to create sparse buffer")
}

pub fn sparse_buffer_create_and_map<B: Backend>(
    context: &B::Context,
    capacity: usize,
) -> B::SparseBuffer {
    let mut buffer = sparse_buffer_create::<B>(context, capacity);
    buffer.map(context, &(0..buffer.total_pages())).expect("Failed to map sparse buffer");
    buffer
}

pub fn sparse_buffer_create_with<B: Backend, T: ArrayElement>(
    context: &B::Context,
    data: &[T],
) -> B::SparseBuffer {
    let capacity_bytes = allocation_size_bytes::<T>(data.len());
    let mut buffer = sparse_buffer_create_and_map::<B>(context, capacity_bytes);
    sparse_buffer_write::<B, T>(context, &mut buffer, data);
    buffer
}

pub fn sparse_buffer_read_allocation<B: Backend>(
    context: &B::Context,
    buffer: &B::SparseBuffer,
    size: usize,
) -> Allocation<B> {
    let mut allocation = alloc_allocation::<B, u8>(context, size);
    let range = 0..size;

    let mut encoder = Encoder::new(context).expect("Failed to create encoder");
    encoder.encode_copy(buffer, range.clone(), &mut allocation, range.clone());
    submit_encoder(encoder);

    allocation
}

pub fn sparse_buffer_read_vec<B: Backend, T: ArrayElement>(
    context: &B::Context,
    buffer: &B::SparseBuffer,
    elements_count: usize,
) -> Vec<T> {
    let allocation =
        sparse_buffer_read_allocation::<B>(context, buffer, elements_count * T::data_type().size_in_bytes());
    allocation_to_vec(&allocation)
}

pub fn sparse_buffer_write<B: Backend, T: ArrayElement>(
    context: &B::Context,
    buffer: &mut B::SparseBuffer,
    data: &[T],
) {
    let data_allocation = alloc_allocation_with_data::<B, T>(context, data);
    let data_range = 0..allocation_size_bytes::<T>(data.len());

    let mut encoder = Encoder::new(context).expect("Failed to create encoder");
    encoder.encode_copy(&data_allocation, data_range.clone(), buffer, data_range.clone());
    submit_encoder(encoder);
}
