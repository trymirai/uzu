use std::mem::size_of;

use super::reference::{read_f32, write_f32};
use crate::{
    backends::{
        common::{
            Allocation, AsBufferRangeMut, AsBufferRangeRef, BufferArg, Encoder,
            gpu_types::gemm::GemmDTransform,
            kernel::matmul::{
                MatmulA, MatmulArguments, MatmulB, MatmulError, MatmulMatrixMap, MatmulRouting, MatmulRowMap,
            },
        },
        cpu::{Cpu, error::CpuError},
    },
    data_type::DataType,
    utils::pointers::{SendPtr, SendPtrMut},
};

#[derive(Clone, Copy)]
struct RowMapData {
    indices: SendPtr<u32>,
    index_divisor: usize,
}

impl RowMapData {
    fn new(row_map: MatmulRowMap<'_, Cpu>) -> Self {
        let range = row_map.indices.as_buffer_range_ref();
        let indices = unsafe { &*range.buffer().get() }.as_ptr().wrapping_byte_add(range.range().start) as *const u32;
        Self {
            indices: SendPtr(indices),
            index_divisor: row_map.index_divisor.get() as usize,
        }
    }

    unsafe fn resolve(
        self,
        assignment: usize,
    ) -> usize {
        unsafe { *self.indices.as_ptr().add(assignment) as usize / self.index_divisor }
    }
}

#[derive(Clone, Copy)]
enum MatrixMapData {
    Shared,
    Segmented {
        offsets: SendPtr<u32>,
        matrix_count: usize,
    },
}

const PATH: &str = "CpuGatheredMatmul";

fn invalid_routing(reason: &'static str) -> CpuError {
    MatmulError::InvalidRouting {
        path: PATH,
        reason,
    }
    .into()
}

fn checked_product(
    factors: &[usize],
    reason: &'static str,
) -> Result<usize, CpuError> {
    factors
        .iter()
        .try_fold(1usize, |product, factor| product.checked_mul(*factor))
        .ok_or_else(|| invalid_routing(reason))
}

fn require_allocation_size(
    allocation: &Allocation<Cpu>,
    required_bytes: usize,
    reason: &'static str,
) -> Result<(), CpuError> {
    if allocation.size() < required_bytes {
        return Err(invalid_routing(reason));
    }
    Ok(())
}

pub(super) fn encode<'a, 'b, 'd, TB: BufferArg<'b, Cpu>>(
    arguments: MatmulArguments<'a, 'b, 'd, Cpu, TB>,
    weights_data_type: DataType,
    input_data_type: DataType,
    output_data_type: DataType,
    encoder: &mut Encoder<Cpu>,
) -> Result<(), CpuError> {
    let MatmulRouting::Gathered(gathered) = arguments.routing else {
        unreachable!();
    };

    if arguments.d_transform.rht_factors.is_some() {
        return Err(MatmulError::UnsupportedDOp {
            bit: GemmDTransform::RHT,
            path: PATH,
        }
        .into());
    }

    let MatmulA::FullPrecision {
        values: a,
        offset: a_offset,
    } = arguments.a
    else {
        return Err(MatmulError::IncompatibleA {
            path: PATH,
            reason: "the CPU gathered reference currently requires full-precision A",
        }
        .into());
    };
    let MatmulB::FullPrecision {
        b,
    } = arguments.b
    else {
        return Err(MatmulError::IncompatibleB {
            path: PATH,
            reason: "the CPU gathered reference currently requires full-precision B",
        }
        .into());
    };

    let m = arguments.m as usize;
    let n = arguments.n as usize;
    let k = arguments.k as usize;
    let weights_element_size = weights_data_type.size_in_bytes();
    let input_element_size = input_data_type.size_in_bytes();
    let output_element_size = output_data_type.size_in_bytes();

    let row_map_bytes = checked_product(&[m, size_of::<u32>()], "row-map size overflows usize")?;
    for row_map in [gathered.source_rows, gathered.destination_rows].into_iter().flatten() {
        require_allocation_size(row_map.indices, row_map_bytes, "row-map allocation is too small")?;
    }

    let (matrices, matrix_count) = match gathered.matrices {
        MatmulMatrixMap::Shared => (MatrixMapData::Shared, 1usize),
        MatmulMatrixMap::Segmented {
            offsets,
            matrix_count,
        } => {
            let matrix_count = matrix_count.get() as usize;
            let offset_count =
                matrix_count.checked_add(1).ok_or_else(|| invalid_routing("offset count overflows usize"))?;
            let offset_bytes =
                checked_product(&[offset_count, size_of::<u32>()], "matrix-offset size overflows usize")?;
            require_allocation_size(offsets, offset_bytes, "matrix-offset allocation is too small")?;
            let range = offsets.as_buffer_range_ref();
            let offsets =
                unsafe { &*range.buffer().get() }.as_ptr().wrapping_byte_add(range.range().start) as *const u32;
            (
                MatrixMapData::Segmented {
                    offsets: SendPtr(offsets),
                    matrix_count,
                },
                matrix_count,
            )
        },
    };

    let b_leading_dimension = arguments.b_leading_dimension.map_or_else(
        || {
            if arguments.b_transpose {
                k
            } else {
                n
            }
        },
        |leading_dimension| leading_dimension as usize,
    );
    let minimum_b_leading_dimension = if arguments.b_transpose {
        k
    } else {
        n
    };
    if b_leading_dimension < minimum_b_leading_dimension {
        return Err(MatmulError::IncompatibleB {
            path: PATH,
            reason: "B leading dimension is smaller than its logical row width",
        }
        .into());
    }
    let b_matrix_stride = if arguments.b_transpose {
        checked_product(&[n, b_leading_dimension], "transposed B matrix stride overflows usize")?
    } else {
        checked_product(&[k, b_leading_dimension], "B matrix stride overflows usize")?
    };
    let required_b_bytes =
        checked_product(&[matrix_count, b_matrix_stride, weights_element_size], "B matrix-bank size overflows usize")?;

    let a_range = a.as_buffer_range_ref();
    // MatmulA offsets are byte-addressed, matching BufferArg on accelerator backends.
    if a_offset > a_range.range().len() {
        return Err(MatmulError::IncompatibleA {
            path: PATH,
            reason: "A offset exceeds the operand allocation",
        }
        .into());
    }
    if !a_offset.is_multiple_of(input_element_size) {
        return Err(MatmulError::IncompatibleA {
            path: PATH,
            reason: "A offset is not aligned to its element type",
        }
        .into());
    }
    let a_available_elements = (a_range.range().len() - a_offset) / input_element_size;
    let a_row_count = a_available_elements.checked_div(k).unwrap_or(usize::MAX);
    let a_byte_offset =
        a_range.range().start.checked_add(a_offset).ok_or_else(|| invalid_routing("A byte offset overflows usize"))?;
    let a_ptr = SendPtr(unsafe { &*a_range.buffer().get() }.as_ptr().wrapping_byte_add(a_byte_offset));

    let (b_buffer, b_byte_offset, b_byte_length) = b.into_parts();
    if b_byte_length < required_b_bytes {
        return Err(MatmulError::IncompatibleB {
            path: PATH,
            reason: "B allocation is too small for the selected matrix bank",
        }
        .into());
    }
    let b_ptr = SendPtr(unsafe { &*b_buffer.downcast().get() }.as_ptr().wrapping_byte_add(b_byte_offset));

    let required_d_bytes = checked_product(&[m, n, output_element_size], "D allocation size overflows usize")?;
    let d_range = arguments.d.as_buffer_range_mut();
    if d_range.range().len() < required_d_bytes {
        return Err(invalid_routing("D allocation is too small for routed output rows"));
    }
    let d_ptr =
        SendPtrMut(unsafe { (&*d_range.buffer().get()).as_ptr().wrapping_byte_add(d_range.range().start) as *mut u8 });

    let matrix_scale_bytes =
        checked_product(&[matrix_count, weights_element_size], "matrix-scale allocation size overflows usize")?;
    if let Some(scales) = gathered.matrix_scales {
        require_allocation_size(scales, matrix_scale_bytes, "matrix-scale allocation is too small")?;
    }
    let matrix_bias_bytes =
        checked_product(&[matrix_count, n, weights_element_size], "matrix-bias allocation size overflows usize")?;
    if let Some(biases) = gathered.matrix_biases {
        require_allocation_size(biases, matrix_bias_bytes, "matrix-bias allocation is too small")?;
    }
    let shared_bias_bytes = checked_product(&[n, weights_element_size], "shared-bias allocation size overflows usize")?;
    if let Some(bias) = arguments.d_transform.bias {
        require_allocation_size(bias, shared_bias_bytes, "shared-bias allocation is too small")?;
    }

    let allocation_ptr = |allocation: &Allocation<Cpu>| {
        let range = allocation.as_buffer_range_ref();
        SendPtr(unsafe { &*range.buffer().get() }.as_ptr().wrapping_byte_add(range.range().start))
    };
    let source_rows = gathered.source_rows.map(RowMapData::new);
    let destination_rows = gathered.destination_rows.map(RowMapData::new);
    let matrix_scales = gathered.matrix_scales.map(allocation_ptr);
    let matrix_biases = gathered.matrix_biases.map(allocation_ptr);
    let shared_bias = arguments.d_transform.bias.map(allocation_ptr);

    let b_transpose = arguments.b_transpose;
    let output_scale = arguments.d_transform.ab_scale;
    let accumulate = arguments.d_transform.accumulate;
    let soft_cap = arguments.d_transform.soft_cap;

    let command_buffer = encoder.as_command_buffer_mut();
    command_buffer.push_command(move || {
        let mut resolved_sources = Vec::with_capacity(m);
        let mut resolved_destinations = Vec::with_capacity(m);
        let mut destinations_seen = vec![false; m];
        for assignment in 0..m {
            let source = match source_rows {
                Some(rows) => unsafe { rows.resolve(assignment) },
                None => assignment,
            };
            assert!(source < a_row_count, "gathered source row is outside A");
            resolved_sources.push(source);

            let destination = match destination_rows {
                Some(rows) => unsafe { rows.resolve(assignment) },
                None => assignment,
            };
            assert!(destination < m, "gathered destination row is outside D");
            assert!(!destinations_seen[destination], "gathered destination rows must be unique");
            destinations_seen[destination] = true;
            resolved_destinations.push(destination);
        }

        // Validate every device-produced segment before any operand or output pointer is dereferenced.
        let segments = match matrices {
            MatrixMapData::Shared => vec![(0, m)],
            MatrixMapData::Segmented {
                offsets,
                matrix_count,
            } => unsafe {
                assert_eq!(*offsets.as_ptr(), 0, "matrix offsets must start at zero");
                let mut segments = Vec::with_capacity(matrix_count);
                for matrix in 0..matrix_count {
                    let begin = *offsets.as_ptr().add(matrix) as usize;
                    let end = *offsets.as_ptr().add(matrix + 1) as usize;
                    assert!(begin <= end, "matrix offsets must be nondecreasing");
                    assert!(end <= m, "matrix offsets must not exceed m");
                    segments.push((begin, end));
                }
                assert_eq!(*offsets.as_ptr().add(matrix_count) as usize, m, "matrix offsets must end at m");
                segments
            },
        };

        let encode_assignment = |assignment: usize, matrix: usize| unsafe {
            let source = resolved_sources[assignment];
            let destination = resolved_destinations[assignment];
            let matrix_scale =
                matrix_scales.map(|scales| read_f32(scales.as_ptr(), weights_data_type, matrix)).unwrap_or(1.0);

            for column in 0..n {
                let mut accumulator = 0.0f32;
                for inner in 0..k {
                    let a_value = read_f32(a_ptr.as_ptr(), input_data_type, source * k + inner);
                    let b_index = matrix * b_matrix_stride
                        + if b_transpose {
                            column * b_leading_dimension + inner
                        } else {
                            inner * b_leading_dimension + column
                        };
                    let b_value = read_f32(b_ptr.as_ptr(), weights_data_type, b_index);
                    accumulator += a_value * b_value;
                }

                let output_index = destination * n + column;
                let mut value = output_scale * matrix_scale * accumulator;
                if accumulate {
                    value += read_f32(d_ptr.as_ptr(), output_data_type, output_index);
                }
                if let Some(bias) = shared_bias {
                    value += read_f32(bias.as_ptr(), weights_data_type, column);
                }
                if let Some(biases) = matrix_biases {
                    value += read_f32(biases.as_ptr(), weights_data_type, matrix * n + column);
                }
                if let Some(cap) = soft_cap {
                    value = cap * (value / cap).tanh();
                }
                write_f32(d_ptr.as_ptr(), output_data_type, output_index, value);
            }
        };

        for (matrix, (begin, end)) in segments.into_iter().enumerate() {
            for assignment in begin..end {
                encode_assignment(assignment, matrix);
            }
        }
    });

    Ok(())
}
