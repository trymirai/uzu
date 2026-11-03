//! IOSurfaces in the ANE's layout, aliased zero-copy by Metal buffers.
//!
//! A surface holds `rows` rows of `width` elements, one channel per row and tokens along the row
//! (`[1, 1, rows, width]` in MIL), each row padded to 64 bytes, the allocation to 16 KiB.

use std::{ffi::c_void, ptr::NonNull};

use half::f16;
use metal::{MTLBuffer, MTLDeviceExt, MTLResourceOptions};
use objc2_core_foundation::{CFDictionary, CFNumber, CFRetained, CFString};
use objc2_io_surface::{
    IOSurfaceRef, kIOSurfaceAllocSize, kIOSurfaceBytesPerElement, kIOSurfaceBytesPerRow, kIOSurfaceHeight,
    kIOSurfacePixelFormat, kIOSurfaceWidth,
};

use super::AneError;
use crate::backends::{
    common::Buffer,
    metal::{buffer::external::MetalExternalBuffer, context::MetalContext},
};

const ROW_ALIGNMENT: usize = 64;
const ALLOCATION_ALIGNMENT: usize = 16 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AneElement {
    Int8,
    Float16,
}

impl AneElement {
    pub const fn bytes(self) -> usize {
        match self {
            Self::Int8 => 1,
            Self::Float16 => 2,
        }
    }

    pub const fn mil_type(self) -> &'static str {
        match self {
            Self::Int8 => "int8",
            Self::Float16 => "fp16",
        }
    }

    /// The pixel formats the ANE service expects: 'L008' and 'L00h'.
    const fn pixel_format(self) -> i64 {
        match self {
            Self::Int8 => 0x4c30_3038,
            Self::Float16 => 0x4c30_3068,
        }
    }
}

pub struct AneSurface {
    // Declared before the surface: the alias is released before the memory it aliases.
    buffer: MetalExternalBuffer,
    surface: CFRetained<IOSurfaceRef>,
    rows: u32,
    width: u32,
    element: AneElement,
    row_stride_bytes: usize,
}

// SAFETY: an IOSurface and a Metal buffer are thread-safe handles; access to their memory is
// ordered by the GPU/ANE synchronization of the code that uses them.
unsafe impl Send for AneSurface {}
unsafe impl Sync for AneSurface {}

fn key(string: &CFString) -> &CFString {
    string
}

/// The bytes between rows of a surface `width` elements wide.
pub fn row_stride_bytes(
    width: u32,
    element: AneElement,
) -> usize {
    (width as usize * element.bytes()).next_multiple_of(ROW_ALIGNMENT)
}

/// The MIL strides of a `[1, 1, rows, width]` view of a surface whose rows are
/// `row_stride_elements` apart.
pub fn mil_strides(
    rows: u32,
    row_stride_elements: usize,
) -> String {
    let stride = row_stride_elements as u64;
    let plane = u64::from(rows) * stride;
    format!("[{plane}, {plane}, {stride}, 1]")
}

/// The MIL type of a function parameter reading the leading `width` elements of the leading
/// `rows` rows of a surface whose rows are `row_stride_elements` apart.
pub fn mil_buffer_type(
    element: AneElement,
    rows: u32,
    width: u32,
    row_stride_elements: usize,
) -> String {
    format!(
        "tensor_buffer<{}, shape=[1, 1, {rows}, {width}], strides={}, interleave_factors=[1, 1, 1, 1]>",
        element.mil_type(),
        mil_strides(rows, row_stride_elements)
    )
}

impl AneSurface {
    pub fn new(
        context: &MetalContext,
        rows: u32,
        width: u32,
        element: AneElement,
    ) -> Result<Self, AneError> {
        let row_stride_bytes = row_stride_bytes(width, element);
        let size = (row_stride_bytes * rows as usize).next_multiple_of(ALLOCATION_ALIGNMENT);

        let keys = unsafe {
            [
                key(kIOSurfaceWidth),
                key(kIOSurfaceHeight),
                key(kIOSurfaceBytesPerElement),
                key(kIOSurfaceBytesPerRow),
                key(kIOSurfaceAllocSize),
                key(kIOSurfacePixelFormat),
            ]
        };
        let values = [
            CFNumber::new_i64(i64::from(width)),
            CFNumber::new_i64(i64::from(rows)),
            CFNumber::new_i64(element.bytes() as i64),
            CFNumber::new_i64(row_stride_bytes as i64),
            CFNumber::new_i64(size as i64),
            CFNumber::new_i64(element.pixel_format()),
        ];
        let value_refs: Vec<&CFNumber> = values.iter().map(|value| &**value).collect();
        let properties = CFDictionary::<CFString, CFNumber>::from_slices(&keys, &value_refs);
        let surface = unsafe { IOSurfaceRef::new(properties.as_opaque()) }
            .ok_or_else(|| AneError::Surface(format!("rows={rows} width={width} {element:?}")))?;
        if surface.bytes_per_row() != row_stride_bytes || surface.alloc_size() != size {
            return Err(AneError::Surface("IOSurface took another row stride or size".to_string()));
        }

        let base: NonNull<c_void> = surface.base_address();
        let alias = context
            .device
            .new_buffer_with_bytes_no_copy(base, size, MTLResourceOptions::STORAGE_MODE_SHARED)
            .ok_or_else(|| AneError::Surface("Metal cannot alias the IOSurface".to_string()))?;

        Ok(Self {
            buffer: MetalExternalBuffer::new(context, alias),
            surface,
            rows,
            width,
            element,
            row_stride_bytes,
        })
    }

    pub fn io_surface(&self) -> &IOSurfaceRef {
        &self.surface
    }

    /// The Metal buffer over the surface's memory (shared storage, resident while it lives),
    /// usable as a kernel argument.
    pub fn buffer(&self) -> &MetalExternalBuffer {
        &self.buffer
    }

    pub fn buffer_mut(&mut self) -> &mut MetalExternalBuffer {
        &mut self.buffer
    }

    pub fn rows(&self) -> u32 {
        self.rows
    }

    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn element(&self) -> AneElement {
        self.element
    }

    pub fn row_stride_bytes(&self) -> usize {
        self.row_stride_bytes
    }

    pub fn row_stride_elements(&self) -> usize {
        self.row_stride_bytes / self.element.bytes()
    }

    pub fn size(&self) -> usize {
        self.buffer.size()
    }

    /// The MIL type of a function parameter reading the leading `width` elements of the leading
    /// `rows` rows.
    pub fn mil_buffer_type(
        &self,
        rows: u32,
        width: u32,
    ) -> String {
        mil_buffer_type(self.element, rows, width, self.row_stride_elements())
    }

    fn bytes_mut(&mut self) -> &mut [u8] {
        let size = self.size();
        unsafe { std::slice::from_raw_parts_mut(self.buffer.mtl_buffer().contents().as_ptr().cast::<u8>(), size) }
    }

    fn bytes(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.buffer.mtl_buffer().contents().as_ptr().cast::<u8>(), self.size()) }
    }

    /// The whole allocation as int8 (row `r` starts at `r * row_stride_elements()`).
    pub fn as_i8_mut(&mut self) -> &mut [i8] {
        bytemuck::cast_slice_mut(self.bytes_mut())
    }

    pub fn as_i8(&self) -> &[i8] {
        bytemuck::cast_slice(self.bytes())
    }

    /// The whole allocation as fp16 (row `r` starts at `r * row_stride_elements()`).
    pub fn as_f16_mut(&mut self) -> &mut [f16] {
        bytemuck::cast_slice_mut(self.bytes_mut())
    }

    pub fn as_f16(&self) -> &[f16] {
        bytemuck::cast_slice(self.bytes())
    }

    /// The element at (`row`, `column`) as f32.
    pub fn get(
        &self,
        row: u32,
        column: u32,
    ) -> f32 {
        let index = row as usize * self.row_stride_elements() + column as usize;
        match self.element {
            AneElement::Int8 => f32::from(self.as_i8()[index]),
            AneElement::Float16 => self.as_f16()[index].to_f32(),
        }
    }
}
