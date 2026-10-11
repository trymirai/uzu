use std::{
    collections::BTreeMap,
    ffi::CStr,
    fmt::Debug,
    mem::size_of,
    ops::Range,
    sync::Arc,
    time::{Duration, Instant},
};

use bytemuck::{AnyBitPattern, NoUninit};
use num_traits::Float;

use super::validation_logger::ValidationLogger;
use crate::backends::vulkan::{
    VkBuffer, VkCommandBufferCompleted, VkCommandBufferEncoding, VkContext, VkContextCreateInfo,
};

/// A validated Vulkan context with the buffer and submission helpers shared by the Vulkan tests.
pub struct KernelFixture {
    pub context: Arc<VkContext>,
    logger: ValidationLogger,
}

impl KernelFixture {
    /// Sentinel elements before and after every guarded range, so guarded ranges start at a nonzero offset.
    pub const GUARD: usize = 64;

    pub fn new() -> Self {
        let logger = ValidationLogger::default();
        let context = VkContext::new(VkContextCreateInfo {
            with_validation: true,
            logger: Box::new(logger.clone()),
        })
        .expect("Vulkan context");
        let physical_device = context.physical_device();
        let name = unsafe { CStr::from_ptr(physical_device.properties.device_name.as_ptr()) };
        eprintln!(
            "Vulkan test device: {} ({:?}, subgroup size {})",
            name.to_string_lossy(),
            physical_device.properties.device_type,
            physical_device.subgroup_properties.size
        );
        Self {
            context: Arc::new(context),
            logger,
        }
    }

    pub fn buffer<T: NoUninit>(
        &self,
        values: &[T],
    ) -> Arc<VkBuffer> {
        let bytes = bytemuck::cast_slice::<T, u8>(values);
        let mut buffer = VkBuffer::new(self.context.clone(), bytes.len() as u64).expect("buffer");
        buffer.fill(bytes).expect("host fill");
        Arc::new(buffer)
    }

    /// A buffer holding `values` between `GUARD` sentinels on each side, with the byte range of `values`.
    pub fn guarded<T: NoUninit>(
        &self,
        values: &[T],
        sentinel: T,
    ) -> (Arc<VkBuffer>, Range<u64>) {
        let buffer = self.buffer(&[vec![sentinel; Self::GUARD], values.to_vec(), vec![sentinel; Self::GUARD]].concat());
        (buffer, (Self::GUARD * size_of::<T>()) as u64..((Self::GUARD + values.len()) * size_of::<T>()) as u64)
    }

    pub fn encoding(&self) -> VkCommandBufferEncoding {
        VkCommandBufferEncoding::new(self.context.clone()).expect("command buffer")
    }

    pub fn complete(encoding: VkCommandBufferEncoding) -> VkCommandBufferCompleted {
        encoding.end_encoding().expect("end encoding").submit().wait_until_completed().expect("completion")
    }

    /// # Safety
    /// Same contract as `VkBuffer::get_bytes`.
    pub unsafe fn read<T: NoUninit + AnyBitPattern>(buffer: &VkBuffer) -> Vec<T> {
        bytemuck::pod_collect_to_vec(&unsafe { buffer.get_bytes() })
    }

    /// Reads a range made by `guarded` after asserting every guard element still holds `sentinel`, bit for bit.
    ///
    /// # Safety
    /// Same contract as `VkBuffer::get_bytes`.
    pub unsafe fn read_guarded<T: NoUninit + AnyBitPattern>(
        (buffer, range): &(Arc<VkBuffer>, Range<u64>),
        sentinel: T,
    ) -> Vec<T> {
        let values = unsafe { Self::read::<T>(buffer) };
        let (start, end) = (range.start as usize / size_of::<T>(), range.end as usize / size_of::<T>());
        let mut guards = values[..start].iter().chain(&values[end..]);
        assert!(guards.all(|value| bytemuck::bytes_of(value) == bytemuck::bytes_of(&sentinel)), "a guard was written");
        values[start..end].to_vec()
    }

    /// Bit equality, except that any NaN matches any NaN (payloads after FP32 arithmetic are not portable).
    pub fn assert_bits<T: NoUninit + Float + Debug>(
        expected: &[T],
        actual: &[T],
        case: &str,
    ) {
        assert_eq!(expected.len(), actual.len(), "{case}: length");
        for (index, (&expected, &actual)) in expected.iter().zip(actual).enumerate() {
            let same = match expected.is_nan() {
                true => actual.is_nan(),
                false => bytemuck::bytes_of(&expected) == bytemuck::bytes_of(&actual),
            };
            assert!(same, "{case}: element {index}: CPU {expected:?}, Vulkan {actual:?}");
        }
    }

    /// Reads a read-only range made by `guarded` and asserts it and its guards are unchanged, bit for bit.
    ///
    /// # Safety
    /// Same contract as `VkBuffer::get_bytes`.
    pub unsafe fn assert_unchanged<T: NoUninit + AnyBitPattern>(
        guarded: &(Arc<VkBuffer>, Range<u64>),
        sentinel: T,
        payload: &[T],
        name: &str,
    ) {
        let read = unsafe { Self::read_guarded(guarded, sentinel) };
        assert_eq!(bytemuck::cast_slice::<T, u8>(&read), bytemuck::cast_slice::<T, u8>(payload), "{name} changed");
    }

    /// Position of a value in the total order of its storage type, so differences count representable steps.
    pub fn ordinal<T: NoUninit>(value: T) -> i64 {
        let (bits, sign) = match *bytemuck::bytes_of(&value) {
            [a, b] => (i64::from(u16::from_ne_bytes([a, b])), 1 << 15),
            [a, b, c, d] => (i64::from(u32::from_ne_bytes([a, b, c, d])), 1 << 31),
            _ => unreachable!("storage types are 16 or 32 bits"),
        };
        if bits & sign != 0 {
            -(bits & !sign)
        } else {
            bits
        }
    }

    /// Compares Vulkan with an expected result: NaN must occur exactly where expected and infinities must match exactly;
    /// other elements must be within 2 representable steps for 16-bit storage, or within `relative` or `absolute` for
    /// FP32 storage. Returns the max absolute, relative and step errors plus the number of bound violations, printing
    /// the first one with exact bits.
    pub fn compare<T: NoUninit + Float + Debug>(
        expected: &[T],
        actual: &[T],
        case: &str,
        relative: f64,
        absolute: f64,
    ) -> [f64; 4] {
        assert_eq!(expected.len(), actual.len(), "{case}: length");
        let mut max = [0.0f64; 4];
        for (index, (&expected, &actual)) in expected.iter().zip(actual).enumerate() {
            let (expected_ordinal, actual_ordinal) = (Self::ordinal(expected), Self::ordinal(actual));
            let values =
                format!("expected {expected:?} ({expected_ordinal:#x}), Vulkan {actual:?} ({actual_ordinal:#x})");
            assert_eq!(expected.is_nan(), actual.is_nan(), "{case}: element {index}: {values}");
            if expected.is_nan() {
                continue;
            }
            let (e, a) = (expected.to_f64().unwrap(), actual.to_f64().unwrap());
            let steps = (actual_ordinal - expected_ordinal).abs() as f64;
            let error = [(a - e).abs(), (a - e).abs() / e.abs().max(f64::MIN_POSITIVE), steps];
            let within = e.is_finite()
                && a.is_finite()
                && match size_of::<T>() {
                    4 => error[1] <= relative || error[0] <= absolute,
                    _ => steps <= 2.0,
                };
            if !within && e != a {
                if max[3] == 0.0 {
                    eprintln!("{case}: element {index} exceeds the bound: {values}");
                }
                max[3] += 1.0;
            }
            max = [max[0].max(error[0]), max[1].max(error[1]), max[2].max(error[2]), max[3]];
        }
        max
    }

    /// Prints the max errors and bound violations of each label, then fails if any element violated its bound.
    pub fn report(
        kernel: &str,
        errors: impl IntoIterator<Item = (String, [f64; 4])>,
    ) {
        let mut totals = BTreeMap::<String, [f64; 4]>::new();
        for (label, error) in errors {
            let total = totals.entry(label).or_default();
            *total = [total[0].max(error[0]), total[1].max(error[1]), total[2].max(error[2]), total[3] + error[3]];
        }
        for (label, [absolute, relative, steps, violations]) in &totals {
            eprintln!(
                "{kernel} {label}: max absolute {absolute:.3e}, max relative {relative:.3e}, max {steps} storage \
                 steps, {violations} bound violations"
            );
        }
        assert!(totals.values().all(|total| total[3] == 0.0), "{kernel}: elements exceed the bound");
    }

    /// Median GPU and wall time of 10 submissions after 3 warm-up ones, each holding what `encode` records.
    pub fn median_times(
        &self,
        mut encode: impl FnMut(&mut VkCommandBufferEncoding),
    ) -> (Duration, Duration) {
        let mut samples = (0..13)
            .map(|_| {
                let start = Instant::now();
                let mut encoding = self.encoding();
                encode(&mut encoding);
                (Self::complete(encoding).gpu_execution_time(), start.elapsed())
            })
            .skip(3)
            .collect::<Vec<_>>();
        let mut median = |key: fn(&(Duration, Duration)) -> Duration| {
            samples.sort_by_key(key);
            key(&samples[samples.len() / 2])
        };
        (median(|sample| sample.0), median(|sample| sample.1))
    }

    pub fn assert_clean(&self) {
        self.logger.assert_clean();
    }
}
