//! The W8A8 dense-FFN program for the ANE (MIL text and its constant blob).
//!
//! One program holds a function per row count (`ffn{rows}`): the ANE service keeps intermediate
//! memory per program, so one multi-function program is far cheaper than one program per count.
//! Every function reads the leading `rows` columns of surfaces sized for `max_rows` tokens.
//!
//! Layout: every tensor is `[1, 1, channels, tokens]`, one channel per surface row.
//!
//! Inputs (int8 values are dequantized by 1/128 against fp16 overflow; the scales carry 128 back):
//! - `x{k}`: int8 `[segment, max_rows]`, the FFN input's channels `k*segment..`, quantized per
//!   token; `tx`: fp16 `[1, max_rows]`, each token's scale ×128.
//! - `wg{k}`, `wu{k}`: int8 `[intermediate, segment]`, gate/up (value) weights quantized per row;
//!   `sg`, `su`: fp16 `[intermediate, 1]`, their row scales ×128.
//! - `wd{i}`: int8 `[hidden, segment]`, down weights quantized per row; `sd`: fp16 `[hidden, 1]`.
//!
//! The function computes, per token t (real values):
//! ```text
//! gate = R_gate(Wg x), value = R_value(Wu x)          (R_*: optional block rotations)
//! h = R(silu(gate) * value)                            (R: the intermediate block rotation)
//! out = Wd quantize_per_token(h)
//! ```
//! Output `y`: fp16 `[hidden + 1, max_rows]`. Row `hidden` holds each token's intermediate scale;
//! the value scale `tx` is never applied on the ANE (it keeps `h` in fp16 range), so the real
//! output is `y[c][t] * y[hidden][t] * tx[t]`.
//!
//! SiLU is `g/2 * (1 + tanh(g/2))`: through the ANE's sigmoid, small gate pre-activations lose
//! accuracy (Splash measured 6.5% RMS against 1.6% through tanh).
//!
//! Adapted from Splash `runtime/ops/AneFfn.cpp` (commit 031a25a, Apache License 2.0).

use std::fmt::Write;

use half::f16;

use super::{
    AneElement,
    surface::{mil_buffer_type, mil_strides, row_stride_bytes},
};

/// The divisor of every int8 value, against fp16 overflow.
pub const INT8_UNIT: f64 = 128.0;
/// The int8 peak of the per-token requantization.
pub const INT8_PEAK: f64 = 127.0;
/// The floor of a token's intermediate peak, against a zero scale.
pub const INTERMEDIATE_FLOOR: f64 = 1.0 / 512.0;

#[derive(Debug, Clone, PartialEq)]
pub struct FfnShape {
    pub hidden: u32,
    pub intermediate: u32,
    /// Width of each matmul's reduction segment (input and down).
    pub segment: u32,
    /// The row counts a function is compiled for, ascending.
    pub rows: Vec<u32>,
    /// The token capacity of the token-major surfaces.
    pub max_rows: u32,
    /// The intermediate rotation's block (a grouped 1x1 conv), and its blocks `[intermediate,
    /// rotation_block]` row-major: `h'[c] = sum_i rotation[c][i] * h[block(c) + i]`.
    pub rotation_block: u32,
    pub rotation: Vec<f16>,
    /// The up projection's output rotation block (0: none), applied to the value and the gate
    /// after their row scales, before SiLU; `[intermediate, up_rotation_block]` each.
    pub up_rotation_block: u32,
    pub value_rotation: Vec<f16>,
    pub gate_rotation: Vec<f16>,
    /// Whether each weight matrix is one surface (`wg`, `wu` `[intermediate, hidden]`, `wd`
    /// `[hidden, intermediate]`) the program slices into segments, rather than one per segment.
    pub single_weight_surfaces: bool,
    /// Whether the input is one surface `x` `[hidden, max_rows]` rather than one per segment.
    pub single_input_surface: bool,
    /// Whether the input `x` is fp16 (the int8 codes over 128, unrounded) instead of int8 codes;
    /// needs `single_input_surface`.
    pub half_input: bool,
    /// Whether the intermediate `h` reaches the down projection unrounded (fp16) instead of as
    /// per-token int8 codes.
    pub half_intermediate: bool,
}

/// A surface a program reads or writes: its name, rows and width (elements).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FfnSurfaceSpec {
    pub name: String,
    pub rows: u32,
    pub width: u32,
    pub element: AneElement,
    /// Whether the function reads only the leading `rows` (tokens) of each row.
    pub token_major: bool,
}

impl FfnSurfaceSpec {
    fn new(
        name: String,
        rows: u32,
        width: u32,
        element: AneElement,
        token_major: bool,
    ) -> Self {
        Self {
            name,
            rows,
            width,
            element,
            token_major,
        }
    }

    fn row_stride_elements(&self) -> usize {
        row_stride_bytes(self.width, self.element) / self.element.bytes()
    }
}

impl FfnShape {
    pub fn input_segments(&self) -> u32 {
        self.hidden / self.segment
    }

    pub fn down_segments(&self) -> u32 {
        self.intermediate / self.segment
    }

    pub fn function_name(rows: u32) -> String {
        format!("ffn{rows}")
    }

    /// The smallest function row count holding `tokens`.
    pub fn rows_for(
        &self,
        tokens: u32,
    ) -> Option<u32> {
        self.rows.iter().copied().find(|rows| *rows >= tokens)
    }

    /// The program's inputs, in parameter order.
    pub fn inputs(&self) -> Vec<FfnSurfaceSpec> {
        let mut inputs = Vec::new();
        if self.single_input_surface {
            let element = if self.half_input {
                AneElement::Float16
            } else {
                AneElement::Int8
            };
            inputs.push(FfnSurfaceSpec::new("x".to_string(), self.hidden, self.max_rows, element, true));
        } else {
            for k in 0..self.input_segments() {
                inputs.push(FfnSurfaceSpec::new(format!("x{k}"), self.segment, self.max_rows, AneElement::Int8, true));
            }
        }
        inputs.push(FfnSurfaceSpec::new("tx".to_string(), 1, self.max_rows, AneElement::Float16, true));
        if self.single_weight_surfaces {
            for projection in ["wg", "wu"] {
                inputs.push(FfnSurfaceSpec::new(
                    projection.to_string(),
                    self.intermediate,
                    self.hidden,
                    AneElement::Int8,
                    false,
                ));
            }
        } else {
            for k in 0..self.input_segments() {
                for projection in ["wg", "wu"] {
                    inputs.push(FfnSurfaceSpec::new(
                        format!("{projection}{k}"),
                        self.intermediate,
                        self.segment,
                        AneElement::Int8,
                        false,
                    ));
                }
            }
        }
        for scale in ["sg", "su"] {
            inputs.push(FfnSurfaceSpec::new(scale.to_string(), self.intermediate, 1, AneElement::Float16, false));
        }
        if self.single_weight_surfaces {
            inputs.push(FfnSurfaceSpec::new("wd".to_string(), self.hidden, self.intermediate, AneElement::Int8, false));
        } else {
            for i in 0..self.down_segments() {
                inputs.push(FfnSurfaceSpec::new(format!("wd{i}"), self.hidden, self.segment, AneElement::Int8, false));
            }
        }
        inputs.push(FfnSurfaceSpec::new("sd".to_string(), self.hidden, 1, AneElement::Float16, false));
        inputs
    }

    pub fn output(&self) -> FfnSurfaceSpec {
        FfnSurfaceSpec::new("y".to_string(), self.hidden + 1, self.max_rows, AneElement::Float16, true)
    }

    fn validate(&self) {
        assert!(self.segment > 0 && self.hidden.is_multiple_of(self.segment), "hidden must be whole segments");
        assert!(self.intermediate.is_multiple_of(self.segment), "intermediate must be whole segments");
        assert!(!self.rows.is_empty() && self.rows.is_sorted(), "rows must be ascending");
        assert!(self.rows.iter().all(|rows| *rows <= self.max_rows), "rows exceed max_rows");
        assert!(
            self.rotation_block > 0 && self.intermediate.is_multiple_of(self.rotation_block),
            "intermediate must be whole rotation blocks"
        );
        assert_eq!(self.rotation.len(), (self.intermediate * self.rotation_block) as usize);
        if self.up_rotation_block > 0 {
            assert!(self.intermediate.is_multiple_of(self.up_rotation_block));
            let size = (self.intermediate * self.up_rotation_block) as usize;
            assert_eq!(self.value_rotation.len(), size);
            assert_eq!(self.gate_rotation.len(), size);
        }
    }
}

/// A MIL fp16 literal in hexadecimal float notation (exact for the double given).
fn fp16_literal(value: f64) -> String {
    format!("fp16({})", hex_float(value))
}

fn hex_float(value: f64) -> String {
    if value == 0.0 {
        return "0x0p+0".to_string();
    }
    let sign = if value < 0.0 {
        "-"
    } else {
        ""
    };
    let bits = value.abs().to_bits();
    let exponent = ((bits >> 52) & 0x7ff) as i64 - 1023;
    let mantissa = bits & ((1u64 << 52) - 1);
    if mantissa == 0 {
        return format!("{sign}0x1p{exponent:+}");
    }
    let digits = format!("{mantissa:013x}");
    format!("{sign}0x1.{}p{exponent:+}", digits.trim_end_matches('0'))
}

fn dims(
    rows: u64,
    width: u64,
) -> String {
    format!("[1, 1, {rows}, {width}]")
}

fn tensor(
    element: &str,
    rows: u64,
    width: u64,
) -> String {
    format!("tensor<{element}, {}>", dims(rows, width))
}

/// The constants of a program and their offsets in its blob (Core ML's weight-blob layout: a
/// 64-byte header holding the count, then per constant a 64-byte record and its 64-aligned data).
struct Constants {
    blob: Vec<u8>,
    offsets: Vec<u64>,
}

fn constant_blob(constants: &[&[f16]]) -> Constants {
    const HEADER: usize = 64;
    const RECORD: usize = 64;
    let mut blob = vec![0u8; HEADER];
    blob[0..4].copy_from_slice(&(constants.len() as u32).to_le_bytes());
    blob[4..8].copy_from_slice(&2u32.to_le_bytes());
    let mut offsets = Vec::new();
    for values in constants {
        let record = blob.len();
        let data = record + RECORD;
        let size = values.len() * 2;
        blob.resize((data + size).next_multiple_of(64), 0);
        blob[record..record + 4].copy_from_slice(&0xdead_beefu32.to_le_bytes());
        blob[record + 4..record + 8].copy_from_slice(&1u32.to_le_bytes()); // fp16
        blob[record + 8..record + 16].copy_from_slice(&(size as u64).to_le_bytes());
        blob[record + 16..record + 24].copy_from_slice(&(data as u64).to_le_bytes());
        blob[data..data + size].copy_from_slice(bytemuck::cast_slice(values));
        offsets.push(record as u64);
    }
    Constants {
        blob,
        offsets,
    }
}

struct FunctionWriter<'a> {
    shape: &'a FfnShape,
    rows: u32,
    body: String,
}

impl FunctionWriter<'_> {
    fn line(
        &mut self,
        text: &str,
    ) {
        let _ = writeln!(self.body, "        {text};");
    }

    fn f16(
        &mut self,
        name: &str,
        channels: u32,
        expression: &str,
    ) {
        let declaration = format!("{} {name} = {expression}", tensor("fp16", channels.into(), self.rows.into()));
        self.line(&declaration);
    }

    /// `name = dequantize(weights) @ values`, where `weights` is int8 `[height, width]` or, with
    /// single weight surfaces, segment `segment` of the projection's one surface.
    fn matmul(
        &mut self,
        name: &str,
        projection: &str,
        segment: u32,
        values: &str,
        height: u32,
        width: u32,
    ) {
        let unit = fp16_literal(1.0 / INT8_UNIT);
        let weights = format!("{projection}{segment}");
        let source = if self.shape.single_weight_surfaces {
            let begin = segment * width;
            self.line(&format!(
                "{} {weights}_s = slice_by_size(x = {projection}_t, begin = tensor<int32, [4]>([0, 0, 0, {begin}]), size = tensor<int32, [4]>({}))",
                tensor("int8", height.into(), width.into()),
                dims(height.into(), width.into())
            ));
            format!("{weights}_s")
        } else {
            format!("{weights}_t")
        };
        let dequantized = format!(
            "{} {weights}_d = dequantize(input = {source}, scale = {unit})",
            tensor("fp16", height.into(), width.into())
        );
        self.line(&dequantized);
        self.f16(
            name,
            height,
            &format!("matmul(transpose_x = bool(false), transpose_y = bool(false), x = {weights}_d, y = {values})"),
        );
    }

    /// Sums `prefix0..prefix{terms-1}`, returning the name of the sum.
    fn sum(
        &mut self,
        prefix: &str,
        terms: u32,
        channels: u32,
    ) -> String {
        let mut total = format!("{prefix}0");
        for term in 1..terms {
            let next = format!("{prefix}_sum{term}");
            self.f16(&next, channels, &format!("add(x = {total}, y = {prefix}{term})"));
            total = next;
        }
        total
    }

    /// A grouped 1x1 conv of `input` `[channels, rows]` by the constant at `offset`.
    fn block_rotation(
        &mut self,
        output: &str,
        input: &str,
        constant: &str,
        block: u32,
        offset: u64,
    ) {
        let channels = self.shape.intermediate;
        let rows = self.rows;
        self.line(&format!(
            "tensor<fp16, [1, {channels}, 1, {rows}]> {input}_4 = reshape(x = {input}, shape = tensor<int32, [4]>([1, {channels}, 1, {rows}]))"
        ));
        self.line(&format!(
            "tensor<fp16, [{channels}, {block}, 1, 1]> {constant} = const()[name = string(\"{constant}\"), val = tensor<fp16, [{channels}, {block}, 1, 1]>(BLOBFILE(path = string(\"@model_path/weights.bin\"), offset = uint64({offset})))]"
        ));
        self.line(&format!(
            "tensor<fp16, [1, {channels}, 1, {rows}]> {output}_4 = conv(dilations = tensor<int32, [2]>([1, 1]), groups = int32({}), pad = tensor<int32, [4]>([0, 0, 0, 0]), pad_type = string(\"valid\"), strides = tensor<int32, [2]>([1, 1]), weight = {constant}, x = {input}_4)",
            channels / block
        ));
        self.f16(
            output,
            channels,
            &format!("reshape(x = {output}_4, shape = tensor<int32, [4]>({}))", dims(channels.into(), rows.into())),
        );
    }

    fn write(
        mut self,
        offsets: &[u64],
    ) -> String {
        let shape = self.shape;
        let rows = self.rows;
        let channels = shape.intermediate;
        let unit = fp16_literal(1.0 / INT8_UNIT);

        let mut parameters = Vec::new();
        for input in shape.inputs() {
            let width = if input.token_major {
                rows
            } else {
                input.width
            };
            parameters.push(format!(
                "{} {}",
                mil_buffer_type(input.element, input.rows, width, input.row_stride_elements()),
                input.name
            ));
            self.line(&format!(
                "{} {}_t = tensor_buffer_to_tensor<ios17>(input = {})",
                tensor(input.element.mil_type(), input.rows.into(), width.into()),
                input.name,
                input.name
            ));
        }

        if shape.single_input_surface {
            let x_d = if shape.half_input {
                "x_t"
            } else {
                self.f16("x_d", shape.hidden, &format!("dequantize(input = x_t, scale = {unit})"));
                "x_d"
            };
            for k in 0..shape.input_segments() {
                let begin = k * shape.segment;
                self.f16(
                    &format!("x{k}_d"),
                    shape.segment,
                    &format!(
                        "slice_by_size(x = {x_d}, begin = tensor<int32, [4]>([0, 0, {begin}, 0]), size = tensor<int32, [4]>({}))",
                        dims(shape.segment.into(), rows.into())
                    ),
                );
            }
        } else {
            for k in 0..shape.input_segments() {
                self.f16(&format!("x{k}_d"), shape.segment, &format!("dequantize(input = x{k}_t, scale = {unit})"));
            }
        }
        for projection in ["g", "u"] {
            for k in 0..shape.input_segments() {
                self.matmul(
                    &format!("{projection}m{k}"),
                    &format!("w{projection}"),
                    k,
                    &format!("x{k}_d"),
                    channels,
                    shape.segment,
                );
            }
            let total = self.sum(&format!("{projection}m"), shape.input_segments(), channels);
            self.f16(&format!("{projection}s"), channels, &format!("mul(x = {total}, y = s{projection}_t)"));
        }

        let (gate, value) = if shape.up_rotation_block > 0 {
            self.block_rotation("gr", "gs", "gate_rotation", shape.up_rotation_block, offsets[2]);
            self.block_rotation("ur", "us", "value_rotation", shape.up_rotation_block, offsets[1]);
            ("gr", "ur")
        } else {
            ("gs", "us")
        };

        self.f16("gt", channels, &format!("mul(x = {gate}, y = tx_t)"));
        self.f16("gh", channels, &format!("mul(x = gt, y = {})", fp16_literal(0.5)));
        self.f16("th", channels, "tanh(x = gh)");
        self.f16("tp", channels, &format!("add(x = th, y = {})", fp16_literal(1.0)));
        self.f16("silu", channels, "mul(x = gh, y = tp)");
        self.f16("h", channels, &format!("mul(x = silu, y = {value})"));
        self.block_rotation("hr", "h", "rotation", shape.rotation_block, offsets[0]);

        self.f16("habs", channels, "abs(x = hr)");
        self.f16("peak", 1, "reduce_max(x = habs, axes = tensor<int32, [1]>([2]), keep_dims = bool(true))");
        self.f16("floor", 1, &format!("maximum(x = peak, y = {})", fp16_literal(INTERMEDIATE_FLOOR)));
        self.f16("inverse", 1, &format!("real_div(x = {}, y = floor)", fp16_literal(INT8_PEAK)));
        self.f16("hs", channels, "mul(x = hr, y = inverse)");
        if shape.half_intermediate {
            self.f16("hd", channels, &format!("mul(x = hs, y = {unit})"));
        } else {
            self.line(&format!(
                "{} hq = quantize(input = hs, scale = fp16(1), output_dtype = string(\"int8\"))",
                tensor("int8", channels.into(), rows.into())
            ));
            self.f16("hd", channels, &format!("dequantize(input = hq, scale = {unit})"));
        }
        self.f16("hscale", 1, &format!("mul(x = floor, y = {})", fp16_literal(INT8_UNIT / INT8_PEAK)));

        for i in 0..shape.down_segments() {
            let begin = i * shape.segment;
            self.f16(
                &format!("hd{i}"),
                shape.segment,
                &format!(
                    "slice_by_size(x = hd, begin = tensor<int32, [4]>([0, 0, {begin}, 0]), size = tensor<int32, [4]>({}))",
                    dims(shape.segment.into(), rows.into())
                ),
            );
            self.matmul(&format!("dm{i}"), "wd", i, &format!("hd{i}"), shape.hidden, shape.segment);
        }
        let total = self.sum("dm", shape.down_segments(), shape.hidden);
        self.f16("ds", shape.hidden, &format!("mul(x = {total}, y = sd_t)"));
        self.f16("yt", shape.hidden + 1, "concat(axis = int32(2), interleave = bool(false), values = (ds, hscale))");
        let output = shape.output();
        let stride = output.row_stride_elements();
        self.line(&format!(
            "{} y = tensor_to_tensor_buffer<ios17>(input = yt, interleave_factors = tensor<uint8, [4]>([1, 1, 1, 1]), strides = tensor<int64, [4]>({}))",
            mil_buffer_type(output.element, output.rows, rows, stride),
            mil_strides(output.rows, stride)
        ));

        format!(
            "    func {}<ios18>({}) {{\n{}    }} -> (y);\n",
            FfnShape::function_name(rows),
            parameters.join(", "),
            self.body
        )
    }
}

/// The program's MIL text and its `weights.bin` blob.
pub fn ffn_program(shape: &FfnShape) -> (String, Vec<u8>) {
    shape.validate();
    let mut constants: Vec<&[f16]> = vec![&shape.rotation];
    if shape.up_rotation_block > 0 {
        constants.push(&shape.value_rotation);
        constants.push(&shape.gate_rotation);
    }
    let constants = constant_blob(&constants);
    let mut text = String::from("program(1.3)\n{\n");
    for rows in &shape.rows {
        let writer = FunctionWriter {
            shape,
            rows: *rows,
            body: String::new(),
        };
        text += &writer.write(&constants.offsets);
    }
    text += "}\n";
    (text, constants.blob)
}

/// A single W8A8 matmul `y = s * (W x)` (`W`: int8 `[output, input]` with one scale per row, `x`:
/// int8 per token), the program of one linear layer shape. Like the FFN program it holds a function
/// per row count (`lin{rows}`). The weight, scale and output surfaces have `max_output` rows, and a
/// function of fewer `output` rows reads and writes their leading rows, so one set of surfaces
/// serves every shape up to `max_output`.
///
/// Inputs `x` (int8 `[input, max_rows]`), `w` (int8 `[max_output, input]`), `s` (fp16 `[max_output,
/// 1]`, the row scales ×128); output `y` (fp16 `[max_output, max_rows]`). The token scales (the
/// FFN's `tx`) are not read: the real output is `y[c][t] * tx[t]`, which the GPU unpack applies.
#[derive(Debug, Clone, PartialEq)]
pub struct LinearShape {
    pub input: u32,
    pub output: u32,
    pub max_output: u32,
    /// Width of each matmul's reduction segment.
    pub segment: u32,
    /// The row counts a function is compiled for, ascending.
    pub rows: Vec<u32>,
    pub max_rows: u32,
    /// The element type of the activations `x` and of the weights `w`: int8 (dequantized by 1/128)
    /// or fp16 (read as they are).
    pub activations: AneElement,
    pub weights: AneElement,
}

impl LinearShape {
    pub fn function_name(rows: u32) -> String {
        format!("lin{rows}")
    }

    /// The smallest function row count holding `tokens`.
    pub fn rows_for(
        &self,
        tokens: u32,
    ) -> Option<u32> {
        self.rows.iter().copied().find(|rows| *rows >= tokens)
    }

    pub fn segments(&self) -> u32 {
        self.input / self.segment
    }

    /// The program's inputs, in parameter order.
    pub fn inputs(&self) -> Vec<FfnSurfaceSpec> {
        vec![
            FfnSurfaceSpec::new("x".to_string(), self.input, self.max_rows, self.activations, true),
            FfnSurfaceSpec::new("w".to_string(), self.max_output, self.input, self.weights, false),
            FfnSurfaceSpec::new("s".to_string(), self.max_output, 1, AneElement::Float16, false),
        ]
    }

    /// The surface of the token scales `tx` (fp16 `[1, max_rows]`, ×128) the GPU pack writes.
    pub fn token_scales(&self) -> FfnSurfaceSpec {
        FfnSurfaceSpec::new("tx".to_string(), 1, self.max_rows, AneElement::Float16, true)
    }

    pub fn output_spec(&self) -> FfnSurfaceSpec {
        FfnSurfaceSpec::new("y".to_string(), self.max_output, self.max_rows, AneElement::Float16, true)
    }

    fn validate(&self) {
        assert!(self.segment > 0 && self.input.is_multiple_of(self.segment), "input must be whole segments");
        assert!(self.output > 0 && self.output <= self.max_output, "output exceeds max_output");
        assert!(!self.rows.is_empty() && self.rows.is_sorted(), "rows must be ascending");
        assert!(self.rows.iter().all(|rows| *rows <= self.max_rows), "rows exceed max_rows");
    }

    fn function(
        &self,
        rows: u32,
    ) -> String {
        let (input, output, segment) = (self.input, self.output, self.segment);
        let unit = fp16_literal(1.0 / INT8_UNIT);
        let mut body = String::new();
        let mut line = |text: String| {
            let _ = writeln!(body, "        {text};");
        };

        let mut parameters = Vec::new();
        for spec in self.inputs() {
            let width = if spec.token_major {
                rows
            } else {
                spec.width
            };
            parameters.push(format!(
                "{} {}",
                mil_buffer_type(spec.element, spec.rows, width, spec.row_stride_elements()),
                spec.name
            ));
            line(format!(
                "{} {}_t = tensor_buffer_to_tensor<ios17>(input = {})",
                tensor(spec.element.mil_type(), spec.rows.into(), width.into()),
                spec.name,
                spec.name
            ));
        }
        let activations = if self.activations == AneElement::Int8 {
            line(format!(
                "{} x_d = dequantize(input = x_t, scale = {unit})",
                tensor("fp16", input.into(), rows.into())
            ));
            "x_d"
        } else {
            "x_t"
        };

        for k in 0..self.segments() {
            let begin = k * segment;
            line(format!(
                "{} x{k}_d = slice_by_size(x = {activations}, begin = tensor<int32, [4]>([0, 0, {begin}, 0]), size = tensor<int32, [4]>({}))",
                tensor("fp16", segment.into(), rows.into()),
                dims(segment.into(), rows.into())
            ));
            let weights_name = if self.weights == AneElement::Int8 {
                format!("w{k}_s")
            } else {
                format!("w{k}_d")
            };
            line(format!(
                "{} {weights_name} = slice_by_size(x = w_t, begin = tensor<int32, [4]>([0, 0, 0, {begin}]), size = tensor<int32, [4]>({}))",
                tensor(self.weights.mil_type(), output.into(), segment.into()),
                dims(output.into(), segment.into())
            ));
            if self.weights == AneElement::Int8 {
                line(format!(
                    "{} w{k}_d = dequantize(input = w{k}_s, scale = {unit})",
                    tensor("fp16", output.into(), segment.into())
                ));
            }
            line(format!(
                "{} m{k} = matmul(transpose_x = bool(false), transpose_y = bool(false), x = w{k}_d, y = x{k}_d)",
                tensor("fp16", output.into(), rows.into())
            ));
        }
        let mut total = "m0".to_string();
        for k in 1..self.segments() {
            line(format!("{} m_sum{k} = add(x = {total}, y = m{k})", tensor("fp16", output.into(), rows.into())));
            total = format!("m_sum{k}");
        }
        line(format!(
            "{} s_n = slice_by_size(x = s_t, begin = tensor<int32, [4]>([0, 0, 0, 0]), size = tensor<int32, [4]>({}))",
            tensor("fp16", output.into(), 1),
            dims(output.into(), 1)
        ));
        line(format!("{} ys = mul(x = {total}, y = s_n)", tensor("fp16", output.into(), rows.into())));
        let spec = self.output_spec();
        let stride = spec.row_stride_elements();
        line(format!(
            "{} y = tensor_to_tensor_buffer<ios17>(input = ys, interleave_factors = tensor<uint8, [4]>([1, 1, 1, 1]), strides = tensor<int64, [4]>({}))",
            mil_buffer_type(spec.element, output, rows, stride),
            mil_strides(output, stride)
        ));
        format!("    func {}<ios18>({}) {{\n{body}    }} -> (y);\n", Self::function_name(rows), parameters.join(", "))
    }
}

/// The MIL text of a linear program; it has no constants, so its `weights.bin` is empty.
pub fn linear_program(shape: &LinearShape) -> (String, Vec<u8>) {
    shape.validate();
    let mut text = String::from("program(1.3)\n{\n");
    for rows in &shape.rows {
        text += &shape.function(*rows);
    }
    text += "}\n";
    (text, constant_blob(&[]).blob)
}
