use std::{
    sync::{
        Arc,
        atomic::{AtomicU8, Ordering},
    },
    time::{Duration, Instant},
};

use half::f16;
use metal::{MTLDeviceExt, MTLSharedEvent};
use uzu_engine_macros::uzu_test;

use crate::{
    backends::metal::{
        Metal,
        ane::{
            AneElement, AneProgram, AneRuntime, AneSurface,
            fake::run_ffn_reference,
            mil::{FfnShape, INT8_UNIT, ffn_program},
        },
        context::MetalContext,
    },
    tests::helpers::create_context,
};

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn uniform(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 24) as f32
    }

    fn normal(&mut self) -> f32 {
        (self.uniform() + self.uniform() + self.uniform() + self.uniform() - 2.0) * 1.732_050_8
    }

    fn sign(&mut self) -> f32 {
        if self.next() & 1 == 1 {
            -1.0
        } else {
            1.0
        }
    }
}

fn runtime() -> Option<Arc<AneRuntime>> {
    match AneRuntime::new() {
        Ok(runtime) => Some(runtime),
        Err(error) => {
            println!("skipping: {error}");
            None
        },
    }
}

/// `[channels, block]` row-major blocks of `H_block * diag(signs) / sqrt(block)`.
fn signed_hadamard(
    channels: u32,
    block: u32,
    rng: &mut Rng,
) -> Vec<f16> {
    let signs: Vec<f32> = (0..block).map(|_| rng.sign()).collect();
    let norm = 1.0 / (block as f32).sqrt();
    (0..channels)
        .flat_map(|channel| {
            let row = channel % block;
            let signs = &signs;
            (0..block).map(move |column| {
                let hadamard = if (row & column).count_ones() % 2 == 1 {
                    -1.0
                } else {
                    1.0
                };
                f16::from_f32(hadamard * signs[column as usize] * norm)
            })
        })
        .collect()
}

#[derive(Clone, Copy, Debug)]
struct Layout {
    single_weight_surfaces: bool,
    single_input_surface: bool,
}

const SEGMENTED: Layout = Layout {
    single_weight_surfaces: false,
    single_input_surface: false,
};

fn shape(
    hidden: u32,
    intermediate: u32,
    segment: u32,
    rows: Vec<u32>,
    rotation_block: u32,
    up_rotation_block: u32,
    layout: Layout,
    rng: &mut Rng,
) -> FfnShape {
    let max_rows = *rows.last().unwrap();
    let (value_rotation, gate_rotation) = if up_rotation_block > 0 {
        (signed_hadamard(intermediate, up_rotation_block, rng), signed_hadamard(intermediate, up_rotation_block, rng))
    } else {
        (Vec::new(), Vec::new())
    };
    FfnShape {
        hidden,
        intermediate,
        segment,
        rows,
        max_rows,
        rotation_block,
        rotation: signed_hadamard(intermediate, rotation_block, rng),
        up_rotation_block,
        value_rotation,
        gate_rotation,
        single_weight_surfaces: layout.single_weight_surfaces,
        single_input_surface: layout.single_input_surface,
    }
}

struct Surfaces {
    inputs: Vec<(String, AneSurface)>,
    output: AneSurface,
}

impl Surfaces {
    fn named(&self) -> Vec<(&str, &AneSurface)> {
        self.inputs.iter().map(|(name, surface)| (name.as_str(), surface)).collect()
    }

    fn input_mut(
        &mut self,
        name: &str,
    ) -> &mut AneSurface {
        &mut self.inputs.iter_mut().find(|(input, _)| input == name).unwrap().1
    }

    fn input(
        &self,
        name: &str,
    ) -> &AneSurface {
        &self.inputs.iter().find(|(input, _)| input == name).unwrap().1
    }
}

fn create_surfaces(
    context: &MetalContext,
    shape: &FfnShape,
) -> Surfaces {
    let inputs = shape
        .inputs()
        .into_iter()
        .map(|spec| (spec.name.clone(), AneSurface::new(context, spec.rows, spec.width, spec.element).unwrap()))
        .collect();
    let output = shape.output();
    Surfaces {
        inputs,
        output: AneSurface::new(context, output.rows, output.width, output.element).unwrap(),
    }
}

/// Quantizes `rows` rows of `columns` normal values (standard deviation `deviation`) per row into
/// int8 codes written by `store(row, column, code)`; returns the row scales.
fn quantize_rows(
    rows: u32,
    columns: u32,
    deviation: f32,
    rng: &mut Rng,
    mut store: impl FnMut(u32, u32, i8),
) -> Vec<f32> {
    let mut scales = Vec::with_capacity(rows as usize);
    let mut values = vec![0.0f32; columns as usize];
    for row in 0..rows {
        for value in values.iter_mut() {
            *value = rng.normal() * deviation;
        }
        let peak = values.iter().fold(1e-12f32, |peak, value| peak.max(value.abs()));
        let scale = peak / 127.0;
        for (column, value) in values.iter().enumerate() {
            store(row, column as u32, (value / scale).round() as i8);
        }
        scales.push(scale);
    }
    scales
}

/// Fills every input with quantized random weights and activations.
fn fill(
    surfaces: &mut Surfaces,
    shape: &FfnShape,
    rng: &mut Rng,
) {
    let segment = shape.segment;
    let mut fill_weights = |prefix: &str, scale_name: &str, rows: u32, columns: u32, rng: &mut Rng| {
        let mut codes = vec![0i8; (rows * columns) as usize];
        let scales = quantize_rows(rows, columns, 1.0 / (columns as f32).sqrt(), rng, |row, column, code| {
            codes[(row * columns + column) as usize] = code;
        });
        // One surface per segment, or one surface holding every column.
        let (pieces, width) = if shape.single_weight_surfaces {
            (vec![prefix.to_string()], columns)
        } else {
            ((0..columns / segment).map(|k| format!("{prefix}{k}")).collect(), segment)
        };
        for (k, piece) in pieces.iter().enumerate() {
            let surface = surfaces.input_mut(piece);
            let stride = surface.row_stride_elements();
            let values = surface.as_i8_mut();
            for row in 0..rows {
                for column in 0..width {
                    values[row as usize * stride + column as usize] =
                        codes[(row * columns + k as u32 * width + column) as usize];
                }
            }
        }
        let surface = surfaces.input_mut(scale_name);
        let stride = surface.row_stride_elements();
        let values = surface.as_f16_mut();
        for (row, scale) in scales.iter().enumerate() {
            values[row * stride] = f16::from_f32(scale * INT8_UNIT as f32);
        }
    };
    fill_weights("wg", "sg", shape.intermediate, shape.hidden, rng);
    fill_weights("wu", "su", shape.intermediate, shape.hidden, rng);
    fill_weights("wd", "sd", shape.hidden, shape.intermediate, rng);

    // Activations: per token over the hidden channels, channel-major in the surfaces.
    let mut codes = vec![0i8; (shape.max_rows * shape.hidden) as usize];
    let scales = quantize_rows(shape.max_rows, shape.hidden, 1.0, rng, |token, channel, code| {
        codes[(token * shape.hidden + channel) as usize] = code;
    });
    let (pieces, height) = if shape.single_input_surface {
        (vec!["x".to_string()], shape.hidden)
    } else {
        ((0..shape.input_segments()).map(|k| format!("x{k}")).collect(), segment)
    };
    for (k, piece) in pieces.iter().enumerate() {
        let surface = surfaces.input_mut(piece);
        let stride = surface.row_stride_elements();
        let values = surface.as_i8_mut();
        for channel in 0..height {
            for token in 0..shape.max_rows {
                values[channel as usize * stride + token as usize] =
                    codes[(token * shape.hidden + k as u32 * height + channel) as usize];
            }
        }
    }
    let surface = surfaces.input_mut("tx");
    let values = surface.as_f16_mut();
    for (token, scale) in scales.iter().enumerate() {
        values[token] = f16::from_f32(scale * INT8_UNIT as f32);
    }
}

fn poison(surface: &mut AneSurface) {
    for value in surface.as_f16_mut() {
        *value = f16::NAN;
    }
}

/// The real output `y[c][t] * y[hidden][t] * tx[t]` of the leading `tokens` tokens.
fn real_output(
    output: &AneSurface,
    token_scales: &AneSurface,
    hidden: u32,
    tokens: u32,
) -> Vec<f32> {
    let mut values = Vec::with_capacity((hidden * tokens) as usize);
    for channel in 0..hidden {
        for token in 0..tokens {
            values.push(output.get(channel, token) * output.get(hidden, token) * token_scales.get(0, token));
        }
    }
    values
}

fn relative_rms(
    estimate: &[f32],
    reference: &[f32],
) -> (f64, usize) {
    let mut error = 0.0f64;
    let mut energy = 0.0f64;
    let mut nonfinite = 0;
    for (estimate, reference) in estimate.iter().zip(reference) {
        if !estimate.is_finite() {
            nonfinite += 1;
            continue;
        }
        error += f64::from(estimate - reference).powi(2);
        energy += f64::from(*reference).powi(2);
    }
    ((error / energy).sqrt(), nonfinite)
}

fn check_against_reference(
    up_rotation_block: u32,
    layout: Layout,
) {
    let Some(runtime) = runtime() else {
        return;
    };
    let context = create_context::<Metal>();
    let mut rng = Rng(0x2545_f491_4f6c_dd1d);
    let shape = shape(512, 1024, 256, vec![128, 256], 32, up_rotation_block, layout, &mut rng);
    let (mil, weights) = ffn_program(&shape);
    let program = AneProgram::new(runtime, &mil, &weights).unwrap();
    println!(
        "procedures: {:?}; compile {:.2}s (cached {}), load {:.2}s",
        program.procedures().iter().map(|procedure| procedure.function.as_str()).collect::<Vec<_>>(),
        program.compile_seconds(),
        program.was_cached(),
        program.load_seconds()
    );

    let mut surfaces = create_surfaces(&context, &shape);
    fill(&mut surfaces, &shape, &mut rng);
    let output_spec = shape.output();
    let mut reference = AneSurface::new(&context, output_spec.rows, output_spec.width, AneElement::Float16).unwrap();

    for rows in shape.rows.clone() {
        let procedure = program.procedure(&FfnShape::function_name(rows)).unwrap();
        poison(&mut surfaces.output);
        let binding = program.bind(procedure, &surfaces.named(), &surfaces.output).unwrap();
        program.evaluate(&binding).unwrap();
        run_ffn_reference(&shape, rows, &surfaces.named(), &mut reference);

        let tx = surfaces.input("tx");
        let (rms, nonfinite) = relative_rms(
            &real_output(&surfaces.output, tx, shape.hidden, rows),
            &real_output(&reference, tx, shape.hidden, rows),
        );
        println!("ffn{rows} (up rotation {up_rotation_block}, {layout:?}): rel RMS {rms:.5}, non-finite {nonfinite}");
        assert_eq!(nonfinite, 0);
        assert!(rms <= 0.015, "ANE vs reference rel RMS {rms}");

        // The function writes only the leading `rows` columns.
        for row in 0..=shape.hidden {
            for column in rows..shape.max_rows {
                assert!(surfaces.output.get(row, column).is_nan(), "ffn{rows} wrote column {column}");
            }
        }
    }
}

#[uzu_test]
fn ane_ffn_matches_reference() {
    check_against_reference(0, SEGMENTED);
}

#[uzu_test]
fn ane_ffn_with_up_rotation_matches_reference() {
    check_against_reference(32, SEGMENTED);
}

#[uzu_test]
fn ane_ffn_single_surfaces_match_reference() {
    check_against_reference(
        32,
        Layout {
            single_weight_surfaces: true,
            single_input_surface: false,
        },
    );
    check_against_reference(
        32,
        Layout {
            single_weight_surfaces: true,
            single_input_surface: true,
        },
    );
}

#[uzu_test]
fn ane_ffn_enqueue_waits_for_and_signals_shared_event() {
    let Some(runtime) = runtime() else {
        return;
    };
    let context = create_context::<Metal>();
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
    let shape = shape(512, 1024, 256, vec![128], 32, 32, SEGMENTED, &mut rng);
    let (mil, weights) = ffn_program(&shape);
    let program = AneProgram::new(runtime, &mil, &weights).unwrap();
    let mut surfaces = create_surfaces(&context, &shape);
    fill(&mut surfaces, &shape, &mut rng);
    poison(&mut surfaces.output);

    let event = context.device.new_shared_event().unwrap();
    event.set_signaled_value(0);
    let procedure = program.procedure("ffn128").unwrap();
    let binding = program.bind(procedure, &surfaces.named(), &surfaces.output).unwrap();
    // 0: pending, 1: succeeded, 2: failed
    let done = Arc::new(AtomicU8::new(0));
    let reported = done.clone();
    program
        .enqueue(
            &binding,
            &event,
            1,
            2,
            Box::new(move |success| {
                reported.store(
                    if success {
                        1
                    } else {
                        2
                    },
                    Ordering::SeqCst,
                );
            }),
        )
        .unwrap();

    std::thread::sleep(Duration::from_millis(100));
    assert_eq!(event.signaled_value(), 0, "the evaluation signalled before its wait was met");
    assert!(surfaces.output.get(0, 0).is_nan(), "the evaluation ran before its wait was met");
    assert_eq!(done.load(Ordering::SeqCst), 0);

    let released = Instant::now();
    event.set_signaled_value(1);
    while event.signaled_value() < 2 {
        assert!(released.elapsed() < Duration::from_secs(5), "the evaluation never signalled");
        std::thread::sleep(Duration::from_micros(200));
    }
    let latency = released.elapsed();
    while done.load(Ordering::SeqCst) == 0 {
        assert!(released.elapsed() < Duration::from_secs(5), "the completion handler never ran");
        std::thread::sleep(Duration::from_micros(200));
    }
    println!("enqueue: signalled {:.2} ms after its wait was met", latency.as_secs_f64() * 1e3);
    assert_eq!(done.load(Ordering::SeqCst), 1);
    assert!(surfaces.output.get(0, 0).is_finite());
}

// Ignored: compiles the 9B-shaped program (tens of seconds on first run).
#[uzu_test]
#[ignore]
fn ane_ffn_perf_9b_shape() {
    let Some(runtime) = runtime() else {
        return;
    };
    let context = create_context::<Metal>();
    let layouts = [
        SEGMENTED,
        Layout {
            single_weight_surfaces: true,
            single_input_surface: false,
        },
        Layout {
            single_weight_surfaces: true,
            single_input_surface: true,
        },
        SEGMENTED,
    ];
    for layout in layouts {
        let mut rng = Rng(0x1234_5678_9abc_def1);
        let shape = shape(4096, 12288, 2048, vec![256, 512, 768, 1024], 512, 32, layout, &mut rng);
        let (mil, weights) = ffn_program(&shape);
        let started = Instant::now();
        let program = AneProgram::new(runtime.clone(), &mil, &weights).unwrap();
        println!(
            "9B program {layout:?}: build {:.2}s (compile {:.2}s, cached {}, load {:.2}s)",
            started.elapsed().as_secs_f64(),
            program.compile_seconds(),
            program.was_cached(),
            program.load_seconds()
        );
        let mut surfaces = create_surfaces(&context, &shape);
        fill(&mut surfaces, &shape, &mut rng);
        for rows in shape.rows.clone() {
            let procedure = program.procedure(&FfnShape::function_name(rows)).unwrap();
            let binding = program.bind(procedure, &surfaces.named(), &surfaces.output).unwrap();
            for _ in 0..3 {
                program.evaluate(&binding).unwrap();
            }
            let mut times: Vec<f64> = (0..15)
                .map(|_| {
                    let started = Instant::now();
                    program.evaluate(&binding).unwrap();
                    started.elapsed().as_secs_f64() * 1e3
                })
                .collect();
            times.sort_by(f64::total_cmp);
            let operations = 2.0 * 3.0 * f64::from(shape.hidden) * f64::from(shape.intermediate) * f64::from(rows);
            println!(
                "  ffn{rows}: median {:.2} ms (min {:.2}), {:.1} TOPS",
                times[times.len() / 2],
                times[0],
                operations / (times[times.len() / 2] * 1e-3) / 1e12
            );
        }
    }
}
