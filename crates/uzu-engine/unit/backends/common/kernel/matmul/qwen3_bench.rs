use std::rc::Rc;
#[cfg(backend = "vulkan")]
use std::sync::Arc;

use criterion::{BenchmarkId, Criterion};
use futures::executor::block_on;
use half::bf16;
use num_traits::Float;
use uzu_engine_macros::uzu_bench;

use crate::{
    backends::common::{
        Backend, Context,
        gpu_types::quantization_method::QuantizationMethod,
        kernel::{Kernels, matmul::MatmulKernel},
    },
    data_type::DataTypeRepr,
    tests::{
        cold_pool::ColdPool,
        helpers::{create_buffer, for_each_non_cpu_backend},
        matmul::{
            QuantInput, iter_encode_loop, production_quant_buffers, production_quant_input, quant_arguments,
            qwen3_layer_shapes,
        },
        util::type_short_name,
    },
};
#[cfg(backend = "vulkan")]
use crate::{
    backends::{
        common::{
            CommandBufferEncoding, CommandBufferExecutable, CommandBufferPending,
            kernel::matmul::{ActivationFormat, MatmulShape},
        },
        cpu::Cpu,
        vulkan::{VkContext, VkContextCreateInfo, VkMatmulKernel, VkMatmulPlan, Vulkan},
    },
    data_type::DataType,
    tests::{
        assert::parity_mismatches,
        helpers::buffer_to_vec,
        matmul::{QuantBuffers, Shape, quant_tolerance, run_quant, time_encode_loop},
    },
};

/// Qwen3 linears as loaded models run them: the backend's weight layout, activation format and output transform. The
/// weights rotate through cold copies while every product writes one output, so each waits for the last as in decoding.
fn bench_qwen3_layers_typed<B: Backend, T: DataTypeRepr + Float>(
    c: &mut Criterion,
    context: &B::Context,
    label: &str,
    group_size: u32,
    bits: u32,
    quant_method: QuantizationMethod,
) {
    let new_matmul = || {
        <<B as Backend>::Kernels as Kernels>::MatmulKernel::new(context, T::data_type(), T::data_type(), T::data_type())
            .unwrap()
    };
    let mut group = c.benchmark_group(format!("{}/Kernel/Qwen3Layers/{}", type_short_name::<B>(), label));
    for (layer, shape) in qwen3_layer_shapes(bits) {
        let (m, k, n) = (shape.m, shape.k, shape.n);
        let mut problem = None;
        group.bench_function(BenchmarkId::from_parameter(format!("{layer}_{shape}")), |b| {
            let (input, matmul, buffers, output) = problem.get_or_insert_with(|| {
                let layout = new_matmul();
                let input = QuantInput::<T>::new(m, k, n, group_size, bits, quant_method, 42)
                    .with_group_output()
                    .with_output_rht(true);
                let input =
                    production_quant_input::<B, T>(context, &layout, input.clone()).map_or(input, |(input, _)| input);
                let input = Rc::new(input);
                let pool_input = input.clone();
                let buffers = ColdPool::new(input.weight_buffer_bytes(), move || {
                    production_quant_buffers::<B, T>(context, &layout, &pool_input)
                });
                println!("{layer}_{shape}: {} weight bytes per product", input.weight_buffer_bytes());
                (input, new_matmul(), buffers, create_buffer::<B, T>(context, (m as usize) * (n as usize)))
            });
            iter_encode_loop::<B, _>(context, b, |command_buffer| {
                let mut arguments = quant_arguments(buffers.next_mut(), input);
                arguments.output.values = &mut *output;
                matmul.encode(arguments, command_buffer).expect("encode qwen3 layer");
            });
        });
    }
}

#[uzu_bench]
fn bench_qwen3_layers(c: &mut Criterion) {
    for_each_non_cpu_backend!(|B| {
        let context = block_on(<B as Backend>::Context::new()).unwrap();
        bench_qwen3_layers_typed::<B, bf16>(
            c,
            &context,
            "ScaleBias_BF16_gs64_4b",
            64,
            4,
            QuantizationMethod::ScaleBias,
        );
        bench_qwen3_layers_typed::<B, bf16>(c, &context, "ZP_BF16_gs64_4b", 64, 4, QuantizationMethod::ScaleZeroPoint);
    });
}

/// Explicit calibration of the recorded checkpoints' products on Vulkan, each as its loaded model runs it. Every legal
/// plan qualifies against the CPU's product of the same codes and the explicit default's full output before any
/// timing; then the plans time against the default and Criterion samples the selected one, always by explicit plan. A
/// problem calibrates once, only when its benchmark runs; a stable winner across devices belongs in the default rules.
#[cfg(backend = "vulkan")]
#[uzu_bench]
fn bench_qwen3_calibration(c: &mut Criterion) {
    // The quantization group of each checkpoint whose header is recorded, by label prefix; others are not calibrated.
    const CHECKPOINT_GROUPS: [(&str, u32); 3] = [("0.8b_", 32), ("4b_", 32), ("27b_", 64)];
    let mut session = None;
    let mut group = c.benchmark_group("Vulkan/Calibration");
    for (label, shape) in qwen3_layer_shapes(4) {
        let Some(&(_, group_size)) = CHECKPOINT_GROUPS.iter().find(|(prefix, _)| label.starts_with(prefix)) else {
            continue;
        };
        let id = format!("{label}_{shape}");
        let mut problem = None;
        group.bench_function(BenchmarkId::from_parameter(&id), |b| {
            let context = session.get_or_insert_with(|| {
                VkContext::with_create_info(VkContextCreateInfo {
                    with_validation: false,
                    ..Default::default()
                })
                .expect("calibration context")
            });
            let (input, kernel, buffers, output, plan) = problem.get_or_insert_with(|| {
                calibrate(context.clone(), &id, label, group_size, shape)
                    .unwrap_or_else(|error| panic!("calibration {id} failed: {error}"))
            });
            iter_encode_loop::<Vulkan, _>(context, b, |command_buffer| {
                let mut arguments = quant_arguments(buffers.next_mut(), input);
                arguments.output.values = &mut *output;
                kernel.encode_plan(arguments, Some(*plan), command_buffer).expect("encode the selected plan");
            });
        });
    }
    group.finish();
}

/// Sets up `label`'s product at `shape` with its checkpoint's contract, qualifies every plan and selects one, printing
/// the evidence. Returns the input, the encoder, the cold weight copies, the one output every plan writes and the plan
/// to sample: the winner, else the default.
#[cfg(backend = "vulkan")]
fn calibrate(
    context: Arc<VkContext>,
    id: &str,
    label: &str,
    group_size: u32,
    shape: Shape,
) -> Result<
    (
        Rc<QuantInput<bf16>>,
        VkMatmulKernel,
        ColdPool<QuantBuffers<Vulkan, bf16>, impl FnMut() -> QuantBuffers<Vulkan, bf16> + use<>>,
        <Vulkan as Backend>::GlobalBuffer,
        VkMatmulPlan,
    ),
    String,
> {
    let (m, k, n) = (shape.m, shape.k, shape.n);
    let new_matmul = || VkMatmulKernel::new(&context, DataType::BF16, DataType::BF16, DataType::BF16).unwrap();
    // Linears load U4 zero-point codes with group-output metadata and, except the untied readout, an output transform
    // without bias, as A8 where the backend prepares it. Tied readouts multiply BF16 activations by the embedding
    // table's row-major codes with per-row metadata.
    let tied = label.ends_with("tied_readout");
    let input = QuantInput::<bf16>::new(m, k, n, group_size, 4, QuantizationMethod::ScaleZeroPoint, 42);
    let input = match (tied, label.ends_with("readout")) {
        (true, _) => input,
        (false, true) => input.with_group_output(),
        (false, false) => input.with_group_output().with_output_rht(false),
    };
    let layout = new_matmul();
    let (input, format) = match tied {
        true => (input, ActivationFormat::Bf16),
        false => production_quant_input::<Vulkan, bf16>(&context, &layout, input)
            .ok_or("the checkpoint linear's activation format is unsupported")?,
    };
    let input = Rc::new(input);
    let pool_input = input.clone();
    let pool_context = context.clone();
    let mut buffers = ColdPool::new(input.weight_buffer_bytes(), move || match tied {
        true => QuantBuffers::allocate(pool_context.as_ref(), &pool_input),
        false => production_quant_buffers::<Vulkan, bf16>(&pool_context, &layout, &pool_input),
    });
    let mut output = create_buffer::<Vulkan, bf16>(&context, (m as usize) * (n as usize));
    let mut kernel = new_matmul();
    let qualifying = buffers.next_mut();
    let weights_layout = qualifying.weights_layout;
    let matmul_shape = MatmulShape::from_arguments(&quant_arguments(qualifying, &input));
    let (baseline, plans, excluded) =
        kernel.candidates(&matmul_shape, weights_layout, &context).map_err(|error| error.to_string())?;
    // The CPU computes the first columns of the full-N dispatch, a full 1024 and N's tail past whole 128s; every plan's
    // full output must also match the default's.
    let columns = n.min(1024 + n % 128);
    println!("calibration {id}: {format:?} activations, {weights_layout:?} weights");
    println!("calibration {id} default {baseline:?}, CPU columns {columns} of {n}");
    for (plan, reason) in &excluded {
        println!("calibration {id} excluded {plan:?}: {reason}");
    }
    // A legal default leads the plans.
    if plans.first() != Some(&baseline) {
        return Err(format!("the default {baseline:?} is not legal"));
    }
    let others = &plans[1..];
    let quantization = input.prepared_a.as_ref().map(|prepared| prepared.quantization);
    let mut rejected: Vec<(VkMatmulPlan, String)> = Vec::new();
    let mut untuned = None;
    for (fixture, x) in fixtures(&input.x, k) {
        let mut bounded = input.columns(columns);
        bounded.x = x;
        if let Some(quantization) = quantization {
            bounded = bounded.with_prepared_a(quantization.scale_group_size(), quantization.sum_group_size());
        }
        let reference = run_quant::<Cpu, bf16>(&bounded);
        qualifying.upload_activations(&context, &bounded);
        // NaN-filled first, so an output a plan leaves unwritten fails.
        let mut run = |plan: VkMatmulPlan| {
            let mut command_buffer = context.create_command_buffer(None, None).map_err(|error| error.to_string())?;
            CommandBufferEncoding::encode_fill(&mut command_buffer, &mut output, 0xFF);
            let mut arguments = quant_arguments(&mut *qualifying, &input);
            arguments.output.values = &mut output;
            kernel.encode_plan(arguments, Some(plan), &mut command_buffer).map_err(|error| error.to_string())?;
            block_on(command_buffer.end_encoding().submit().wait_until_completed())
                .map_err(|error| error.to_string())?;
            Ok::<_, String>(buffer_to_vec::<Vulkan, bf16>(&context, &output))
        };
        let cpu = format!("{fixture} against the CPU");
        let default = match run(baseline) {
            Ok(default) => default,
            Err(error) => {
                untuned = Some(format!("{fixture}: the baseline fails, {error}"));
                break;
            },
        };
        if let Err(reason) = parity(&cpu, &reference, &first_columns(&default, n, columns), format) {
            untuned = Some(format!("the baseline fails qualification, {reason}"));
            break;
        }
        for &plan in others {
            if rejected.iter().any(|(other, _)| *other == plan) {
                continue;
            }
            let outcome = run(plan).and_then(|actual| {
                parity(&cpu, &reference, &first_columns(&actual, n, columns), format)?;
                parity(&format!("{fixture} against the baseline"), &default, &actual, format)
            });
            if let Err(reason) = outcome {
                rejected.push((plan, reason));
            }
        }
    }
    for (plan, reason) in &rejected {
        println!("calibration {id} rejected {plan:?}: {reason}");
    }
    let selected = match untuned {
        Some(reason) => {
            println!("calibration {id} untuned: {reason}");
            return Err(reason);
        },
        None => {
            let qualified = others.iter().filter(|&plan| rejected.iter().all(|(other, _)| other != plan));
            let qualified = qualified.copied().collect::<Vec<_>>();
            select(&context, id, &input, &mut kernel, &mut buffers, &mut output, baseline, &qualified)
        },
    };
    println!("calibration {id} selected {selected:?}, None being the default {baseline:?}");
    Ok((input, kernel, buffers, output, selected.unwrap_or(baseline)))
}

/// The qualified plan to run instead of the explicit `baseline`, from serialized batches over the cold weight copies
/// into the shared output, in GPU nanoseconds a product. Every plan warms first;
/// screening pairs each plan's sample with an adjacent baseline sample, plans interleaved within each round, and the
/// FINALISTS of lowest median ratio rerun so in the finals. The population of a finalist's statistics is its kept final
/// pairs: its median and IQR of plan/baseline ratios and its p95 against the paired baseline samples' p95.
#[cfg(backend = "vulkan")]
fn select(
    context: &<Vulkan as Backend>::Context,
    id: &str,
    input: &QuantInput<bf16>,
    kernel: &mut VkMatmulKernel,
    buffers: &mut ColdPool<QuantBuffers<Vulkan, bf16>, impl FnMut() -> QuantBuffers<Vulkan, bf16>>,
    output: &mut <Vulkan as Backend>::GlobalBuffer,
    baseline: VkMatmulPlan,
    qualified: &[VkMatmulPlan],
) -> Option<VkMatmulPlan> {
    const SCREENING_ROUNDS: usize = 5;
    const FINAL_ROUNDS: usize = 12;
    const FINALISTS: usize = 3;
    // A plan replaces the baseline only with a paired median gain of 3% beyond the ratios' IQR and p95 within 3%; a
    // baseline drifting 3% between the finals' halves invalidates them.
    const MARGIN: f64 = 0.03;
    let mut measure = |plan: VkMatmulPlan, products| {
        let time = time_encode_loop::<Vulkan, _>(context, id, products, |command_buffer| {
            let mut arguments = quant_arguments(buffers.next_mut(), input);
            arguments.output.values = &mut *output;
            kernel.encode_plan(arguments, Some(plan), command_buffer).expect("a qualified plan encodes");
        });
        time.as_nanos() as f64 / products as f64
    };
    if qualified.is_empty() {
        return None;
    }
    let baseline_ns = measure(baseline, 1);
    let products = (20_000_000.0 / baseline_ns.max(1.0)).ceil().clamp(1.0, 16.0) as u64;
    println!("calibration {id} {products} dependent products per sample, GPU ns a product; segment gaps excluded");
    let mut sample = |plan| measure(plan, products);
    let warmup = std::time::Instant::now();
    for (index, plan) in std::iter::once(baseline).chain(qualified.iter().copied()).cycle().enumerate() {
        sample(plan);
        if index >= qualified.len() && warmup.elapsed() >= std::time::Duration::from_secs(3) {
            break;
        }
    }
    println!("calibration {id} warmed {} plans for {:.3} s", qualified.len() + 1, warmup.elapsed().as_secs_f64());
    // (baseline, plan) pairs of each plan, alternating which runs first by round.
    let mut rounds = |plans: &[VkMatmulPlan], count: usize| {
        let mut pairs = vec![Vec::new(); plans.len()];
        for round in 0..count {
            for (&plan, pairs) in plans.iter().zip(&mut pairs) {
                pairs.push(match round % 2 {
                    0 => (sample(baseline), sample(plan)),
                    _ => {
                        let plan = sample(plan);
                        (sample(baseline), plan)
                    },
                });
            }
        }
        pairs
    };
    let ratios = |pairs: &[(f64, f64)]| pairs.iter().map(|&(baseline, plan)| plan / baseline).collect::<Vec<_>>();
    let screening = rounds(qualified, SCREENING_ROUNDS);
    let mut ranked = Vec::new();
    for (&plan, pairs) in qualified.iter().zip(&screening) {
        println!("calibration {id} screening {plan:?} (baseline, plan) ns {pairs:?}");
        ranked.push((quantile(&ratios(pairs), 0.5), plan));
    }
    ranked.sort_by(|left, right| left.0.total_cmp(&right.0));
    let finalists = ranked.iter().take(FINALISTS).map(|&(_, plan)| plan).collect::<Vec<_>>();
    for attempt in 1..=2 {
        let count = FINAL_ROUNDS * attempt;
        let finals = rounds(&finalists, count);
        let kept = finals.iter().map(|pairs| &pairs[1..]).collect::<Vec<_>>();
        let defaults = (0..count - 1).flat_map(|round| kept.iter().map(move |pairs| pairs[round].0));
        let defaults = defaults.collect::<Vec<_>>();
        let (early, late) = defaults.split_at(defaults.len() / 2);
        let drift = quantile(late, 0.5) / quantile(early, 0.5);
        let mut winner = None;
        for (&plan, &pairs) in finalists.iter().zip(&kept) {
            let paired = ratios(pairs);
            let (median, spread) = (quantile(&paired, 0.5), quantile(&paired, 0.75) - quantile(&paired, 0.25));
            let p95 = |side: fn(&(f64, f64)) -> f64| quantile(&pairs.iter().map(side).collect::<Vec<_>>(), 0.95);
            let p95_ratio = p95(|pair| pair.1) / p95(|pair| pair.0);
            println!(
                "calibration {id} final {attempt} {plan:?} median ratio {median:.4}, IQR {spread:.4}, p95 ratio \
                 {p95_ratio:.4}, (baseline, plan) ns {pairs:?}"
            );
            let accepted = median <= 1.0 - MARGIN && 1.0 - median > spread && p95_ratio <= 1.0 + MARGIN;
            if accepted && winner.is_none_or(|(best, _)| median < best) {
                winner = Some((median, plan));
            }
        }
        if (drift - 1.0).abs() <= MARGIN {
            return winner.map(|(_, plan)| plan);
        }
        println!("calibration {id} final {attempt}: the baseline drifted by a factor of {drift:.4}");
    }
    println!("calibration {id} inconclusive: the baseline drifted in both finals");
    None
}

/// The activations every plan qualifies on: zeros, the seeded ones with one 32 a row, alternating +-4, then the seeded
/// ones, which the qualifying weight copy keeps for timing.
#[cfg(backend = "vulkan")]
fn fixtures(
    seeded: &[bf16],
    k: u32,
) -> [(&'static str, Vec<bf16>); 4] {
    let mut outlier = seeded.to_vec();
    outlier.iter_mut().skip(k as usize / 2).step_by(k as usize).for_each(|value| *value = bf16::from_f32(32.0));
    let alternating = (0..seeded.len()).map(|index| bf16::from_f32(4.0 - 8.0 * (index % 2) as f32)).collect();
    [
        ("zero", vec![bf16::ZERO; seeded.len()]),
        ("outlier", outlier),
        ("alternating", alternating),
        ("seeded", seeded.to_vec()),
    ]
}

/// The first `columns` of each of `output`'s rows of `n`.
#[cfg(backend = "vulkan")]
fn first_columns(
    output: &[bf16],
    n: u32,
    columns: u32,
) -> Vec<bf16> {
    output.chunks(n as usize).flat_map(|row| &row[..columns as usize]).copied().collect()
}

/// Whether `actual` holds `expected` within the format's unchanged tolerance, else the mismatch count and the first.
#[cfg(backend = "vulkan")]
fn parity(
    what: &str,
    expected: &[bf16],
    actual: &[bf16],
    format: ActivationFormat,
) -> Result<(), String> {
    let (rel_tol, abs_tol) = quant_tolerance(format, 4, expected);
    let mut mismatches = parity_mismatches(expected, actual, rel_tol, abs_tol);
    match mismatches.next() {
        None => Ok(()),
        Some((index, expected, actual)) => Err(format!(
            "{what}: {} mismatches, the first at {index}: expected {expected}, got {actual}",
            1 + mismatches.count()
        )),
    }
}

/// The `q` quantile of `values` by nearest rank.
#[cfg(backend = "vulkan")]
fn quantile(
    values: &[f64],
    q: f64,
) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    sorted[(sorted.len() as f64 * q).ceil().max(1.0) as usize - 1]
}
