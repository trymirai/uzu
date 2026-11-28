use uzu_engine_macros::uzu_test;

use crate::{
    backends::{
        common::{Backend, BufferMut, BufferRef, CommandBufferEncoding, Context},
        metal::{
            Metal,
            command_buffer::{MetalCommandBufferEncoding, MetalCondition},
            metal_extensions::PredicateComparison,
        },
    },
    tests::helpers::{buffer_to_vec, create_buffer_with_data, create_context, submit_command_buffer},
};

const COMPARISONS: [(PredicateComparison, fn(&u32, &u32) -> bool); 6] = [
    (PredicateComparison::Equal, u32::eq),
    (PredicateComparison::NotEqual, u32::ne),
    (PredicateComparison::Greater, u32::gt),
    (PredicateComparison::GreaterEqual, u32::ge),
    (PredicateComparison::Less, u32::lt),
    (PredicateComparison::LessEqual, u32::le),
];

fn shift_left(
    command_buffer: &mut MetalCommandBufferEncoding,
    sequence: &mut <Metal as Backend>::GlobalGpuBuffer,
    scratch: &mut <Metal as Backend>::GlobalGpuBuffer,
) {
    let shifted_size = sequence.size() - size_of::<u32>();
    command_buffer.encode_copy((&*sequence).subrange(size_of::<u32>()..), (&mut *scratch).subrange_mut(..shifted_size));
    command_buffer.encode_copy((&*scratch).subrange(..shifted_size), (&mut *sequence).subrange_mut(..shifted_size));
}

fn shifted(
    initial: &[u32],
    iterations: usize,
) -> Vec<u32> {
    (0..initial.len()).map(|index| initial[(index + iterations).min(initial.len() - 1)]).collect()
}

#[uzu_test]
fn encode_if_compares_the_predicate_with_the_reference() {
    let context = create_context::<Metal>();
    for (comparison, compare) in COMPARISONS {
        for predicate_value in [0, 1, 2] {
            let predicate = create_buffer_with_data::<Metal, u32>(&context, &[predicate_value]);
            let mut output = create_buffer_with_data::<Metal, u32>(&context, &[0]);
            let mut command_buffer = context.create_command_buffer(None, None).unwrap();
            command_buffer.encode_if(&MetalCondition::new(&predicate, comparison, 1), |command_buffer| {
                command_buffer.encode_fill(&mut output, 0xFF);
            });
            submit_command_buffer(command_buffer);
            let expected = if compare(&predicate_value, &1) {
                u32::MAX
            } else {
                0
            };
            assert_eq!(
                buffer_to_vec::<Metal, u32>(&context, &output),
                [expected],
                "{predicate_value} {comparison:?} 1"
            );
        }
    }
}

#[uzu_test]
fn encode_if_else_runs_exactly_one_branch() {
    let context = create_context::<Metal>();
    for predicate_value in [0, 1] {
        let predicate = create_buffer_with_data::<Metal, u32>(&context, &[predicate_value]);
        let mut then_output = create_buffer_with_data::<Metal, u32>(&context, &[0]);
        let mut else_output = create_buffer_with_data::<Metal, u32>(&context, &[0]);
        let mut command_buffer = context.create_command_buffer(None, None).unwrap();
        command_buffer.encode_if_else(
            &MetalCondition::new(&predicate, PredicateComparison::NotEqual, 0),
            |command_buffer| command_buffer.encode_fill(&mut then_output, 0xFF),
            |command_buffer| command_buffer.encode_fill(&mut else_output, 0xFF),
        );
        submit_command_buffer(command_buffer);
        let ran_then = predicate_value != 0;
        assert_eq!(
            buffer_to_vec::<Metal, u32>(&context, &then_output),
            [if ran_then {
                u32::MAX
            } else {
                0
            }]
        );
        assert_eq!(
            buffer_to_vec::<Metal, u32>(&context, &else_output),
            [if ran_then {
                0
            } else {
                u32::MAX
            }]
        );
    }
}

#[uzu_test]
fn encode_if_sees_a_predicate_written_earlier_in_the_same_command_buffer() {
    let context = create_context::<Metal>();
    let mut predicate = create_buffer_with_data::<Metal, u32>(&context, &[0]);
    let condition = MetalCondition::new(&predicate, PredicateComparison::NotEqual, 0);
    for iteration in 0..64 {
        let run = iteration % 2 == 1;
        let mut inside = create_buffer_with_data::<Metal, u32>(&context, &[0]);
        let mut after = create_buffer_with_data::<Metal, u32>(&context, &[0]);
        let mut command_buffer = context.create_command_buffer(None, None).unwrap();
        command_buffer.encode_fill(&mut predicate, u8::from(run));
        command_buffer.encode_if(&condition, |command_buffer| command_buffer.encode_fill(&mut inside, 0xFF));
        command_buffer.encode_fill(&mut after, 0xFF);
        submit_command_buffer(command_buffer);
        let expected_inside = if run {
            u32::MAX
        } else {
            0
        };
        assert_eq!(buffer_to_vec::<Metal, u32>(&context, &inside), [expected_inside], "iteration {iteration}");
        assert_eq!(buffer_to_vec::<Metal, u32>(&context, &after), [u32::MAX], "iteration {iteration}");
    }
}

#[uzu_test]
fn encode_while_repeats_until_the_body_clears_the_predicate() {
    let context = create_context::<Metal>();
    for (initial, iterations) in [([5, 6, 7, 0, 11, 12, 13, 14], 3), ([0, 11, 12, 13, 14, 15, 16, 17], 0)] {
        let mut sequence = create_buffer_with_data::<Metal, u32>(&context, &initial);
        let mut scratch = create_buffer_with_data::<Metal, u32>(&context, &[0; 8]);
        let condition = MetalCondition::new((&sequence).subrange(..size_of::<u32>()), PredicateComparison::NotEqual, 0);
        let mut command_buffer = context.create_command_buffer(None, None).unwrap();
        command_buffer
            .encode_while(&condition, |command_buffer| shift_left(command_buffer, &mut sequence, &mut scratch));
        submit_command_buffer(command_buffer);
        assert_eq!(buffer_to_vec::<Metal, u32>(&context, &sequence), shifted(&initial, iterations), "{initial:?}");
    }
}

#[uzu_test]
fn encode_do_while_runs_the_body_before_checking_the_predicate() {
    let context = create_context::<Metal>();
    for (initial, iterations) in [([0, 0, 11, 12, 13, 14, 15, 16], 1), ([9, 8, 0, 11, 12, 13, 14, 15], 2)] {
        let mut sequence = create_buffer_with_data::<Metal, u32>(&context, &initial);
        let mut scratch = create_buffer_with_data::<Metal, u32>(&context, &[0; 8]);
        let condition = MetalCondition::new((&sequence).subrange(..size_of::<u32>()), PredicateComparison::NotEqual, 0);
        let mut command_buffer = context.create_command_buffer(None, None).unwrap();
        command_buffer
            .encode_do_while(&condition, |command_buffer| shift_left(command_buffer, &mut sequence, &mut scratch));
        submit_command_buffer(command_buffer);
        assert_eq!(buffer_to_vec::<Metal, u32>(&context, &sequence), shifted(&initial, iterations), "{initial:?}");
    }
}
