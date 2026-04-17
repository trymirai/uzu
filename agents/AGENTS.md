# Rules

The rules below are CRITICAL. Each rule should be respected. Rules can only be overridden by the user.

## Workflow rules

- Agent should use provided skills, especially the architecture description.
- When given a coding task, agent carefully and deeply inspects the codebase to understand the context.
- Agent doesn't ask for permission to read the contents of files and directories.
- When given a task, agent first thinks about multiple high-level strategies, and then asks the user for feedback on every option.
- Agent splits tasks into smallest possible subtasks, and solves them one by one while asking the user for detailed feedback during every step.
- Agent never edits code before proposing a short draft to the user and receiving a feedback.
- Agent is not allowed to make even the tiniest architectural decisions, it can only propose multiple options to the user.
- Agent behaves like an interviewer and asks the user questions all the time.
- Instead of asking multiple questions in one message, agent asks them one by one.
- Agent always asks for permission before launching CLI tools which may take lots of time or produce side effects.
- Before using any function from an external library, agent thoroughly inspects the code or documentation for the function to ensure that it uses it correctly.
- When encountering an error, agent never rushes to quickly fix it. Instead, it uses the scientific method to understand the source of the error: comes up with hypotheses and tests them by performing a series of experiments. It only attempts to fix the error once the hypothesis has been sufficiently validated.
- Agent avoids creating .md files with explanations unless explicitly requested by the user.
- When receiving general recommendations from the user, immediately add them to agents/AGENTS.md under the appropriate section.

## General coding guidelines

- Agent never resorts to quick hacks or stubs to make something work. Instead it always investigates the problem thoroughly and comes up with a well-thought-out solution.
- By default, agent does not write documentation or comments unless the code contains complex logic which is difficult to understand without them.
- Agent strives for simplicity and maintainability. Agent avoids writing boilerplate code, instead it tries to come up with the right abstractions.
- Agent prefers functional programming style, namely:
  - Avoids mutable state.
  - Avoids modifying objects in-place.
  - Avoids implicit side effects.
  - Prefers immutable data structures.
  - Strives to make invalid state unrepresentable.
- Agent tries to write code in the way that makes bugs hard to make and easy to spot.
- Agent writes the code so that assumptions about invariants are as explicit as possible.
- Agent leverages strong typing as much as possible, for example, it always uses enums instead of literals.
- Agent uses good descriptive variable names that make it easy to understand the purpose of the variable without looking at the code that uses it.
- Agent does not manually parse things such as JSON configs. Instead it first models a schema using types, and then uses a serialisation library.

## Testing guidelines

- Agent strives for simplicity and minimalism when writing tests.
- Agent avoids extensive mocking and stubbing.
- Agent focuses on testing core functionality, and does not test minor details such as property accessors.
- Agent avoids boilerplate in the testing code, and tries to come up with the right generic abstractions.

## Rust coding rules

- Prefer `Box<[T]>` to `Vec<T>` if the contents of the container are not going to change.
- Avoid mutable variables as much as possible. Use iterator expressions to collect data into containers.
- For calling API that was never used before, read the sources of the installed dependencies, otherwise, search in docs.rs.
- Avoid inline imports in trait bounds or function signatures; always import with `use` at the top of the file
- In sub-modules use `mod ...;` declarations with explicit `pub use module::{Item1, Item2}` — never global `pub use module::*`
- Format with `cargo +nightly fmt` to enable unstable rustfmt features
- Use full names, never abbreviate (e.g. `GroupDefinition` not `GroupDef`, `definition` not `def`)
- Avoid excessive comments; code should be self-explanatory
- Always use descriptive variable names; never single-letter names like `a`, `b`, `i`, `s`
- Do not extract helper methods that are only called from one place; inline the logic instead
- Avoid `super::` imports; always use `crate::` level imports
- Derive `Serialize`, `Deserialize`, `Clone`, `Debug`, `PartialEq`, `Eq`, `Hash` on all data types where applicable; all configs must be `Serialize`/`Deserialize`
- In recursive structures, prefer the base type name for child collections (e.g., `groups: Vec<Group>` not `children: Vec<Group>`)
- Never panic or force unwrap in library code; use typed errors with `thiserror`; panics/unwraps are only acceptable in tests
- All tests in `tests/` folder, named `test_<module>.rs`; test functions named `test_<module>_<behavior>`
- Shared test helpers in `tests/helpers/`; avoid duplicating helpers between test files
- Move dependencies to `[dev-dependencies]` when only used in tests
- Always check for unused dependencies and imports

## Python coding rules

- Never use pip to manage dependencies. Instead use uv commands, such as `uv add`.
- Assume Python 3.12. Use `T | None` instead of `Optional[T]` and `dict[K, V]` instead of `typing.Dict[K, V]`.
- Every function should have full type annotations.
- Prefer comprehensions to map and filter expressions.
- Prefer dataclasses over vanilla python classes.
- All path-like arguments should have type `Path | str`.
- Prefer frozen dataclasses or named tuples as primary data structures.
- Never use dicts in place of dataclasses.
- __init__.py files should only contain reexports.