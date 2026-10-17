module definitions;

__include definitions.dsl;
__include definitions.storage_float;
__include definitions.bf16;
__include definitions.float_variants;
__include definitions.subgroup;

// A kernel precondition the binding checks, a host expression: phase "new" over specializations, checked by the
// constructor before creating anything; phase "encode" also over uniforms, checked before recording.
[[__AttributeUsage(_AttributeTargets.Function)]]
struct PreconditionAttribute {
  string phase;
  string condition;
};
