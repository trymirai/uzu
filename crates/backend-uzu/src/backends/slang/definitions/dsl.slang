implementing definitions;

// Kernel annotations

[[__AttributeUsage(_AttributeTargets.Function)]]
struct VariantsAttribute {
    string name;
    string variants;
};

[[__AttributeUsage(_AttributeTargets.Function)]]
struct GroupSharedAttribute {
    string name;
    string size;
};

[[__AttributeUsage(_AttributeTargets.Function)]]
struct ConstraintAttribute {
    string constraint;
};

[[__AttributeUsage(_AttributeTargets.Function)]]
struct PublicAttribute;

[[__AttributeUsage(_AttributeTargets.Function)]]
struct KernelAttribute;

// Kernel argument annotation

[[__AttributeUsage(_AttributeTargets.Param)]]
struct ConstantAttribute;

[[__AttributeUsage(_AttributeTargets.Param)]]
struct OptionalAttribute {
    string condition;
};

[[__AttributeUsage(_AttributeTargets.Param)]]
struct SpecializeAttribute;

[[__AttributeUsage(_AttributeTargets.Param)]]
struct AxisAttribute {
    string threads;
    string threads_in_group;
};

[[__AttributeUsage(_AttributeTargets.Param)]]
struct GroupsAttribute {
    string groups;
};

[[__AttributeUsage(_AttributeTargets.Param)]]
struct ThreadsAttribute {
    string threads;
};
