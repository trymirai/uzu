@attached(peer, names: suffixed(Tool), prefixed(__UzuToolArguments_))
public macro UzuToolFunction(
    name: String? = nil,
    description: String? = nil
) = #externalMacro(module: "UzuMacros", type: "UzuToolFunctionMacro")
