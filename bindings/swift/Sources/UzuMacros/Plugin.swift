import SwiftCompilerPlugin
import SwiftSyntaxMacros

@main
struct UzuMacrosPlugin: CompilerPlugin {
    let providingMacros: [Macro.Type] = [
        UzuToolFunctionMacro.self,
    ]
}
