import SwiftUI

extension Font {
    // MARK: - Monospaced

    /// Body text, monospaced 16 pt, regular weight
    static let monoBody16 = Font.system(size: 16, design: .monospaced)
    /// Body text, monospaced 16 pt, semibold weight
    static let monoBody16Semibold = Font.system(size: 16, weight: .semibold, design: .monospaced)

    /// Caption text, monospaced 12 pt, regular weight
    static let monoCaption12 = Font.system(size: 12, design: .monospaced)
    /// Caption text, monospaced 12 pt, semibold weight
    static let monoCaption12Semibold = Font.system(size: 12, weight: .semibold, design: .monospaced)

    /// Heading text, monospaced 14 pt, bold weight
    static let monoHeading14Bold = Font.system(size: 14, weight: .bold, design: .monospaced)
    /// Heading text, monospaced 14 pt, medium weight
    static let monoHeading14Medium = Font.system(size: 14, weight: .medium, design: .monospaced)
    /// Heading text, monospaced 14 pt, semibold weight
    static let monoHeading14Semibold = Font.system(size: 14, weight: .semibold, design: .monospaced)
    /// Heading text, monospaced 14 pt, regular weight
    static let monoHeading14 = Font.system(size: 14, weight: .regular, design: .monospaced)

    // MARK: - Proportional (default) design

    /// Body text, proportional 16 pt, semibold weight
    static let body16Semibold = Font.system(size: 16, weight: .semibold)
    /// Title text, proportional 20 pt, regular weight
    static let title20 = Font.system(size: 20, weight: .regular)
    /// Large number/title, proportional 24 pt, light weight
    static let title24Light = Font.system(size: 24, weight: .light)

    /// Caption, proportional 12 pt
    static let caption12 = Font.system(size: 12, weight: .regular)
    /// Caption, proportional 12 pt, semibold weight
    static let caption12Semibold = Font.system(size: 12, weight: .semibold)

    // MARK: - Badges / small text

    /// Badge text, monospaced 10 pt, semibold weight
    static let monoBadge10Semibold = Font.system(size: 10, weight: .semibold, design: .monospaced)
}
