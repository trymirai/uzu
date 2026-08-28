import Foundation
import Uzu

// Shared reply-stat formatting: fixed-unit MeasurementFormatter + ByteCountFormatter.
enum ReplyStatsFormat {
    static let placeholder = "—"

    // macOS = SoC package power; iOS = whole-device charger "wall" power.
    static var powerLabel: String {
        #if os(macOS)
        return "Power (SoC):"
        #else
        return "Power (wall):"
        #endif
    }

    // Power/energy values are uncertain on iOS (and unset until a run completes), so the
    // caller renders these rows only when the values are present — these just format.
    static func power(average: Double) -> String {
        let avg = measurement.string(from: Measurement(value: average, unit: UnitPower.watts))
        return "\(avg) avg"
    }

    static func energy(_ joules: Double) -> String {
        measurement.string(from: Measurement(value: joules, unit: UnitEnergy.joules))
    }

    static func energyPerToken(_ stats: ChatReplyJoulesPerToken) -> String {
        let cpu = number.string(from: NSNumber(value: stats.cpu)) ?? placeholder
        let gpu = number.string(from: NSNumber(value: stats.gpu)) ?? placeholder
        let ane = number.string(from: NSNumber(value: stats.ane)) ?? placeholder
        let dram = number.string(from: NSNumber(value: stats.dram)) ?? placeholder
        let combined = number.string(from: NSNumber(value: stats.combined)) ?? placeholder
        return "CPU \(cpu), GPU \(gpu), ANE \(ane), DRAM \(dram), combined \(combined) J/tok"
    }

    static func memory(_ bytes: Int64?) -> String {
        guard let bytes else { return placeholder }
        return byteCount.string(fromByteCount: max(bytes, 0))
    }

    private static let number: NumberFormatter = {
        let formatter = NumberFormatter()
        formatter.numberStyle = .decimal
        formatter.minimumFractionDigits = 2
        formatter.maximumFractionDigits = 2
        return formatter
    }()

    private static let measurement: MeasurementFormatter = {
        let formatter = MeasurementFormatter()
        formatter.unitOptions = .providedUnit
        formatter.numberFormatter = number
        return formatter
    }()

    private static let byteCount: ByteCountFormatter = {
        let formatter = ByteCountFormatter()
        formatter.countStyle = .memory
        formatter.allowedUnits = [.useMB, .useGB]
        return formatter
    }()
}
