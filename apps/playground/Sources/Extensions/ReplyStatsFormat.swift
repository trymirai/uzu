import Foundation
import Uzu

enum ReplyStatsFormat {
    static let placeholder = "—"

    static func energy(_ joules: Double) -> String {
        measurement.string(from: Measurement(value: joules, unit: UnitEnergy.joules))
    }

    static func energyPerToken(_ stats: ChatReplyJoulesPerToken) -> String {
        switch stats {
        case let .total(total):
            "total \(formattedNumber(total)) J/tok"
        case let .components(cpu, gpu, _, dram):
            "CPU+GPU+DRAM \(formattedNumber(cpu + gpu + dram)) J/tok"
        }
    }

    static func memory(_ bytes: Int64?) -> String {
        guard let bytes else { return placeholder }
        return byteCount.string(fromByteCount: max(bytes, 0))
    }

    private static func formattedNumber(_ value: Double) -> String {
        number.string(from: NSNumber(value: value)) ?? placeholder
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
