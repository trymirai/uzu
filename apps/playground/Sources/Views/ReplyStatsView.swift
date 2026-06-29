import SwiftUI

// Shared stats footer shown under a chat reply, classification result, or summary.
struct ReplyStatsView: View {
    let stats: ReplyStats

    var body: some View {
        Rectangle()
            .fill(MiraiAsset.cardBorder.swiftUIColor)
            .frame(height: 1)
            .padding(.vertical, 4)

        VStack(alignment: .leading, spacing: 4) {
            metricRow(label: "Time to first token:", value: String(format: "%.3f s", stats.timeToFirstToken))
            if stats.tokensPerSecond > 0 {
                metricRow(label: "Tokens per second:", value: String(format: "%.3f t/s", stats.tokensPerSecond))
            }
            metricRow(label: "Memory used:", value: ReplyStatsFormat.memory(stats.memoryUsedBytes))
            if let average = stats.averagePackagePower, let peak = stats.maxPackagePower {
                metricRow(label: ReplyStatsFormat.powerLabel, value: ReplyStatsFormat.power(average: average, maximum: peak))
            }
            if let energy = stats.packageEnergy {
                metricRow(label: "Energy:", value: ReplyStatsFormat.energy(energy))
            }
            if let energy = stats.packageEnergy, let tokens = stats.tokensCount, tokens > 0 {
                metricRow(label: "Energy / token:", value: ReplyStatsFormat.energyPerToken(joules: energy, tokens: tokens))
            }
            metricRow(label: "Total time:", value: String(format: "%.3f s", stats.totalTime))
        }
    }

    @ViewBuilder
    private func metricRow(label: String, value: String) -> some View {
        HStack(alignment: .firstTextBaseline, spacing: 6) {
            Text(label)
                .font(.monoCaption12)
                .foregroundStyle(MiraiAsset.secondary.swiftUIColor)
            Text(value)
                .font(.monoCaption12Semibold)
                .monospacedDigit()
                .lineLimit(1)
                .minimumScaleFactor(0.7)
        }
    }
}
