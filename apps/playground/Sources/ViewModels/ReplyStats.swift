import Foundation
import Uzu

struct ReplyStats: Equatable {
    let timeToFirstToken: Double
    let tokensPerSecond: Double
    let memoryUsedBytes: Int64?
    let totalPower: Double?
    let totalEnergy: Double?
    let joulesPerToken: Double?
    let totalTime: Double

    init(stats: ChatReplyStats) {
        timeToFirstToken = stats.timeToFirstToken ?? 0.0
        tokensPerSecond = stats.generateTokensPerSecond ?? 0.0
        memoryUsedBytes = stats.memoryUsedBytes
        totalPower = stats.powerStats?.averageTotalWatts
        totalEnergy = stats.powerStats?.energyJoules
        joulesPerToken = stats.joulesPerToken()
        totalTime = stats.duration
    }
}
