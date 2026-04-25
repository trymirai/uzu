import SFSymbols
import SwiftUI
import Uzu

struct ModelRowView: View, Equatable {

    // MARK: - Stored Properties
    let identifier: String
    let model: Uzu.Model
    let state: DownloadState
    let isSelected: Bool

    // MARK: - Equatable
    static func == (lhs: ModelRowView, rhs: ModelRowView) -> Bool {
        lhs.identifier == rhs.identifier &&
        lhs.isSelected == rhs.isSelected &&
        lhs.state == rhs.state
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 8) {
                statusIcon

                VStack(alignment: .leading, spacing: 4) {
                    Text(modelNameOnly)
                        .font(.monoHeading14)
                        .foregroundColor(MiraiAsset.primary.swiftUIColor)
                        .lineLimit(2)

                    HStack(spacing: 4) {
                        if let vendor = vendor {
                            badge(text: vendor)
                        }
                        if let qlabel = quantizationLabel {
                            badge(text: qlabel)
                        }
                    }
                }
                .frame(maxWidth: .infinity, alignment: .leading)

                Spacer()

                VStack(spacing: 8) {
                    statusText
                    if let bytes = bytesInfo {
                        Text(bytes)
                            .font(.monoCaption12)
                            .foregroundColor(MiraiAsset.secondary.swiftUIColor)
                    }
                }
            }

            if state.phase == .downloading || state.phase == .paused {
                let progress = min(max(state.progress(), 0.0), 1.0)
                VStack(alignment: .leading, spacing: 4) {
                    ProgressView(value: progress)
                        .progressViewStyle(
                            LinearProgressViewStyle(tint: MiraiAsset.primary.swiftUIColor)
                        )
                        .frame(height: 4)
                        .frame(maxWidth: .infinity)
                        .clipShape(Capsule())
                        .accessibilityIdentifier("modelProgressBar_\(identifier)")
                }
            }
        }
        .padding(16)
        .background(MiraiAsset.card.swiftUIColor)
        .cornerRadius(12)
        .accessibilityIdentifier("modelRow_\(identifier)")
    }

    // MARK: - Status helpers

    @ViewBuilder
    private var statusIcon: some View {
        Image(symbol: isSelected ? .checkmarkCircleFill : .circle)
            .font(.title24Light)
            .foregroundColor(MiraiAsset.primary.swiftUIColor)
    }

    @ViewBuilder
    private var statusText: some View {
        let statusFont = Font.monoHeading14
        switch state.phase {
        case .notDownloaded:
            Text("not installed")
                .font(statusFont)
                .foregroundColor(MiraiAsset.secondary.swiftUIColor)
        case .downloading:
            Text("installing…")
                .font(statusFont)
                .foregroundColor(MiraiAsset.secondary.swiftUIColor)
        case .downloaded:
            Text("installed")
                .font(statusFont)
                .foregroundColor(MiraiAsset.secondary.swiftUIColor)
        case .paused:
            Text("paused")
                .font(statusFont)
                .foregroundColor(MiraiAsset.secondary.swiftUIColor)
        case .error:
            Text("error")
                .font(statusFont)
                .foregroundColor(.red)
        case .locked:
            Text("locked")
                .font(statusFont)
                .foregroundColor(.red)
        }
    }

    // MARK: - Metadata helpers

    private var vendor: String? {
        model.family?.vendor.name()
    }

    private var quantizationLabel: String? {
        model.quantization?.name()
    }

    private var modelNameOnly: String {
        model.name()
    }

    // MARK: - Downloaded bytes info

    private var bytesInfo: String? {
        switch state.phase {
        case .downloading, .paused:
            return "\(formatBytes(state.downloadedBytes)) / \(formatBytes(state.totalBytes))"
        case .notDownloaded:
            return formatBytes(state.totalBytes)
        default:
            return nil
        }
    }

    private func formatBytes(_ bytes: Int64) -> String {
        let kb: Double = 1024
        let mb = kb * 1024
        let gb = mb * 1024

        let value = Double(bytes)

        if value >= gb {
            return String(format: "%.1f GB", value / gb)
        } else if value >= mb {
            return String(format: "%.1f MB", value / mb)
        } else if value >= kb {
            return String(format: "%.0f KB", value / kb)
        } else {
            return "\(bytes) B"
        }
    }

    // MARK: - Badge helper

    @ViewBuilder
    private func badge(text: String) -> some View {
        Text(text)
            .font(.monoBadge10Semibold)
            .padding(.horizontal, 6)
            .padding(.vertical, 2)
            .background(MiraiAsset.cardBorder.swiftUIColor)
            .foregroundColor(MiraiAsset.secondary.swiftUIColor)
            .clipShape(RoundedRectangle(cornerRadius: 4, style: .continuous))
    }
}
