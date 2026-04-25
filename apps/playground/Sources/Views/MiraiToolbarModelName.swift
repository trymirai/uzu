import SwiftUI

struct MiraiToolbarModelName: ToolbarContent {

    let identifier: String

    var body: some ToolbarContent {
        if #available(iOS 26, macOS 26, *) {
            withHiddenBackground
        } else {
            defaultContent
        }
    }

    @available(iOS 26.0, macOS 26.0, *)
    @ToolbarContentBuilder
    private var withHiddenBackground: some ToolbarContent {
        #if os(macOS)
        ToolbarItem(placement: .automatic) {
            label(padding: 18)
        }
        #else
        ToolbarItem(placement: .topBarTrailing) {
            label(padding: 18)
        }
        #endif
    }

    @ToolbarContentBuilder
    private var defaultContent: some ToolbarContent {
        #if os(macOS)
        ToolbarItem(placement: .automatic) {
            label(padding: 12)
        }
        #else
        ToolbarItem(placement: .topBarTrailing) {
            label(padding: 12)
        }
        #endif
    }

    private func label(padding: CGFloat) -> some View {
        Text(identifier)
            .font(.monoCaption12Semibold)
            .foregroundColor(.secondary)
            .lineLimit(1)
            .truncationMode(.middle)
            .padding(.horizontal, padding)
            .padding(.vertical, 6)
    }
}
