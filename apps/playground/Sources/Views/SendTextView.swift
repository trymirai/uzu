import SwiftUI

#if os(iOS)
import UIKit
#endif

/// Multiline text input that expands vertically up to five lines and triggers `onSend` on the *Send* key.
struct SendTextView: View {
    @Binding var text: String
    var onSend: () -> Void

    var body: some View {
        #if os(iOS)
        IOSMultilineTextField(text: $text, onSend: onSend)
        #else
        MacMultilineTextField(text: $text)
        #endif
    }
}

#if os(iOS)
// MARK: - iOS implementation (SwiftUI TextEditor)

private struct IOSMultilineTextField: View {
    @Binding var text: String
    var onSend: () -> Void

    var body: some View {
        TextField("", text: $text, axis: .vertical)
            .font(.monoBody16)
            .lineLimit(5, reservesSpace: false)
            .onSubmit {
                onSend()
            }
    }
}
#endif

#if os(macOS)
// MARK: - macOS implementation

private struct MacMultilineTextField: View {
    @Binding var text: String

    var body: some View {
        TextField("", text: $text, axis: .vertical)
            .textFieldStyle(.plain)
            .font(.monoBody16)
            .lineLimit(1...5)
            .fixedSize(horizontal: false, vertical: true)
    }
}
#endif
