import SwiftUI
import Uzu
import FirebaseCrashlytics

#if os(macOS)
    import AppKit
#endif

struct AboutView: View {
    func linkView(title: String, subtitle: String?) -> some View {
        VStack(alignment: .leading, spacing: 4.0) {
            Text(title)
                .foregroundStyle(MiraiAsset.primary.swiftUIColor)
                .font(.monoBody16)
                .frame(maxWidth: .infinity, alignment: .leading)
            if let subtitle {
                Text(subtitle)
                    .foregroundStyle(MiraiAsset.secondary.swiftUIColor)
                    .font(.monoBody16)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
        }
        .padding(16.0)
        .background(MiraiAsset.card.swiftUIColor)
        .cornerRadius(16.0)
    }

    func linkButton(url: String, title: String, subtitle: String? = nil) -> some View {
        Button(action: {
            if let url = URL(string: url) {
                #if os(iOS)
                    UIApplication.shared.open(url)
                #else
                    NSWorkspace.shared.open(url)
                #endif
            }
        }) {
            linkView(title: title, subtitle: subtitle)
        }
        .buttonStyle(.plain)
    }

    var body: some View {
        VStack(alignment: .center, spacing: 0.0) {
            ScrollView {
                VStack(alignment: .center, spacing: 16.0) {
                    linkButton(
                        url: "https://trymirai.com",
                        title: "Mirai",
                        subtitle:
                            "AI which run directly on your devices, bringing powerful capabilities closer to where decisions are made."
                    )
                    linkButton(
                        url: "https://saikollm.com",
                        title: "Saiko",
                        subtitle:
                            "A family of small AI models, highly optimized for on-device tasks. Zero cloud dependency, instant response times."
                    )
                    linkButton(
                        url: "https://trymirai.com/blog/deploying-llms-on-mobile",
                        title: "Introduction to Deploying LLMs on Mobile"
                    )
                    linkButton(
                        url: "https://trymirai.com/blog/how-to-understand-on-device-ai",
                        title: "How to Understand On-Device AI"
                    )
                    linkButton(
                        url: "https://trymirai.com/blog/iphone-hardware",
                        title: "iPhone Hardware",
                        subtitle: "How It Powers On-Device AI"
                    )
                    linkButton(
                        url: "https://trymirai.com/blog/brief-history-of-apple-ml-stack",
                        title: "Brief history of Apple ML Stack"
                    )
                    
                    #if DEBUG
                    // Crashlytics test button (Debug only)
                    Button(action: {
                        Crashlytics.crashlytics().log("User triggered test crash")
                        Crashlytics.crashlytics().setCustomValue("test", forKey: "crash_source")
                        fatalError("Test crash triggered from About view")
                    }) {
                        linkView(title: "Test Crashlytics", subtitle: "⚠️ This will crash the app (Debug only)")
                    }
                    .buttonStyle(.plain)
                    #endif
                }
            }
            .scrollIndicators(.hidden)
            .safeAreaPadding(16.0)
        }
        .navigationTitle("About Us")
        #if os(iOS)
            .navigationBarTitleDisplayMode(.inline)
        #endif
        .toolbarRole(.editor)
    }
}
