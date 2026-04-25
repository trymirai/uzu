import SwiftUI

struct MiraiToolbarLogo: ToolbarContent {

    struct Configuration {
        let topPadding: CGFloat
        let bottomPadding: CGFloat

        static let `default` = Configuration(topPadding: 0, bottomPadding: 0)
    }

    let configuration: Configuration

    init(configuration: Configuration = .default) {
        self.configuration = configuration
    }

    var body: some ToolbarContent {
        if #available(iOS 26, macOS 26, *) {
            logoWithHiddenBackground
        } else {
            logoDefault
        }
    }
    
    @available(iOS 26.0, macOS 26.0, *)
    @ToolbarContentBuilder
    private var logoWithHiddenBackground: some ToolbarContent {
        ToolbarItem(placement: .principal) {
            logoImage
        }
        .sharedBackgroundVisibility(.hidden)
    }
    
    @ToolbarContentBuilder
    private var logoDefault: some ToolbarContent {
        ToolbarItem(placement: .principal) {
            logoImage
        }
    }
    
    private var logoImage: some View {
        Image(asset: MiraiAsset.logo)
            .padding(.top, configuration.topPadding)
            .padding(.bottom, configuration.bottomPadding)
    }
}

