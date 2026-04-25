import Foundation
import FirebaseCore
import FirebaseCrashlytics

#if os(iOS)
import UIKit
#elseif os(macOS)
import AppKit
#endif

private func configureCrashlyticsOptions() {
    #if os(macOS)
    UserDefaults.standard.register(defaults: ["NSApplicationCrashOnExceptions": true])
    #endif
    
    Crashlytics.crashlytics().setCrashlyticsCollectionEnabled(true)
    
    #if DEBUG
    Crashlytics.crashlytics().setUserID("debug-user")
    #endif
}

#if os(iOS)
final class AppDelegate: UIResponder, UIApplicationDelegate {
    func application(
        _ application: UIApplication,
        didFinishLaunchingWithOptions launchOptions: [UIApplication.LaunchOptionsKey: Any]? = nil
    ) -> Bool {
        configureCrashlyticsOptions()
        return true
    }
}
#elseif os(macOS)
final class AppDelegate: NSObject, NSApplicationDelegate {
    func applicationDidFinishLaunching(_ notification: Notification) {
        configureCrashlyticsOptions()
    }
}
#endif


