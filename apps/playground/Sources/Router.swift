import Observation
import SwiftUI
import Uzu

@MainActor
@Observable
final class Router {
    enum ModelListDestination: Hashable {
        case classification
        case summarization
        case chat
        case textToSpeech
    }

    enum Destination: Hashable {
        case modelSelection(next: ModelListDestination)
        case chat(identifier: String)
        case classification(identifier: String)
        case summarization(identifier: String)
        case textToSpeech(identifier: String)
        case about
        case modelManagement

        static func == (lhs: Destination, rhs: Destination) -> Bool {
            switch (lhs, rhs) {
            case let (.modelSelection(lhsNext), .modelSelection(rhsNext)):
                return lhsNext == rhsNext
            case let (.chat(lhsId), .chat(rhsId)):
                return lhsId == rhsId
            case let (.classification(lhsId), .classification(rhsId)):
                return lhsId == rhsId
            case let (.summarization(lhsId), .summarization(rhsId)):
                return lhsId == rhsId
            case let (.textToSpeech(lhsId), .textToSpeech(rhsId)):
                return lhsId == rhsId
            case (.about, .about):
                return true
            case (.modelManagement, .modelManagement):
                return true
            default:
                return false
            }
        }

        func hash(into hasher: inout Hasher) {
            switch self {
            case let .modelSelection(next):
                hasher.combine(0)
                hasher.combine(next)
            case let .chat(identifier):
                hasher.combine(1)
                hasher.combine(identifier)
            case let .classification(identifier):
                hasher.combine(2)
                hasher.combine(identifier)
            case let .summarization(identifier):
                hasher.combine(3)
                hasher.combine(identifier)
            case let .textToSpeech(identifier):
                hasher.combine(4)
                hasher.combine(identifier)
            case .about:
                hasher.combine(5)
            case .modelManagement:
                hasher.combine(6)
            }
        }
    }

    var navPath: [Destination] = []

    func navigate(to destination: Destination) {
        navPath.append(destination)
    }

    func navigateBack(_ count: Int? = nil) {
        guard !navPath.isEmpty else { return }
        if let count {
            navPath.removeLast(min(count, navPath.count))
        } else {
            navPath.removeLast()
        }
    }

    func reset() {
        navPath.removeAll()
    }
}
