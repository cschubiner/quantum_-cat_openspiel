import Foundation

enum BotKind: String, CaseIterable, Identifiable, Codable {
    case championBeliefPolicy
    case setPoolDistill
    case rawPolicyLeague
    case strictQHead
    case heuristicTarget
    case heuristicAdjacency
    case random

    var id: String { rawValue }

    var name: String {
        switch self {
        case .championBeliefPolicy: "Champion ML"
        case .setPoolDistill: "SetPool Distill"
        case .rawPolicyLeague: "Raw Policy League"
        case .strictQHead: "Strict Q Head"
        case .heuristicTarget: "Target Heuristic"
        case .heuristicAdjacency: "Adjacency Heuristic"
        case .random: "Random"
        }
    }

    var subtitle: String {
        switch self {
        case .championBeliefPolicy:
            "Best gated phase-one survival q-policy checkpoint, bundled as Core ML plus source PyTorch artifact."
        case .setPoolDistill:
            "Set-pooling distilled policy/value student, bundled as Core ML."
        case .rawPolicyLeague:
            "League-trained raw-policy diversity checkpoint, bundled as Core ML."
        case .strictQHead:
            "Action-value tactical checkpoint, bundled as Core ML."
        case .heuristicTarget:
            "Rule-aware bot that prioritizes trick target matching."
        case .heuristicAdjacency:
            "Rule-aware bot that prioritizes board clusters."
        case .random:
            "Uniform random baseline."
        }
    }

    var mobileStatus: String {
        switch self {
        case .championBeliefPolicy:
            ModelArtifactStatus.isBundled(resource: "champion_belief_policy", extension: "mlmodelc")
                ? "Core ML active; heuristic fallback ready."
                : "Core ML missing; heuristic fallback active."
        case .setPoolDistill:
            ModelArtifactStatus.isBundled(resource: "setpool_distill", extension: "mlmodelc")
                ? "Core ML active; heuristic fallback ready."
                : "Core ML missing; heuristic fallback active."
        case .rawPolicyLeague:
            ModelArtifactStatus.isBundled(resource: "raw_policy_league", extension: "mlmodelc")
                ? "Core ML active; heuristic fallback ready."
                : "Core ML missing; heuristic fallback active."
        case .strictQHead:
            ModelArtifactStatus.isBundled(resource: "strict_q_head", extension: "mlmodelc")
                ? "Core ML active; heuristic fallback ready."
                : "Core ML missing; heuristic fallback active."
        case .heuristicTarget:
            "Native, fast, target-aware."
        case .heuristicAdjacency:
            "Native, fast, cluster-aware."
        case .random:
            "Native random baseline."
        }
    }

    var weights: BotWeights {
        switch self {
        case .championBeliefPolicy:
            BotWeights(adjacency: 1.45, target: 2.35, winner: 1.35, highCard: 0.95, noise: 0.02)
        case .setPoolDistill:
            BotWeights(adjacency: 1.20, target: 1.95, winner: 1.15, highCard: 0.85, noise: 0.04)
        case .rawPolicyLeague:
            BotWeights(adjacency: 1.05, target: 1.65, winner: 1.25, highCard: 1.10, noise: 0.07)
        case .strictQHead:
            BotWeights(adjacency: 1.55, target: 1.40, winner: 1.65, highCard: 0.75, noise: 0.03)
        case .heuristicTarget:
            BotWeights(adjacency: 1.0, target: 2.0, winner: 1.0, highCard: 1.0, noise: 0.05)
        case .heuristicAdjacency:
            BotWeights(adjacency: 2.0, target: 1.5, winner: 1.0, highCard: 1.0, noise: 0.05)
        case .random:
            BotWeights(adjacency: 0, target: 0, winner: 0, highCard: 0, noise: 1)
        }
    }

    var coreMLResource: String? {
        switch self {
        case .championBeliefPolicy: "champion_belief_policy"
        case .setPoolDistill: "setpool_distill"
        case .rawPolicyLeague: "raw_policy_league"
        case .strictQHead: "strict_q_head"
        default: nil
        }
    }

    var coreMLPolicyOutputName: String? {
        switch self {
        case .setPoolDistill: "var_237"
        case .championBeliefPolicy, .strictQHead: "var_175"
        case .rawPolicyLeague: "var_175"
        default: nil
        }
    }

    var coreMLActionValueOutputName: String? {
        switch self {
        case .championBeliefPolicy, .strictQHead: "var_208"
        default: nil
        }
    }

    var coreMLActionValueSelectionWeight: Double {
        switch self {
        case .championBeliefPolicy, .strictQHead: 0.8
        default: 0.0
        }
    }

    var coreMLActionValueRerankClip: Double {
        switch self {
        case .championBeliefPolicy, .strictQHead: 0.6
        default: 0.0
        }
    }

    var coreMLActionValueRerankPhases: Set<GamePhase> {
        switch self {
        case .championBeliefPolicy, .strictQHead: [.play]
        default: []
        }
    }

    var coreMLActionFeatureSize: Int {
        switch self {
        case .championBeliefPolicy, .strictQHead: 214
        case .setPoolDistill, .rawPolicyLeague: 122
        default: 122
        }
    }
}

struct ModelArtifactStatus: Identifiable {
    let id: String
    let title: String
    let detail: String
    let resource: String
    let fileExtension: String

    var isPresent: Bool {
        Self.isBundled(resource: resource, extension: fileExtension)
    }

    var displayState: String {
        isPresent ? "Ready" : "Missing"
    }

    static let bundledArtifacts: [ModelArtifactStatus] = [
        ModelArtifactStatus(
            id: "champion-coreml",
            title: "Champion ML",
            detail: "Core ML phase-one survival q-policy",
            resource: "champion_belief_policy",
            fileExtension: "mlmodelc"
        ),
        ModelArtifactStatus(
            id: "champion-pytorch",
            title: "Champion checkpoint",
            detail: "Source PyTorch checkpoint",
            resource: "champion_belief_policy",
            fileExtension: "pt"
        ),
        ModelArtifactStatus(
            id: "setpool-distill",
            title: "SetPool Distill",
            detail: "Distilled Core ML policy",
            resource: "setpool_distill",
            fileExtension: "mlmodelc"
        ),
        ModelArtifactStatus(
            id: "setpool-distill-pytorch",
            title: "SetPool checkpoint",
            detail: "Source PyTorch checkpoint",
            resource: "setpool_distill",
            fileExtension: "pt"
        ),
        ModelArtifactStatus(
            id: "raw-policy-league",
            title: "Raw Policy League",
            detail: "League Core ML policy",
            resource: "raw_policy_league",
            fileExtension: "mlmodelc"
        ),
        ModelArtifactStatus(
            id: "raw-policy-league-pytorch",
            title: "League checkpoint",
            detail: "Source PyTorch checkpoint",
            resource: "raw_policy_league",
            fileExtension: "pt"
        ),
        ModelArtifactStatus(
            id: "strict-q-head",
            title: "Strict Q Head",
            detail: "Action-value Core ML policy",
            resource: "strict_q_head",
            fileExtension: "mlmodelc"
        ),
        ModelArtifactStatus(
            id: "strict-q-head-pytorch",
            title: "Q-head checkpoint",
            detail: "Source PyTorch checkpoint",
            resource: "strict_q_head",
            fileExtension: "pt"
        ),
    ]

    static func isBundled(resource: String, extension fileExtension: String) -> Bool {
        guard let url = Bundle.main.url(forResource: resource, withExtension: fileExtension) else {
            return false
        }
        return FileManager.default.fileExists(atPath: url.path)
    }
}

struct BotWeights {
    let adjacency: Double
    let target: Double
    let winner: Double
    let highCard: Double
    let noise: Double
}

struct SeatConfig: Identifiable, Equatable {
    let id = UUID()
    var kind: SeatKind
}

enum SeatKind: Equatable, Codable {
    case human
    case bot(BotKind)

    private enum CodingKeys: String, CodingKey {
        case type
        case botKind
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let type = try container.decode(String.self, forKey: .type)
        switch type {
        case "human":
            self = .human
        case "bot":
            self = .bot(try container.decode(BotKind.self, forKey: .botKind))
        default:
            throw DecodingError.dataCorruptedError(forKey: .type, in: container, debugDescription: "Unknown seat type \(type)")
        }
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        switch self {
        case .human:
            try container.encode("human", forKey: .type)
        case .bot(let kind):
            try container.encode("bot", forKey: .type)
            try container.encode(kind, forKey: .botKind)
        }
    }

    var isHuman: Bool {
        if case .human = self { return true }
        return false
    }

    var botKind: BotKind? {
        if case .bot(let kind) = self { return kind }
        return nil
    }

    var label: String {
        switch self {
        case .human: "Human"
        case .bot(let kind): kind.name
        }
    }
}
