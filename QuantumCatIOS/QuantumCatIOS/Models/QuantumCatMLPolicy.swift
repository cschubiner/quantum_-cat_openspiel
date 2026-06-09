import Foundation

#if canImport(CoreML)
import CoreML
#endif

struct MLPolicyUsageSnapshot: Identifiable {
    let resource: String
    let attempts: Int
    let successes: Int
    let failures: Int

    var id: String { resource }
}

final class QuantumCatMLPolicy {
    static let shared = QuantumCatMLPolicy()

    #if canImport(CoreML)
    private var models: [String: MLModel] = [:]
    #else
    private let models: [String: Any] = [:]
    #endif

    private let observationSize = 76
    private let maxActions = 1000
    private var usage: [String: MLPolicyUsageSnapshot] = [:]

    private init() {}

    func resetUsage() {
        usage = [:]
    }

    func usageSnapshot() -> [MLPolicyUsageSnapshot] {
        usage.values.sorted { $0.resource < $1.resource }
    }

    func chooseMove(kind: BotKind, game: QuantumCatGame, player: Int, legalMoves: [Move]) -> Move? {
        #if canImport(CoreML)
        guard
            let resource = kind.coreMLResource,
            let outputName = kind.coreMLPolicyOutputName,
            !legalMoves.isEmpty,
            legalMoves.count <= maxActions
        else { return nil }
        let actionFeatureSize = kind.coreMLActionFeatureSize
        guard let model = model(resource: resource) else {
            recordUsage(resource: resource, succeeded: false)
            return nil
        }
        do {
            let observation = try MLMultiArray(shape: [1, NSNumber(value: observationSize)], dataType: .float32)
            let actionFeatures = try MLMultiArray(
                shape: [1, NSNumber(value: maxActions), NSNumber(value: actionFeatureSize)],
                dataType: .float32
            )
            fillObservation(observation, game: game, player: player)
            fillActionFeatures(actionFeatures, game: game, player: player, legalMoves: legalMoves)
            let input = try MLDictionaryFeatureProvider(dictionary: [
                "observation": MLFeatureValue(multiArray: observation),
                "action_features": MLFeatureValue(multiArray: actionFeatures),
            ])
            let output = try model.prediction(from: input)
            guard let logits = output.featureValue(for: outputName)?.multiArrayValue else {
                recordUsage(resource: resource, succeeded: false)
                return nil
            }
            let actionValues = kind.coreMLActionValueOutputName.flatMap { name in
                output.featureValue(for: name)?.multiArrayValue
            }
            let bestIndex = (0..<legalMoves.count).max { lhs, rhs in
                qPolicyScore(kind: kind, phase: game.phase, logits: logits, actionValues: actionValues, actionIndex: lhs)
                    < qPolicyScore(kind: kind, phase: game.phase, logits: logits, actionValues: actionValues, actionIndex: rhs)
            }
            recordUsage(resource: resource, succeeded: bestIndex != nil)
            return bestIndex.map { legalMoves[$0] }
        } catch {
            recordUsage(resource: resource, succeeded: false)
            return nil
        }
        #else
        return nil
        #endif
    }

    #if canImport(CoreML)
    private func recordUsage(resource: String, succeeded: Bool) {
        let current = usage[resource] ?? MLPolicyUsageSnapshot(resource: resource, attempts: 0, successes: 0, failures: 0)
        usage[resource] = MLPolicyUsageSnapshot(
            resource: resource,
            attempts: current.attempts + 1,
            successes: current.successes + (succeeded ? 1 : 0),
            failures: current.failures + (succeeded ? 0 : 1)
        )
    }

    private func model(resource: String) -> MLModel? {
        if let existing = models[resource] {
            return existing
        }
        guard let url = Bundle.main.url(forResource: resource, withExtension: "mlmodelc"),
              let loaded = try? MLModel(contentsOf: url) else {
            return nil
        }
        models[resource] = loaded
        return loaded
    }

    private func fillObservation(_ observation: MLMultiArray, game: QuantumCatGame, player: Int) {
        zero(observation)
        set(observation, [0, 0], phaseValue(game.phase))
        set(observation, [0, 1], Double(player) / 4.0)
        set(observation, [0, 2], Double(game.players.count) / 5.0)
        set(observation, [0, 3], Double(game.rankCount) / 9.0)
        set(observation, [0, 4], Double(game.trickNumber) / Double(max(1, game.tricksInRound)))
        set(observation, [0, 5], game.ledSuit.map { Double(Suit.allCases.firstIndex(of: $0) ?? 0) / 3.0 } ?? -1.0)

        let active = game.players[player]
        set(observation, [0, 6], Double(active.tricksWon) / Double(max(1, game.tricksInRound)))
        set(observation, [0, 7], active.prediction.map { Double($0) / 4.0 } ?? 0.0)
        set(observation, [0, 8], active.hasParadoxed ? 1.0 : 0.0)

        for rankIndex in 0..<min(game.rankCount, 9) {
            set(observation, [0, 9 + rankIndex], Double(active.hand[rankIndex]) / 5.0)
        }

        var offset = 18
        for suitIndex in Suit.allCases.indices {
            guard offset < observationSize else { return }
            set(observation, [0, offset], active.colorTokens[suitIndex] ? 1.0 : 0.0)
            offset += 1
        }

        for row in game.board {
            for owner in row {
                guard offset < observationSize else { return }
                let value: Double
                if owner == -2 {
                    value = -0.5
                } else if owner < 0 {
                    value = 0.0
                } else if owner == player {
                    value = 1.0
                } else {
                    value = -1.0
                }
                set(observation, [0, offset], value)
                offset += 1
            }
        }
    }

    private func fillActionFeatures(_ actionFeatures: MLMultiArray, game: QuantumCatGame, player: Int, legalMoves: [Move]) {
        zero(actionFeatures)
        for (actionIndex, move) in legalMoves.enumerated() {
            set(actionFeatures, [0, actionIndex, 0], 1.0)
            set(actionFeatures, [0, actionIndex, 1], phaseValue(game.phase))
            set(actionFeatures, [0, actionIndex, 2], Double(player) / 4.0)
            set(actionFeatures, [0, actionIndex, 3], Double(game.trickNumber) / Double(max(1, game.tricksInRound)))
            switch move {
            case .discard(let rank):
                set(actionFeatures, [0, actionIndex, 4], 1.0)
                set(actionFeatures, [0, actionIndex, 8], Double(rank) / Double(game.rankCount))
                set(actionFeatures, [0, actionIndex, 9], Double(game.players[player].hand[rank - 1]) / 5.0)
            case .prediction(let value):
                set(actionFeatures, [0, actionIndex, 5], 1.0)
                set(actionFeatures, [0, actionIndex, 10], Double(value) / 4.0)
            case .play(let rank, let suit):
                let suitIndex = Suit.allCases.firstIndex(of: suit) ?? 0
                set(actionFeatures, [0, actionIndex, 6], 1.0)
                set(actionFeatures, [0, actionIndex, 8], Double(rank) / Double(game.rankCount))
                set(actionFeatures, [0, actionIndex, 11], Double(suitIndex) / 3.0)
                set(actionFeatures, [0, actionIndex, 12], game.ledSuit == suit ? 1.0 : 0.0)
                set(actionFeatures, [0, actionIndex, 13], suit == .red ? 1.0 : 0.0)
                set(actionFeatures, [0, actionIndex, 14], wouldCompleteTrick(game: game, player: player) ? 1.0 : 0.0)
            case .paradox:
                set(actionFeatures, [0, actionIndex, 7], 1.0)
            }
        }
    }

    private func phaseValue(_ phase: GamePhase) -> Double {
        switch phase {
        case .dealing: 0.0
        case .discard: 0.25
        case .prediction: 0.5
        case .play: 0.75
        case .scoring: 1.0
        }
    }

    private func wouldCompleteTrick(game: QuantumCatGame, player: Int) -> Bool {
        game.currentTrick.enumerated().allSatisfy { index, card in
            index == player || card != nil
        }
    }

    private func zero(_ array: MLMultiArray) {
        for index in 0..<array.count {
            array[index] = 0
        }
    }

    private func set(_ array: MLMultiArray, _ indexes: [Int], _ value: Double) {
        array[indexes.map(NSNumber.init(value:))] = NSNumber(value: value)
    }

    private func logit(_ array: MLMultiArray, actionIndex: Int) -> Double {
        array[[0, actionIndex].map(NSNumber.init(value:))].doubleValue
    }

    private func qPolicyScore(
        kind: BotKind,
        phase: GamePhase,
        logits: MLMultiArray,
        actionValues: MLMultiArray?,
        actionIndex: Int
    ) -> Double {
        var score = logit(logits, actionIndex: actionIndex)
        guard
            kind.coreMLActionValueSelectionWeight > 0,
            kind.coreMLActionValueRerankPhases.contains(phase),
            let actionValues
        else { return score }
        let rawValue = logit(actionValues, actionIndex: actionIndex)
        let clip = kind.coreMLActionValueRerankClip
        let clippedValue = clip > 0 ? min(max(rawValue, -clip), clip) : rawValue
        score += kind.coreMLActionValueSelectionWeight * clippedValue
        return score
    }
    #endif
}
