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

struct MLMoveParityResult {
    let passed: Bool
    let fixtureCount: Int
    let mismatchCount: Int
    let maxAbsDiff: Double
    let liveInputMismatchCount: Int
    let maxLiveInputAbsDiff: Double
    let error: String?
}

struct MLModelProvenance {
    let checkpointSHA256: String?
    let liveInputContract: String?
}

private struct MLMoveParityPayload: Decodable {
    let schemaVersion: Int
    let checkpointSha256: String
    let modelResource: String
    let policyOutputName: String
    let stateValueOutputName: String?
    let actionParadoxOutputName: String?
    let actionValueOutputName: String?
    let liveInputContract: String
    let liveInputTolerance: Double
    let observationSize: Int
    let actionFeatureSize: Int
    let maxActions: Int
    let logitTolerance: Double
    let learnedPredictionEnabled: Bool?
    let predictionFixtureCount: Int?
    let fixtures: [MLMoveParityFixture]
}

private struct MLMoveParityFixture: Decodable {
    let id: String
    let legalActions: [Int]
    let observation: [Double]
    let actionFeatures: [[Double]]
    let expectedIndex: Int
    let expectedAction: Int
    let expectedLogits: [Double]
    let expectedStateValues: [Double]?
    let expectedActionParadoxLogits: [Double]?
    let expectedActionValues: [Double]?
    let topMargin: Double
    let liveState: QuantumCatMLLiveState
}

final class QuantumCatMLPolicy {
    static let shared = QuantumCatMLPolicy()
    static let verifiedLiveInputContract = "python-observation76-action214-v1"
    static let unifiedLiveInputContract = "python-observation113-action214-padded345-v2"

    private struct ModelRoute {
        let modelResource: String
        let fixtureResource: String
        let contract: String
        let observationSize: Int
        let policyOutputName: String
        let supportedPlayers: Set<Int>
    }

    private static let legacyRoute = ModelRoute(
        modelResource: "champion_belief_policy",
        fixtureResource: "champion_move_parity",
        contract: verifiedLiveInputContract,
        observationSize: QuantumCatMLFeatureEncoder.observationSize,
        policyOutputName: "var_175",
        supportedPlayers: [3]
    )
    private static let unifiedRoute = ModelRoute(
        modelResource: "champion_unified_345",
        fixtureResource: "champion_unified_move_parity",
        contract: unifiedLiveInputContract,
        observationSize: QuantumCatMLFeatureEncoder.unifiedObservationSize,
        policyOutputName: "policy_logits",
        supportedPlayers: [3, 4, 5]
    )
    private static let fourPlayerRoute = ModelRoute(
        modelResource: "champion_4p",
        fixtureResource: "champion_4p_move_parity",
        contract: "python-observation97-action214-4p-v2",
        observationSize: 97,
        policyOutputName: "policy_logits",
        supportedPlayers: [4]
    )
    private static let fivePlayerRoute = ModelRoute(
        modelResource: "champion_5p",
        fixtureResource: "champion_5p_move_parity",
        contract: "python-observation113-action214-5p-v2",
        observationSize: 113,
        policyOutputName: "policy_logits",
        supportedPlayers: [5]
    )

    #if canImport(CoreML)
    private var models: [String: MLModel] = [:]
    #else
    private let models: [String: Any] = [:]
    #endif

    private let maxActions = 1000
    private var usage: [String: MLPolicyUsageSnapshot] = [:]
    private var verifiedLiveResources: [String: Bool] = [:]

    private init() {}

    func resetUsage() {
        usage = [:]
    }

    func usageSnapshot() -> [MLPolicyUsageSnapshot] {
        usage.values.sorted { $0.resource < $1.resource }
    }

    func provenance(for kind: BotKind, players: Int = 3) -> MLModelProvenance {
        guard let defaultResource = kind.coreMLResource else {
            return MLModelProvenance(
                checkpointSHA256: nil,
                liveInputContract: "native-v1"
            )
        }
        let route = verifiedRoute(for: kind, players: players)
        let resource = route?.modelResource ?? defaultResource
        let fixtureResource = route?.fixtureResource ?? Self.legacyRoute.fixtureResource
        var checkpointSHA256: String?
        if let url = Bundle.main.url(forResource: fixtureResource, withExtension: "json"),
           let data = try? Data(contentsOf: url),
           let payload = try? JSONDecoder().decode(MLMoveParityPayload.self, from: data),
           payload.modelResource == resource {
            checkpointSHA256 = payload.checkpointSha256
        }
        return MLModelProvenance(
            checkpointSHA256: checkpointSHA256,
            liveInputContract: route?.contract ?? "live-input-parity-unverified"
        )
    }

    func liveInferenceIsVerified(for kind: BotKind, players: Int = 3) -> Bool {
        guard kind.coreMLResource != nil else { return true }
        return verifiedRoute(for: kind, players: players) != nil
    }

    func usesPlayerCountSpecialist(for kind: BotKind, players: Int) -> Bool {
        guard let route = verifiedRoute(for: kind, players: players) else {
            return false
        }
        return route.modelResource == Self.fourPlayerRoute.modelResource
            || route.modelResource == Self.fivePlayerRoute.modelResource
    }

    func usesLearnedPrediction(for kind: BotKind, players: Int) -> Bool {
        guard let route = verifiedRoute(for: kind, players: players),
              let url = Bundle.main.url(
                forResource: route.fixtureResource,
                withExtension: "json"
              ),
              let data = try? Data(contentsOf: url),
              let payload = try? JSONDecoder().decode(
                MLMoveParityPayload.self,
                from: data
              ) else { return false }
        return payload.learnedPredictionEnabled == true
            && (payload.predictionFixtureCount ?? 0) > 0
    }

    private func verifiedRoute(for kind: BotKind, players: Int) -> ModelRoute? {
        guard kind == .championBeliefPolicy else { return nil }
        let candidates: [ModelRoute]
        switch players {
        case 3: candidates = [Self.unifiedRoute, Self.legacyRoute]
        case 4: candidates = [Self.fourPlayerRoute, Self.unifiedRoute]
        case 5: candidates = [Self.fivePlayerRoute, Self.unifiedRoute]
        default: return nil
        }
        for route in candidates {
            guard route.supportedPlayers.contains(players),
                  Bundle.main.url(forResource: route.modelResource, withExtension: "mlmodelc") != nil,
                  Bundle.main.url(forResource: route.fixtureResource, withExtension: "json") != nil else {
                continue
            }
            let cacheKey = "\(route.modelResource):\(players)"
            if let cached = verifiedLiveResources[cacheKey] {
                if cached { return route }
                continue
            }
            let result = runMoveParityFixtures(resource: route.fixtureResource, players: players)
            let verified = result.passed
                && result.fixtureCount > 0
                && result.liveInputMismatchCount == 0
            verifiedLiveResources[cacheKey] = verified
            if verified { return route }
        }
        return nil
    }

    private static func route(forFixtureResource resource: String) -> ModelRoute? {
        [legacyRoute, unifiedRoute, fourPlayerRoute, fivePlayerRoute].first {
            $0.fixtureResource == resource
        }
    }

    func runMoveParityFixtures(
        resource: String = "champion_move_parity",
        players: Int? = nil
    ) -> MLMoveParityResult {
        #if canImport(CoreML)
        guard let url = Bundle.main.url(forResource: resource, withExtension: "json") else {
            return MLMoveParityResult(
                passed: false,
                fixtureCount: 0,
                mismatchCount: 0,
                maxAbsDiff: 0,
                liveInputMismatchCount: 0,
                maxLiveInputAbsDiff: 0,
                error: "missing_fixture_resource"
            )
        }
        do {
            let payload = try JSONDecoder().decode(
                MLMoveParityPayload.self,
                from: Data(contentsOf: url)
            )
            guard let route = Self.route(forFixtureResource: resource),
                  payload.schemaVersion == 2,
                  payload.modelResource == route.modelResource,
                  payload.liveInputContract == route.contract,
                  payload.policyOutputName == route.policyOutputName else {
                throw ParityError.invalidFixture("schema_version")
            }
            if let players, !route.supportedPlayers.contains(players) {
                throw ParityError.invalidFixture("unsupported_player_count")
            }
            let fixtures = payload.fixtures.filter {
                players == nil || $0.liveState.numPlayers == players
            }
            guard
                payload.observationSize == route.observationSize,
                payload.actionFeatureSize == QuantumCatMLFeatureEncoder.actionFeatureSize,
                payload.maxActions > 0,
                payload.maxActions <= maxActions,
                !fixtures.isEmpty,
                fixtures.allSatisfy({ route.supportedPlayers.contains($0.liveState.numPlayers) })
            else {
                throw ParityError.invalidFixture("dimensions")
            }
            if route.modelResource != Self.legacyRoute.modelResource {
                let representedCounts = Set(fixtures.map(\.liveState.numPlayers))
                for representedCount in representedCounts {
                    let includesAboveBid = fixtures.contains { fixture in
                        guard fixture.liveState.numPlayers == representedCount,
                              fixture.liveState.predictions.indices.contains(
                                  fixture.liveState.actingPlayer
                              ),
                              fixture.liveState.tricksWon.indices.contains(
                                  fixture.liveState.actingPlayer
                              ) else { return false }
                        let prediction = fixture.liveState.predictions[
                            fixture.liveState.actingPlayer
                        ]
                        return prediction >= 0
                            && fixture.liveState.tricksWon[fixture.liveState.actingPlayer]
                                > prediction
                    }
                    guard includesAboveBid else {
                        throw ParityError.invalidFixture("missing_above_bid_fixture")
                    }
                }
                if payload.learnedPredictionEnabled == true {
                    let predictionFixtures = fixtures.filter {
                        $0.liveState.phase == 2
                    }
                    let actualBids = Set(
                        predictionFixtures.map { $0.expectedAction - 100 }
                    )
                    guard predictionFixtures.count
                        == (payload.predictionFixtureCount ?? 0),
                          actualBids.contains(1),
                          actualBids.contains(where: { $0 > 1 }) else {
                        throw ParityError.invalidFixture(
                            "missing_diverse_learned_prediction_fixtures"
                        )
                    }
                }
            }
            guard let parityModel = model(resource: payload.modelResource) else {
                throw ParityError.invalidFixture("model_resource")
            }
            var mismatches = 0
            var maxAbsDiff = 0.0
            var liveInputMismatches = 0
            var maxLiveInputAbsDiff = 0.0
            for fixture in fixtures {
                let legalCount = fixture.legalActions.count
                let legalMoves = fixture.legalActions.compactMap {
                    QuantumCatMLFeatureEncoder.move(forPythonAction: $0, state: fixture.liveState)
                }
                guard
                    legalCount > 0,
                    legalCount <= payload.maxActions,
                    legalMoves.count == legalCount,
                    fixture.observation.count == payload.observationSize,
                    fixture.actionFeatures.count == legalCount,
                    fixture.actionFeatures.allSatisfy({ $0.count == payload.actionFeatureSize }),
                    fixture.expectedLogits.count == legalCount,
                    fixture.expectedIndex >= 0,
                    fixture.expectedIndex < legalCount,
                    fixture.legalActions[fixture.expectedIndex] == fixture.expectedAction
                else {
                    throw ParityError.invalidFixture(fixture.id)
                }

                let liveObservation = try QuantumCatMLFeatureEncoder.observation(
                    state: fixture.liveState,
                    paddedTo: payload.observationSize
                )
                let liveActionFeatures = try QuantumCatMLFeatureEncoder.actionFeatures(
                    state: fixture.liveState,
                    legalMoves: legalMoves
                )
                var fixtureLiveInputDiff = 0.0
                var fixtureLiveInputLocation = "none"
                var fixtureLiveInputActual = 0.0
                var fixtureLiveInputExpected = 0.0
                for index in liveObservation.indices {
                    let diff = abs(liveObservation[index] - fixture.observation[index])
                    if diff > fixtureLiveInputDiff {
                        fixtureLiveInputDiff = diff
                        fixtureLiveInputLocation = "observation[\(index)]"
                        fixtureLiveInputActual = liveObservation[index]
                        fixtureLiveInputExpected = fixture.observation[index]
                    }
                }
                for row in liveActionFeatures.indices {
                    for column in liveActionFeatures[row].indices {
                        let diff = abs(
                            liveActionFeatures[row][column] - fixture.actionFeatures[row][column]
                        )
                        if diff > fixtureLiveInputDiff {
                            fixtureLiveInputDiff = diff
                            fixtureLiveInputLocation = "action_features[\(row)][\(column)]"
                            fixtureLiveInputActual = liveActionFeatures[row][column]
                            fixtureLiveInputExpected = fixture.actionFeatures[row][column]
                        }
                    }
                }
                maxLiveInputAbsDiff = max(maxLiveInputAbsDiff, fixtureLiveInputDiff)
                let liveInputPassed = fixtureLiveInputDiff <= payload.liveInputTolerance
                if !liveInputPassed {
                    liveInputMismatches += 1
                    print(
                        "LIVE_INPUT_PARITY_MISMATCH fixture=\(fixture.id) " +
                        "location=\(fixtureLiveInputLocation) " +
                        "swift=\(fixtureLiveInputActual) " +
                        "python=\(fixtureLiveInputExpected) " +
                        "abs_diff=\(fixtureLiveInputDiff)"
                    )
                }

                let observation = try MLMultiArray(
                    shape: [1, NSNumber(value: payload.observationSize)],
                    dataType: .float32
                )
                let actionFeatures = try MLMultiArray(
                    shape: [
                        1,
                        NSNumber(value: payload.maxActions),
                        NSNumber(value: payload.actionFeatureSize),
                    ],
                    dataType: .float32
                )
                zero(observation)
                zero(actionFeatures)
                // The model is deliberately fed the Swift-native encoding. The
                // fixture tensors are reference evidence only.
                for (index, value) in liveObservation.enumerated() {
                    set(observation, [0, index], value)
                }
                for (row, features) in liveActionFeatures.enumerated() {
                    for (column, value) in features.enumerated() {
                        set(actionFeatures, [0, row, column], value)
                    }
                }
                let input = try MLDictionaryFeatureProvider(dictionary: [
                    "observation": MLFeatureValue(multiArray: observation),
                    "action_features": MLFeatureValue(multiArray: actionFeatures),
                ])
                let output = try parityModel.prediction(from: input)
                guard let logits = output.featureValue(
                    for: payload.policyOutputName
                )?.multiArrayValue else {
                    throw ParityError.invalidFixture("policy_output")
                }
                var actualIndex = 0
                var actualBest = -Double.infinity
                var fixtureMaxAbsDiff = 0.0
                for index in 0..<legalCount {
                    let actual = logit(logits, actionIndex: index)
                    let expected = fixture.expectedLogits[index]
                    fixtureMaxAbsDiff = max(
                        fixtureMaxAbsDiff,
                        abs(actual - expected)
                    )
                    if actual > actualBest {
                        actualBest = actual
                        actualIndex = index
                    }
                }
                if route.modelResource != Self.legacyRoute.modelResource {
                    guard
                        let stateOutputName = payload.stateValueOutputName,
                        let paradoxOutputName = payload.actionParadoxOutputName,
                        let valueOutputName = payload.actionValueOutputName,
                        let expectedStateValues = fixture.expectedStateValues,
                        let expectedActionParadox = fixture.expectedActionParadoxLogits,
                        let expectedActionValues = fixture.expectedActionValues,
                        let stateValues = output.featureValue(
                            for: stateOutputName
                        )?.multiArrayValue,
                        let actionParadox = output.featureValue(
                            for: paradoxOutputName
                        )?.multiArrayValue,
                        let actionValues = output.featureValue(
                            for: valueOutputName
                        )?.multiArrayValue,
                        expectedStateValues.count == stateValues.count,
                        expectedActionParadox.count == legalCount,
                        expectedActionValues.count == legalCount
                    else {
                        throw ParityError.invalidFixture("auxiliary_outputs")
                    }
                    for (index, expected) in expectedStateValues.enumerated() {
                        fixtureMaxAbsDiff = max(
                            fixtureMaxAbsDiff,
                            abs(logit(stateValues, actionIndex: index) - expected)
                        )
                    }
                    for index in 0..<legalCount {
                        fixtureMaxAbsDiff = max(
                            fixtureMaxAbsDiff,
                            abs(
                                logit(actionParadox, actionIndex: index)
                                    - expectedActionParadox[index]
                            ),
                            abs(
                                logit(actionValues, actionIndex: index)
                                    - expectedActionValues[index]
                            )
                        )
                    }
                }
                maxAbsDiff = max(maxAbsDiff, fixtureMaxAbsDiff)
                if
                    !liveInputPassed
                    ||
                    actualIndex != fixture.expectedIndex
                    || fixtureMaxAbsDiff > payload.logitTolerance
                {
                    mismatches += 1
                }
            }
            let passed = mismatches == 0 && liveInputMismatches == 0
            for representedCount in Set(fixtures.map(\.liveState.numPlayers)) {
                verifiedLiveResources["\(payload.modelResource):\(representedCount)"] = passed
            }
            return MLMoveParityResult(
                passed: passed,
                fixtureCount: fixtures.count,
                mismatchCount: mismatches,
                maxAbsDiff: maxAbsDiff,
                liveInputMismatchCount: liveInputMismatches,
                maxLiveInputAbsDiff: maxLiveInputAbsDiff,
                error: nil
            )
        } catch {
            if let route = Self.route(forFixtureResource: resource), let players {
                verifiedLiveResources["\(route.modelResource):\(players)"] = false
            }
            return MLMoveParityResult(
                passed: false,
                fixtureCount: 0,
                mismatchCount: 0,
                maxAbsDiff: 0,
                liveInputMismatchCount: 0,
                maxLiveInputAbsDiff: 0,
                error: String(describing: error)
            )
        }
        #else
        return MLMoveParityResult(
            passed: false,
            fixtureCount: 0,
            mismatchCount: 0,
            maxAbsDiff: 0,
            liveInputMismatchCount: 0,
            maxLiveInputAbsDiff: 0,
            error: "coreml_unavailable"
        )
        #endif
    }

    func chooseMove(
        kind: BotKind,
        game: QuantumCatGame,
        player: Int,
        legalMoves: [Move],
        scoreObserver: (([Double]) -> Void)? = nil
    ) -> Move? {
        #if canImport(CoreML)
        guard
            kind == .championBeliefPolicy,
            QuantumCatMLFeatureEncoder.supports(game: game),
            let route = verifiedRoute(for: kind, players: game.players.count),
            !legalMoves.isEmpty,
            legalMoves.count <= maxActions
        else { return nil }
        let resource = route.modelResource
        let actionFeatureSize = kind.coreMLActionFeatureSize
        guard let model = model(resource: resource) else {
            recordUsage(resource: resource, succeeded: false)
            return nil
        }
        do {
            let observationValues = try QuantumCatMLFeatureEncoder.observation(
                game: game,
                player: player,
                paddedTo: route.observationSize
            )
            let actionFeatureRows = try QuantumCatMLFeatureEncoder.actionFeatures(
                game: game, player: player, legalMoves: legalMoves
            )
            let observation = try MLMultiArray(
                shape: [1, NSNumber(value: route.observationSize)],
                dataType: .float32
            )
            let actionFeatures = try MLMultiArray(
                shape: [1, NSNumber(value: maxActions), NSNumber(value: actionFeatureSize)],
                dataType: .float32
            )
            zero(observation)
            zero(actionFeatures)
            for (index, value) in observationValues.enumerated() {
                set(observation, [0, index], value)
            }
            for (row, features) in actionFeatureRows.enumerated() {
                for (column, value) in features.enumerated() {
                    set(actionFeatures, [0, row, column], value)
                }
            }
            let input = try MLDictionaryFeatureProvider(dictionary: [
                "observation": MLFeatureValue(multiArray: observation),
                "action_features": MLFeatureValue(multiArray: actionFeatures),
            ])
            let output = try model.prediction(from: input)
            guard let logits = output.featureValue(
                for: route.policyOutputName
            )?.multiArrayValue else {
                recordUsage(resource: resource, succeeded: false)
                return nil
            }
            if game.phase == .prediction,
               usesLearnedPrediction(for: kind, players: game.players.count) {
                let values = (0..<legalMoves.count).map {
                    logit(logits, actionIndex: $0)
                }
                scoreObserver?(values)
                let best = values.indices.max { values[$0] < values[$1] }
                recordUsage(resource: resource, succeeded: best != nil)
                return best.map { legalMoves[$0] }
            }
            if route.modelResource == Self.fivePlayerRoute.modelResource {
                guard let actionValues = output.featureValue(
                    for: "action_values"
                )?.multiArrayValue,
                let actionParadox = output.featureValue(
                    for: "action_paradox_logits"
                )?.multiArrayValue else {
                    recordUsage(resource: resource, succeeded: false)
                    return nil
                }
                let legalCount = legalMoves.count
                let policyLogits = (0..<legalCount).map {
                    logit(logits, actionIndex: $0)
                }
                let qValues = (0..<legalCount).map {
                    logit(actionValues, actionIndex: $0)
                }
                let riskLogits = (0..<legalCount).map {
                    logit(actionParadox, actionIndex: $0)
                }
                scoreObserver?(
                    Self.fivePlayerSpecialistPreferenceScores(
                        policyLogits: policyLogits,
                        actionValues: qValues,
                        actionParadoxLogits: riskLogits,
                        actionFeatures: actionFeatureRows
                    ) ?? policyLogits
                )
                let bestIndex = Self.selectFivePlayerSpecialistActionIndex(
                    policyLogits: policyLogits,
                    actionValues: qValues,
                    actionParadoxLogits: riskLogits,
                    actionFeatures: actionFeatureRows
                )
                recordUsage(resource: resource, succeeded: bestIndex != nil)
                return bestIndex.map { legalMoves[$0] }
            }
            let actionValues = kind.coreMLActionValueOutputName.flatMap { name in
                output.featureValue(for: name)?.multiArrayValue
            }
            let actionRisks = kind.coreMLActionRiskOutputName.flatMap { name in
                output.featureValue(for: name)?.multiArrayValue
            }
            scoreObserver?((0..<legalMoves.count).map {
                logit(logits, actionIndex: $0)
            })
            let bestIndex = selectActionIndex(
                kind: kind,
                phase: game.phase,
                logits: logits,
                actionValues: actionValues,
                actionRisks: actionRisks,
                legalCount: legalMoves.count
            )
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

    func movePreferenceLogits(
        kind: BotKind,
        game: QuantumCatGame,
        player: Int,
        legalMoves: [Move]
    ) -> [Double]? {
        var values: [Double]?
        _ = chooseMove(
            kind: kind,
            game: game,
            player: player,
            legalMoves: legalMoves,
            scoreObserver: { values = $0 }
        )
        return values
    }

    static func fivePlayerSpecialistPreferenceScores(
        policyLogits: [Double],
        actionValues: [Double],
        actionParadoxLogits: [Double],
        actionFeatures: [[Double]]
    ) -> [Double]? {
        let count = policyLogits.count
        guard count > 0,
              actionValues.count == count,
              actionParadoxLogits.count == count,
              actionFeatures.count == count,
              actionFeatures.allSatisfy({ $0.count == 214 }) else {
            return nil
        }
        let baseline = policyLogits.indices.max {
            policyLogits[$0] < policyLogits[$1]
        } ?? 0
        let risks = actionParadoxLogits.map { 1.0 / (1.0 + exp(-$0)) }
        guard let minimumRisk = risks.min() else { return nil }
        return policyLogits.indices.map { index in
            guard risks[index] <= minimumRisk + 0.02,
                  policyLogits[baseline] - policyLogits[index] <= 2.0 else {
                return -Double.infinity
            }
            return specialistScore(
                policyLogit: policyLogits[index],
                actionValue: actionValues[index],
                features: actionFeatures[index]
            )
        }
    }

    static func selectFivePlayerSpecialistActionIndex(
        policyLogits: [Double],
        actionValues: [Double],
        actionParadoxLogits: [Double],
        actionFeatures: [[Double]]
    ) -> Int? {
        let count = policyLogits.count
        guard count > 0,
              actionValues.count == count,
              actionParadoxLogits.count == count,
              actionFeatures.count == count,
              actionFeatures.allSatisfy({ $0.count == 214 }) else {
            return nil
        }
        let baseline = policyLogits.indices.max {
            policyLogits[$0] < policyLogits[$1]
        } ?? 0
        let risks = actionParadoxLogits.map { 1.0 / (1.0 + exp(-$0)) }
        guard let minimumRisk = risks.min() else { return nil }
        let admissible = policyLogits.indices.filter { index in
            risks[index] <= minimumRisk + 0.02
                && policyLogits[baseline] - policyLogits[index] <= 2.0
        }
        guard !admissible.isEmpty else { return baseline }
        var selected = admissible[0]
        var selectedScore = specialistScore(
            policyLogit: policyLogits[selected],
            actionValue: actionValues[selected],
            features: actionFeatures[selected]
        )
        for index in admissible.dropFirst() {
            let score = specialistScore(
                policyLogit: policyLogits[index],
                actionValue: actionValues[index],
                features: actionFeatures[index]
            )
            if score > selectedScore
                || (score == selectedScore && risks[index] < risks[selected]) {
                selected = index
                selectedScore = score
            }
        }
        return selected
    }

    private static func specialistScore(
        policyLogit: Double,
        actionValue: Double,
        features: [Double]
    ) -> Double {
        let feasibility = 1.20 * features[209]
            + 0.80 * features[199]
            + 0.35 * features[200]
            + 0.25 * features[207]
            - 0.90 * features[211]
            - 0.65 * features[213]
            - 0.65 * features[201]
            - 0.45 * features[202]
            - 0.20 * features[203]
        let clippedValue = min(max(actionValue, -0.5), 0.5)
        return policyLogit + 20.0 * clippedValue + 2.0 * feasibility
    }

    #if canImport(CoreML)
    private enum ParityError: Error {
        case invalidFixture(String)
    }

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
        let configuration = MLModelConfiguration()
        // The policy is an ML Program.  On the simulator, Core ML can otherwise
        // select an MPSGraph backend from the host SDK that is newer than the
        // installed simulator runtime and silently return corrupt logits after
        // logging an "incompatible OS" validation error.  CPU-only inference is
        // deterministic for parity checks and avoids that unsafe backend choice.
        configuration.computeUnits = .cpuOnly
        guard let url = Bundle.main.url(forResource: resource, withExtension: "mlmodelc"),
              let loaded = try? MLModel(contentsOf: url, configuration: configuration) else {
            return nil
        }
        models[resource] = loaded
        return loaded
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

    private func selectActionIndex(
        kind: BotKind,
        phase: GamePhase,
        logits: MLMultiArray,
        actionValues: MLMultiArray?,
        actionRisks: MLMultiArray?,
        legalCount: Int
    ) -> Int? {
        guard legalCount > 0 else { return nil }
        let baseline = (0..<legalCount).max { lhs, rhs in
            qPolicyScore(kind: kind, phase: phase, logits: logits, actionValues: actionValues, actionIndex: lhs)
                < qPolicyScore(kind: kind, phase: phase, logits: logits, actionValues: actionValues, actionIndex: rhs)
        } ?? 0
        guard
            kind.coreMLActionRiskRerankPhases.contains(phase),
            let actionRisks
        else { return baseline }

        let minRisk = (0..<legalCount)
            .map { risk(actionRisks, actionIndex: $0) }
            .min() ?? risk(actionRisks, actionIndex: baseline)
        let maxPolicyGap = kind.coreMLActionRiskMaxPolicyGap
        let baselineLogit = logit(logits, actionIndex: baseline)
        let candidates = (0..<legalCount).filter { index in
            risk(actionRisks, actionIndex: index) <= minRisk + kind.coreMLActionRiskSlack
                && (maxPolicyGap < 0 || baselineLogit - logit(logits, actionIndex: index) <= maxPolicyGap)
        }
        guard !candidates.isEmpty else { return baseline }
        return candidates.max { lhs, rhs in
            let lhsScore = logit(logits, actionIndex: lhs) - risk(actionRisks, actionIndex: lhs)
            let rhsScore = logit(logits, actionIndex: rhs) - risk(actionRisks, actionIndex: rhs)
            return lhsScore < rhsScore
        }
    }

    private func risk(_ array: MLMultiArray, actionIndex: Int) -> Double {
        1.0 / (1.0 + exp(-logit(array, actionIndex: actionIndex)))
    }
    #endif
}
