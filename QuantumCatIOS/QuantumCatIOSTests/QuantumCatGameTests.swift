import Foundation
import XCTest
@testable import QuantumCatIOS

final class QuantumCatGameTests: XCTestCase {
    func testBidRegimesDistinguishBelowExactAndAlreadyMissedTargets() {
        XCTAssertTrue(QuantumCatGame.wantsAdditionalTrick(prediction: 3, tricksWon: 2))
        XCTAssertFalse(QuantumCatGame.wantsAdditionalTrick(prediction: 2, tricksWon: 2))
        XCTAssertTrue(QuantumCatGame.wantsAdditionalTrick(prediction: 1, tricksWon: 2))
        XCTAssertTrue(QuantumCatGame.wantsAdditionalTrick(prediction: nil, tricksWon: 2))

        XCTAssertFalse(QuantumCatGame.winningWouldBreakExactPrediction(
            prediction: 3, tricksWon: 2, wouldWin: true
        ))
        XCTAssertTrue(QuantumCatGame.winningWouldBreakExactPrediction(
            prediction: 2, tricksWon: 2, wouldWin: true
        ))
        XCTAssertFalse(QuantumCatGame.winningWouldBreakExactPrediction(
            prediction: 1, tricksWon: 2, wouldWin: true
        ))
        XCTAssertFalse(QuantumCatGame.winningWouldBreakExactPrediction(
            prediction: 2, tricksWon: 2, wouldWin: false
        ))
    }

    func testFivePlayerBidsIncreaseForNinesAndStayLowForWeakHands() {
        XCTAssertEqual(
            QuantumCatGame.sharedPredictionBid(
                hand: [1, 1, 1, 1, 0, 0, 0, 0, 4],
                rankCount: 9,
                isRoundLeader: true
            ),
            4,
            "Four nines must never collapse to bidding one"
        )
        XCTAssertEqual(
            QuantumCatGame.sharedPredictionBid(
                hand: [1, 1, 1, 1, 0, 0, 0, 4, 0],
                rankCount: 9,
                isRoundLeader: true
            ),
            2,
            "Four eights with the opening lead should support more than one trick"
        )
        XCTAssertEqual(
            QuantumCatGame.sharedPredictionBid(
                hand: [2, 2, 1, 1, 1, 1, 0, 0, 0],
                rankCount: 9,
                isRoundLeader: true
            ),
            1,
            "A hand with no nines, eights, or sevens should keep the minimum bid"
        )
    }

    func testFivePlayerBidderUsesMultipleBidsAcrossActualDealtHands() throws {
        var announced: [Int] = []
        for offset in 0..<24 {
            var game = QuantumCatGame(
                seats: Array(repeating: .human, count: 5),
                seed: 2026092800 + offset,
                autoAdvanceBots: false
            )
            while game.phase == .discard {
                let action = try XCTUnwrap(game.legalMoves.first)
                XCTAssertTrue(game.applyHumanMove(action, autoAdvanceBots: false))
            }
            while game.phase == .prediction {
                let state = QuantumCatMLLiveState(game: game, player: game.currentPlayer)
                let bid = QuantumCatGame.sharedPredictionBid(state: state)
                announced.append(bid)
                XCTAssertTrue(game.applyHumanMove(.prediction(bid), autoAdvanceBots: false))
            }
        }

        XCTAssertEqual(announced.count, 120)
        XCTAssertGreaterThan(announced.filter { $0 == 2 }.count, 0)
        XCTAssertGreaterThan(announced.filter { $0 >= 3 }.count, 0)
        XCTAssertLessThan(announced.filter { $0 == 1 }.count, announced.count)
    }

    func testLearnedChampionBidsUseVerifiedRealModelAndRemainDiverse() throws {
        let policy = QuantumCatMLPolicy.shared
        XCTAssertFalse(
            policy.usesLearnedPrediction(for: .championBeliefPolicy, players: 3),
            "The legacy three-player champion must retain its exact setup behavior"
        )
        XCTAssertFalse(
            policy.usesLearnedPrediction(for: .championBeliefPolicy, players: 4),
            "An unsupported four-player model must remain fail-closed"
        )
        let learnedPredictionIsActive = policy.usesLearnedPrediction(
            for: .championBeliefPolicy,
            players: 5
        )
        XCTAssertTrue(
            learnedPredictionIsActive,
            "The bundled full-size five-player champion must use its verified neural bidding head"
        )
        guard learnedPredictionIsActive else { return }

        var bids: [Int] = []
        for offset in 0..<12 {
            var game = QuantumCatGame(
                seats: Array(repeating: .human, count: 5),
                seed: 2026100800 + offset,
                autoAdvanceBots: false
            )
            while game.phase == .discard {
                let move = try XCTUnwrap(game.legalMoves.first)
                XCTAssertTrue(game.applyHumanMove(move, autoAdvanceBots: false))
            }
            while game.phase == .prediction {
                let expected = try XCTUnwrap(policy.chooseMove(
                    kind: .championBeliefPolicy,
                    game: game,
                    player: game.currentPlayer,
                    legalMoves: game.legalMoves
                ))
                let recommended = try XCTUnwrap(
                    game.botRecommendationForCurrentHuman()
                )
                XCTAssertEqual(recommended.move, expected)
                guard case .prediction(let value) = expected else {
                    return XCTFail("Real prediction head selected a non-bid action")
                }
                bids.append(value)
                XCTAssertTrue(game.applyHumanMove(expected, autoAdvanceBots: false))
            }
        }
        XCTAssertEqual(bids.count, 60)
        XCTAssertGreaterThan(bids.filter { $0 > 1 }.count, 0)
        XCTAssertGreaterThan(bids.filter { $0 == 1 }.count, 0)
    }

    func testAboveBidCoreMLFeaturesMatchPythonAndCannotRecoverExactPrediction() throws {
        var game = QuantumCatGame(
            seats: [.human, .human, .human],
            seed: 20260721,
            autoAdvanceBots: false
        )
        while game.phase != .play {
            let move: Move = game.phase == .prediction
                ? .prediction(1)
                : try XCTUnwrap(game.legalMoves.first)
            XCTAssertTrue(game.applyHumanMove(move, autoAdvanceBots: false))
        }
        for _ in 0..<2 {
            let move = try XCTUnwrap(game.legalMoves.first)
            XCTAssertTrue(game.applyHumanMove(move, autoAdvanceBots: false))
        }

        let actor = game.currentPlayer
        let liveState = QuantumCatMLLiveState(game: game, player: actor)
        let encoded = try JSONEncoder().encode(liveState)
        var payload = try XCTUnwrap(
            JSONSerialization.jsonObject(with: encoded) as? [String: Any]
        )
        var predictions = try XCTUnwrap(payload["predictions"] as? [Int])
        var tricksWon = try XCTUnwrap(payload["tricksWon"] as? [Int])
        predictions[actor] = 1
        tricksWon[actor] = 2
        payload["predictions"] = predictions
        payload["tricksWon"] = tricksWon
        let aboveBidState = try JSONDecoder().decode(
            QuantumCatMLLiveState.self,
            from: JSONSerialization.data(withJSONObject: payload)
        )
        let rows = try QuantumCatMLFeatureEncoder.actionFeatures(
            state: aboveBidState,
            legalMoves: game.legalMoves
        )

        XCTAssertFalse(rows.isEmpty)
        for row in rows {
            XCTAssertEqual(row[32], 1, "Above-bid players should still value safe tricks")
            XCTAssertEqual(row[70], 1, "The last action should determine the trick winner")
            XCTAssertEqual(row[72], 1, "An already-missed prediction remains overshot")
            XCTAssertEqual(row[79], 0, "Overshot predictions cannot become exact again")
            XCTAssertEqual(
                row[186],
                0,
                "Only a trick won while currently exact can newly break a prediction"
            )
            XCTAssertEqual(
                row[46],
                row[45] > 0 ? 1 : 0,
                "Known trick wins should align with the above-bid objective"
            )
        }
    }

    func testSupportedHumanBotMixesReachScoringAndPersist() throws {
        let configs: [(String, [SeatKind])] = [
            ("5 humans", [.human, .human, .human, .human, .human]),
            ("2 humans 3 bots", [.human, .human, .bot(.championBeliefPolicy), .bot(.setPoolDistill), .bot(.strictQHead)]),
            ("2 humans 1 bot", [.human, .human, .bot(.championBeliefPolicy)]),
            ("1 human 4 bots", [.human, .bot(.championBeliefPolicy), .bot(.setPoolDistill), .bot(.rawPolicyLeague), .bot(.strictQHead)]),
            ("2 humans", [.human, .human])
        ]

        for (index, config) in configs.enumerated() {
            var game = QuantumCatGame(seats: config.1, seed: 20260606 + index)
            var steps = 0
            var sawPlay = false
            var sawSecondHuman = game.currentPlayer == 1 && game.activeHuman != nil

            while !game.isTerminal {
                steps += 1
                XCTAssertLessThan(steps, 260, "\(config.0) exceeded step guard")
                let legalMoves = game.legalMoves
                XCTAssertFalse(legalMoves.isEmpty, "\(config.0) has no legal moves in \(game.phase.rawValue)")
                if case .play = legalMoves[0] {
                    sawPlay = true
                }
                game.applyHumanMove(legalMoves[0])
                if game.currentPlayer == 1 && game.activeHuman != nil {
                    sawSecondHuman = true
                }
            }

            XCTAssertEqual(game.phase, .scoring, config.0)
            XCTAssertTrue(game.players.allSatisfy { $0.score != nil }, config.0)
            XCTAssertTrue(sawPlay, config.0)
            if config.1.prefix(2).allSatisfy({ $0.isHuman }) {
                XCTAssertTrue(sawSecondHuman, config.0)
            }

            let encoded = try JSONEncoder().encode(game)
            let restored = try JSONDecoder().decode(QuantumCatGame.self, from: encoded)
            XCTAssertEqual(restored.phase, game.phase, config.0)
            XCTAssertEqual(restored.board, game.board, config.0)
            XCTAssertEqual(restored.players.count, game.players.count, config.0)
        }
    }

    func testLegalBoardCellPlayMatchesLegalMoves() {
        var game = QuantumCatGame(seats: [.human, .human, .bot(.championBeliefPolicy)], seed: 20260608)
        var guardCount = 0
        while game.phase != .play && !game.isTerminal {
            guardCount += 1
            XCTAssertLessThan(guardCount, 80)
            game.applyHumanMove(game.legalMoves[0])
        }

        XCTAssertEqual(game.phase, .play)
        for move in game.legalMoves {
            guard case .play(let rank, let suit) = move else { continue }
            let suitIndex = Suit.allCases.firstIndex(of: suit)!
            XCTAssertEqual(game.board[suitIndex][rank - 1], -1)
        }
    }

    func testHumanMoveRejectsIllegalActionWithoutMutatingState() {
        var game = QuantumCatGame(
            seats: [.human, .human, .human],
            seed: 20260721,
            autoAdvanceBots: false
        )
        let before = GameSnapshot(game)

        XCTAssertFalse(game.applyHumanMove(.prediction(4), autoAdvanceBots: false))
        XCTAssertEqual(GameSnapshot(game), before)
    }

    func testBotRecommendationForHumanUsesChampionSnapshotWithoutMutating() throws {
        let humanGame = QuantumCatGame(
            seats: [.human, .bot(.championBeliefPolicy), .bot(.setPoolDistill)],
            seed: 20260618,
            autoAdvanceBots: false
        )
        var championSeatGame = QuantumCatGame(
            seats: [.bot(.championBeliefPolicy), .bot(.championBeliefPolicy), .bot(.setPoolDistill)],
            seed: 20260618,
            autoAdvanceBots: false
        )
        let before = GameSnapshot(humanGame)

        let recommendation = try XCTUnwrap(humanGame.botRecommendationForCurrentHuman())
        let championMove = try XCTUnwrap(championSeatGame.applyBotMoveForCurrentPlayer())

        XCTAssertEqual(recommendation.player, 0)
        XCTAssertEqual(recommendation.botKind, .championBeliefPolicy)
        XCTAssertEqual(recommendation.move, championMove)
        XCTAssertTrue(humanGame.legalMoves.contains(recommendation.move))
        XCTAssertEqual(recommendation.legalMoveCount, humanGame.legalMoves.count)
        XCTAssertEqual(recommendation.rankedMoves.count, humanGame.legalMoves.count)
        XCTAssertEqual(recommendation.rankedMoves.first?.move, recommendation.move)
        XCTAssertEqual(recommendation.rankedMoves.map(\.score).reduce(0, +), 100)
        XCTAssertEqual(GameSnapshot(humanGame), before)
    }

    func testChampionRecommendationRanksRunnerUpAndEveryLegalMove() throws {
        var game = QuantumCatGame(
            seats: [.human, .human, .human, .human, .human],
            seed: 2026092813,
            autoAdvanceBots: false
        )
        while game.phase != .play {
            let action: Move = game.phase == .prediction
                ? .prediction(1)
                : try XCTUnwrap(game.legalMoves.first)
            XCTAssertTrue(game.applyHumanMove(action, autoAdvanceBots: false))
        }
        let before = GameSnapshot(game)
        let recommendation = try XCTUnwrap(game.botRecommendationForCurrentHuman())

        XCTAssertGreaterThanOrEqual(recommendation.rankedMoves.count, 2)
        XCTAssertEqual(recommendation.rankedMoves.count, game.legalMoves.count)
        XCTAssertEqual(recommendation.rankedMoves.first?.move, recommendation.move)
        XCTAssertEqual(recommendation.rankedMoves.map(\.score).reduce(0, +), 100)
        XCTAssertTrue(recommendation.rankedMoves.allSatisfy {
            game.legalMoves.contains($0.move) && (0...100).contains($0.score)
        })
        XCTAssertEqual(
            Set(recommendation.rankedMoves.map(\.id)).count,
            game.legalMoves.count
        )
        XCTAssertEqual(GameSnapshot(game), before)
    }

    func testPreferenceSoftmaxHonorsProtectedMoveAndTotalsExactlyOneHundred() {
        let legal: [Move] = [.prediction(1), .prediction(2), .prediction(3)]
        let preferences = QuantumCatGame.normalizedMovePreferences(
            legalMoves: legal,
            preferenceValues: [8, 3, -.infinity],
            preferredMove: .prediction(2)
        )

        XCTAssertEqual(preferences.count, 3)
        XCTAssertEqual(preferences.first?.move, .prediction(2))
        XCTAssertGreaterThanOrEqual(preferences[0].score, preferences[1].score)
        XCTAssertEqual(preferences.last?.score, 0)
        XCTAssertEqual(preferences.map(\.score).reduce(0, +), 100)
    }

    func testFivePlayerCoachingScoresExcludeUnsafeNeuralMoves() throws {
        let rows = Array(repeating: Array(repeating: 0.0, count: 214), count: 3)
        let scores = try XCTUnwrap(
            QuantumCatMLPolicy.fivePlayerSpecialistPreferenceScores(
                policyLogits: [3.0, 2.8, 2.5],
                actionValues: [0.4, 0.1, 0.2],
                actionParadoxLogits: [0.0, 2.0, -2.0],
                actionFeatures: rows
            )
        )

        XCTAssertFalse(scores[0].isFinite)
        XCTAssertFalse(scores[1].isFinite)
        XCTAssertTrue(scores[2].isFinite)
        XCTAssertEqual(
            QuantumCatMLPolicy.selectFivePlayerSpecialistActionIndex(
                policyLogits: [3.0, 2.8, 2.5],
                actionValues: [0.4, 0.1, 0.2],
                actionParadoxLogits: [0.0, 2.0, -2.0],
                actionFeatures: rows
            ),
            2
        )
    }

    func testBotRecommendationIsAvailableAndLegalAcrossHumanTurnPhases() throws {
        var game = QuantumCatGame(
            seats: [.human, .human, .human],
            seed: 20260619,
            autoAdvanceBots: false
        )

        var sawDiscard = false
        var sawPrediction = false
        var guardCount = 0
        while game.phase != .play {
            guardCount += 1
            XCTAssertLessThan(guardCount, 40)
            let before = GameSnapshot(game)
            let recommendation = try XCTUnwrap(game.botRecommendationForCurrentHuman())

            XCTAssertTrue(game.legalMoves.contains(recommendation.move))
            XCTAssertEqual(recommendation.player, game.currentPlayer)
            XCTAssertEqual(GameSnapshot(game), before)
            sawDiscard = sawDiscard || game.phase == .discard
            sawPrediction = sawPrediction || game.phase == .prediction

            game.applyHumanMove(game.legalMoves[0], autoAdvanceBots: false)
        }

        XCTAssertTrue(sawDiscard)
        XCTAssertTrue(sawPrediction)
        let beforePlayRecommendation = GameSnapshot(game)
        let playRecommendation = try XCTUnwrap(game.botRecommendationForCurrentHuman())
        XCTAssertTrue(game.legalMoves.contains(playRecommendation.move))
        if case .play = playRecommendation.move {
            XCTAssertEqual(GameSnapshot(game), beforePlayRecommendation)
        } else {
            XCTFail("Expected a play recommendation in play phase")
        }
    }

    func testConfiguredStartingPlayerLeadsDeal() {
        let game = QuantumCatGame(
            seats: [.human, .human, .human],
            seed: 20260616,
            startingPlayer: 2,
            autoAdvanceBots: false
        )

        XCTAssertEqual(game.phase, .discard)
        XCTAssertEqual(game.currentPlayer, 2)
        XCTAssertTrue(game.log.last?.text.contains("P2 starts") == true)
    }

    @MainActor
    func testRoundWinnerStartsNextGame() {
        UserDefaults.standard.removeObject(forKey: "quantum-cat.saved-state.v1")
        let store = GameStore()
        store.humanSeats = 3
        store.botSeats = 0
        store.newGame()

        var steps = 0
        while !store.game.isTerminal {
            steps += 1
            XCTAssertLessThan(steps, 180)
            guard let move = store.game.legalMoves.first else {
                XCTFail("Expected a legal move in \(store.game.phase.rawValue)")
                return
            }
            store.apply(move)
        }

        guard let winner = store.game.roundWinner else {
            XCTFail("Expected a scored winner")
            return
        }

        store.newGame()

        XCTAssertEqual(store.game.phase, .discard)
        XCTAssertEqual(store.game.currentPlayer, winner)
        XCTAssertTrue(store.game.log.last?.text.contains("P\(winner) starts") == true)
    }

    @MainActor
    func testHumanMatchTelemetryExportsOneRoundPerSeatWithoutHiddenHands() throws {
        UserDefaults.standard.removeObject(forKey: "quantum-cat.saved-state.v1")
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("quantum-cat-telemetry-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: directory) }
        let store = GameStore(telemetryDirectoryURL: directory)
        store.humanSeats = 3
        store.humanParticipantIDs = ["Clay", "Shannon", "Guest", "player-4", "player-5"]
        store.botSeats = 0
        store.newGame()

        for roundIndex in 0..<3 {
            var steps = 0
            while !store.game.isTerminal {
                steps += 1
                XCTAssertLessThan(steps, 180)
                let move = try XCTUnwrap(store.game.legalMoves.first)
                store.apply(move)
            }
            if roundIndex < 2 {
                store.newGame()
            }
        }

        let telemetry = try XCTUnwrap(store.humanMatchTelemetry)
        XCTAssertTrue(telemetry.isComplete)
        XCTAssertEqual(telemetry.rounds.count, 3)
        XCTAssertEqual(telemetry.targetRounds, 3)
        XCTAssertEqual(telemetry.seats.map(\.participantID), [
            "human:clay", "human:shannon", "human:guest",
        ])
        XCTAssertTrue(telemetry.rounds.allSatisfy { !$0.moves.isEmpty })
        XCTAssertTrue(telemetry.rounds.flatMap(\.moves).allSatisfy {
            $0.legalActionCount == $0.legalActions.count
                && $0.legalActions.contains($0.action)
        })

        let exportURL = try XCTUnwrap(store.latestTelemetryExportURL)
        let data = try Data(contentsOf: exportURL)
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        let restored = try decoder.decode(HumanMatchTelemetry.self, from: data)
        XCTAssertEqual(restored.id, telemetry.id)
        XCTAssertEqual(restored.rounds.count, telemetry.rounds.count)
        XCTAssertFalse(String(decoding: data, as: UTF8.self).contains("\"hand\""))

        let jsonlURL = directory.appendingPathComponent("human_match_telemetry.jsonl")
        let jsonl = try String(contentsOf: jsonlURL, encoding: .utf8)
        XCTAssertEqual(jsonl.split(separator: "\n").count, 1)
    }

    @MainActor
    func testHumanParticipantIDsMustBeUniqueAfterNormalization() {
        UserDefaults.standard.removeObject(forKey: "quantum-cat.saved-state.v1")
        let store = GameStore()
        store.humanSeats = 2
        store.botSeats = 0
        store.humanParticipantIDs = ["Clay", " clay ", "player-3", "player-4", "player-5"]

        XCTAssertFalse(store.setupIsValid)

        store.humanParticipantIDs[1] = "Shannon"
        XCTAssertTrue(store.setupIsValid)
    }

    @MainActor
    func testBotTelemetryIdentityIncludesLiveInputContract() throws {
        UserDefaults.standard.removeObject(forKey: "quantum-cat.saved-state.v1")
        let store = GameStore()
        store.humanSeats = 1
        store.botSeats = 2
        store.botKinds = [.championBeliefPolicy, .heuristicTarget]
        store.newGame()

        let telemetry = try XCTUnwrap(store.humanMatchTelemetry)
        XCTAssertTrue(telemetry.seats[1].participantID.hasSuffix(
            ":python-observation76-action214-v1"
        ))
        XCTAssertEqual(
            telemetry.seats[2].participantID,
            "bot:heuristicTarget:native:native-v1"
        )
    }

    func testManualBotTurnsExposeActiveBotBeforeAdvancing() {
        var game = QuantumCatGame(
            seats: [.human, .bot(.championBeliefPolicy), .bot(.setPoolDistill)],
            seed: 20260610,
            autoAdvanceBots: false
        )

        XCTAssertNotNil(game.activeHuman)
        game.applyHumanMove(game.legalMoves[0], autoAdvanceBots: false)

        XCTAssertNil(game.activeHuman)
        XCTAssertEqual(game.currentPlayer, 1)
        XCTAssertEqual(game.activeBotKind, .championBeliefPolicy)

        let firstBotMove = game.applyBotMoveForCurrentPlayer()
        XCTAssertNotNil(firstBotMove)
        XCTAssertEqual(game.currentPlayer, 2)
        XCTAssertEqual(game.activeBotKind, .setPoolDistill)
    }

    func testChampionDiscardUsesSharedRankRuleBeforeModelPolicy() {
        var sawRankSixTie = false

        for seed in 20260620..<(20260620 + 2_000) {
            var game = QuantumCatGame(
                seats: [.bot(.championBeliefPolicy), .bot(.random), .bot(.random)],
                seed: seed,
                autoAdvanceBots: false
            )
            let hand = game.players[0].hand
            let maxCount = hand.max() ?? 0
            let tiedRanks = hand.enumerated()
                .filter { $0.element == maxCount }
                .map { $0.offset + 1 }
            guard tiedRanks.count > 1, tiedRanks.contains(6) else { continue }

            sawRankSixTie = true
            let expectedRank = expectedSharedDiscardRank(for: hand)
            let move = game.applyBotMoveForCurrentPlayer()

            XCTAssertEqual(move, .discard(rank: expectedRank))
            XCTAssertNotEqual(move, .discard(rank: 6))
            XCTAssertEqual(game.players[0].discardedRank, expectedRank)
            break
        }

        XCTAssertTrue(sawRankSixTie)
    }

    func testChampionMLRunsOnlyAfterLiveFeatureParityIsVerified() {
        let kind = BotKind.championBeliefPolicy
        let provenance = QuantumCatMLPolicy.shared.provenance(for: kind)

        XCTAssertNil(kind.coreMLActionRiskOutputName)
        XCTAssertTrue(kind.coreMLActionRiskRerankPhases.isEmpty)
        XCTAssertEqual(kind.coreMLActionValueSelectionWeight, 0)
        XCTAssertTrue(kind.coreMLActionValueRerankPhases.isEmpty)
        XCTAssertTrue(kind.subtitle.contains("verified native 76/214"))
        XCTAssertTrue(QuantumCatMLPolicy.shared.liveInferenceIsVerified(for: kind))
        XCTAssertEqual(
            provenance.liveInputContract,
            QuantumCatMLPolicy.verifiedLiveInputContract
        )
    }

    func testZeroHumanTablesCanWatchOrBulkSimulate() {
        var watchGame = QuantumCatGame(
            seats: [.bot(.championBeliefPolicy), .bot(.setPoolDistill), .bot(.strictQHead)],
            seed: 20260614,
            autoAdvanceBots: false
        )
        XCTAssertNil(watchGame.activeHuman)
        XCTAssertEqual(watchGame.currentPlayer, 0)
        XCTAssertEqual(watchGame.activeBotKind, .championBeliefPolicy)

        let firstMove = watchGame.applyBotMoveForCurrentPlayer()
        XCTAssertNotNil(firstMove)
        XCTAssertEqual(watchGame.currentPlayer, 1)

        let bulkGame = QuantumCatGame(
            seats: [.bot(.championBeliefPolicy), .bot(.setPoolDistill), .bot(.strictQHead)],
            seed: 20260615,
            autoAdvanceBots: true
        )
        XCTAssertTrue(bulkGame.isTerminal)
        XCTAssertTrue(bulkGame.players.allSatisfy { $0.score != nil })
    }

    func testVerifiedFivePlayerChampionUsesSpecialistCoreMLInference() {
        let seats = Array(repeating: SeatKind.bot(.championBeliefPolicy), count: 5)
        let games = 1
        let seedBase = 20263740
        let seeds = (0..<games).map { seedBase + $0 }
        XCTAssertEqual(Set(seeds).count, games)
        XCTAssertTrue(
            QuantumCatMLPolicy.shared.liveInferenceIsVerified(
                for: .championBeliefPolicy,
                players: seats.count
            )
        )
        XCTAssertEqual(
            QuantumCatMLPolicy.shared.provenance(
                for: .championBeliefPolicy,
                players: seats.count
            ).liveInputContract,
            "python-observation113-action214-5p-v2"
        )
        let result = sameModelParadoxReport(kind: .championBeliefPolicy, seats: seats.count, seeds: seeds)

        print(result.description)

        XCTAssertGreaterThanOrEqual(result.gameParadoxRate, 0.0)
        XCTAssertLessThanOrEqual(result.gameParadoxRate, 1.0)
        #if canImport(CoreML)
        XCTAssertGreaterThan(result.coreMLSuccesses, 0)
        XCTAssertEqual(result.coreMLAttempts, result.coreMLSuccesses)
        XCTAssertEqual(
            QuantumCatMLPolicy.shared.usageSnapshot().map(\.resource),
            ["champion_5p"]
        )
        #endif
    }

    func testUnsupportedFourPlayerChampionStillFailsClosedWithoutCoreMLAttempts() {
        XCTAssertFalse(
            QuantumCatMLPolicy.shared.liveInferenceIsVerified(
                for: .championBeliefPolicy,
                players: 4
            )
        )
        let result = sameModelParadoxReport(
            kind: .championBeliefPolicy,
            seats: 4,
            seeds: [20263740]
        )
        #if canImport(CoreML)
        XCTAssertEqual(result.coreMLSuccesses, 0)
        XCTAssertEqual(result.coreMLAttempts, 0)
        #endif
    }

    private struct SameModelParadoxReport {
        let kind: BotKind
        let seats: Int
        let games: Int
        let seedStart: Int
        let seedEnd: Int
        let seatParadoxRate: Double
        let gameParadoxRate: Double
        let coreMLSuccesses: Int
        let coreMLAttempts: Int

        var description: String {
            "SAME_MODEL_PARADOX kind=\(kind.rawValue) players=\(seats) games=\(games) " +
            "seed_start=\(seedStart) seed_end=\(seedEnd) " +
            "seat_rate=\(String(format: "%.4f", seatParadoxRate)) " +
            "game_rate=\(String(format: "%.4f", gameParadoxRate)) " +
            "coreml_successes=\(coreMLSuccesses) coreml_attempts=\(coreMLAttempts)"
        }
    }

    private func sameModelParadoxReport(kind: BotKind, seats: Int, seeds: [Int]) -> SameModelParadoxReport {
        let seats = Array(repeating: SeatKind.bot(kind), count: seats)
        var paradoxEvents = 0
        var gamesWithParadox = 0

        QuantumCatMLPolicy.shared.resetUsage()

        for seed in seeds {
            let game = QuantumCatGame(seats: seats, seed: seed, autoAdvanceBots: true)
            XCTAssertTrue(game.isTerminal)
            let playerParadoxes = game.players.map(\.hasParadoxed)
            paradoxEvents += playerParadoxes.filter { $0 }.count
            if playerParadoxes.contains(true) {
                gamesWithParadox += 1
            }
        }

        let gameCount = seeds.count
        let seatParadoxRate = Double(paradoxEvents) / Double(gameCount * seats.count)
        let gameParadoxRate = Double(gamesWithParadox) / Double(gameCount)
        let mlUsage = QuantumCatMLPolicy.shared.usageSnapshot()
        let successes = mlUsage.reduce(0) { $0 + $1.successes }
        let attempts = mlUsage.reduce(0) { $0 + $1.attempts }

        return SameModelParadoxReport(
            kind: kind,
            seats: seats.count,
            games: gameCount,
            seedStart: seeds.first ?? 0,
            seedEnd: seeds.last ?? 0,
            seatParadoxRate: seatParadoxRate,
            gameParadoxRate: gameParadoxRate,
            coreMLSuccesses: successes,
            coreMLAttempts: attempts
        )
    }

    @MainActor
    func testStoreNormalizesBotRosterLength() {
        UserDefaults.standard.removeObject(forKey: "quantum-cat.saved-state.v1")
        let store = GameStore()
        store.botSeats = 5
        store.botKinds = [.random]

        XCTAssertEqual(store.botSeats, 4)
        XCTAssertEqual(store.botKinds.count, store.botSeats)
        XCTAssertEqual(store.botKinds[0], .random)
        XCTAssertTrue(store.botKinds.dropFirst().allSatisfy { $0 == .championBeliefPolicy })

        store.botSeats = 2
        XCTAssertGreaterThanOrEqual(store.botKinds.count, 2)
        store.newGame()
        XCTAssertEqual(store.game.players.count, 3)
    }

    @MainActor
    func testStoreRunsZeroHumanBulkSimulation() {
        UserDefaults.standard.removeObject(forKey: "quantum-cat.saved-state.v1")
        let store = GameStore()
        store.humanSeats = 0
        store.botSeats = 3
        store.botOnlyRunMode = .bulk
        store.bulkSimulationGames = 7
        store.botKinds = [.championBeliefPolicy, .setPoolDistill, .strictQHead]

        XCTAssertTrue(store.setupIsValid)
        store.newGame()

        XCTAssertTrue(store.game.isTerminal)
        XCTAssertEqual(store.game.players.count, 3)
        XCTAssertEqual(store.bulkSimulationSummary?.games, 7)
        XCTAssertEqual(store.bulkSimulationSummary?.players.count, 3)
        XCTAssertNotNil(store.bulkSimulationSummary?.seatParadoxRate)
        XCTAssertEqual(store.bulkSimulationSummary?.players.reduce(0) { $0 + $1.paradoxes }, store.bulkSimulationSummary?.totalParadoxes)
    }

    private func expectedSharedDiscardRank(for hand: [Int]) -> Int {
        let maxCount = hand.max() ?? 0
        var candidates = hand.enumerated()
            .filter { $0.element == maxCount && $0.element > 0 }
            .map { $0.offset + 1 }
        if candidates.count > 1, candidates.contains(6) {
            candidates.removeAll { $0 == 6 }
        }
        return candidates.max { lhs, rhs in
            let lhsScore = expectedDiscardDissimilarityScore(hand: hand, rank: lhs)
            let rhsScore = expectedDiscardDissimilarityScore(hand: hand, rank: rhs)
            if lhsScore != rhsScore {
                return lhsScore < rhsScore
            }
            return lhs > rhs
        } ?? 1
    }

    private func expectedDiscardDissimilarityScore(hand: [Int], rank: Int) -> Double {
        var remaining = hand
        remaining[rank - 1] = max(0, remaining[rank - 1] - 1)
        let total = remaining.reduce(0, +)
        guard total > 0 else { return 0.0 }
        let weightedDistance = remaining.enumerated().reduce(0.0) { partial, item in
            partial + Double(abs((item.offset + 1) - rank) * item.element)
        }
        return weightedDistance / Double(total)
    }

    private struct GameSnapshot: Equatable {
        let seats: [SeatKind]
        let phase: String
        let players: [PlayerSnapshot]
        let rankCount: Int
        let tricksInRound: Int
        let currentPlayer: Int
        let trickNumber: Int
        let ledSuit: String
        let board: [[Int]]
        let currentTrick: [String]
        let completedTricks: [[String]]
        let log: [String]

        init(_ game: QuantumCatGame) {
            seats = game.seats
            phase = game.phase.rawValue
            players = game.players.map(PlayerSnapshot.init)
            rankCount = game.rankCount
            tricksInRound = game.tricksInRound
            currentPlayer = game.currentPlayer
            trickNumber = game.trickNumber
            ledSuit = game.ledSuit?.rawValue ?? "-"
            board = game.board
            currentTrick = game.currentTrick.map(Self.cardKey)
            completedTricks = game.completedTricks.map { trick in
                trick.map(Self.cardKey)
            }
            log = game.log.map { "\($0.id.uuidString)|\($0.kind)|\($0.text)" }
        }

        private static func cardKey(_ card: PlayedCard?) -> String {
            guard let card else { return "-" }
            return "\(card.suit.rawValue)\(card.rank)"
        }
    }

    private struct PlayerSnapshot: Equatable {
        let id: Int
        let seatKind: SeatKind
        let hand: [Int]
        let discardedRank: Int?
        let prediction: Int?
        let tricksWon: Int
        let hasParadoxed: Bool
        let colorTokens: [Bool]
        let score: Double?

        init(_ player: PlayerState) {
            id = player.id
            seatKind = player.seatKind
            hand = player.hand
            discardedRank = player.discardedRank
            prediction = player.prediction
            tricksWon = player.tricksWon
            hasParadoxed = player.hasParadoxed
            colorTokens = player.colorTokens
            score = player.score
        }
    }
}
