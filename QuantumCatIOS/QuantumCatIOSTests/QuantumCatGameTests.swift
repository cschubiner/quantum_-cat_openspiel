import Foundation
import XCTest
@testable import QuantumCatIOS

final class QuantumCatGameTests: XCTestCase {
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

    func testChampionMLUsesRawPolicyWithNativeLivenessShield() {
        let kind = BotKind.championBeliefPolicy

        XCTAssertNil(kind.coreMLActionRiskOutputName)
        XCTAssertTrue(kind.coreMLActionRiskRerankPhases.isEmpty)
        XCTAssertEqual(kind.coreMLActionValueSelectionWeight, 0)
        XCTAssertTrue(kind.coreMLActionValueRerankPhases.isEmpty)
        XCTAssertTrue(kind.subtitle.contains("liveness shield"))
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

    func testChampionSameModelAnyParadoxGameRateIsUnderFortyPercent() {
        let seats = Array(repeating: SeatKind.bot(.championBeliefPolicy), count: 5)
        let games = 100
        let seedBase = 20263740
        let seeds = (0..<games).map { seedBase + $0 }
        XCTAssertEqual(Set(seeds).count, games)
        let result = sameModelParadoxReport(kind: .championBeliefPolicy, seats: seats.count, seeds: seeds)

        print(result.description)

        XCTAssertLessThan(result.gameParadoxRate, 0.40)
        #if canImport(CoreML)
        XCTAssertGreaterThan(result.coreMLSuccesses, 0)
        XCTAssertEqual(result.coreMLSuccesses, result.coreMLAttempts)
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
}
