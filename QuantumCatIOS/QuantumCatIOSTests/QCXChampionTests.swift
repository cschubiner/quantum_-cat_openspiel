import Foundation
import XCTest
@testable import QuantumCatIOS

/// Tests for the qcx champion (native C++ information-set search).
final class QCXChampionTests: XCTestCase {
    override func tearDown() {
        QCXChampion.worldsOverride = nil
        super.tearDown()
    }

    // MARK: helpers

    private struct TestRNG {
        var state: UInt64
        mutating func next() -> UInt64 {
            state = state &* 6364136223846793005 &+ 1442695040888963407
            var z = state
            z = (z ^ (z >> 30)) &* 0xbf58476d1ce4e5b9
            z = (z ^ (z >> 27)) &* 0x94d049bb133111eb
            return z ^ (z >> 31)
        }
        mutating func below(_ n: Int) -> Int { n <= 1 ? 0 : Int(next() % UInt64(n)) }
    }

    private func humanGame(players: Int, seed: Int) -> QuantumCatGame {
        QuantumCatGame(
            seats: Array(repeating: .human, count: players),
            seed: seed,
            startingPlayer: seed % players,
            autoAdvanceBots: false
        )
    }

    private func actionIDs(_ moves: [Move], rankCount: Int) -> [Int] {
        moves.map { QCXChampion.action(for: $0, rankCount: rankCount) }.sorted()
    }

    // MARK: (a) legal-action parity, Swift engine vs C++ view

    func testNativeLegalActionsMatchSwiftEngineAcrossRandomRounds() throws {
        var checked = 0
        var phasesSeen = Set<String>()
        var paradoxSeen = false
        for players in 3...5 {
            for seed in 0..<80 {
                var game = humanGame(players: players, seed: 7_000 + seed)
                var rng = TestRNG(state: UInt64(seed * 31 + players))
                var steps = 0
                while !game.isTerminal {
                    steps += 1
                    XCTAssertLessThan(steps, 200)
                    let me = game.currentPlayer
                    let view = try XCTUnwrap(QCXView.build(game: game, player: me))
                    XCTAssertTrue(QCXView.isConsistent(view), "inconsistent view P=\(players) seed=\(seed) step=\(steps)")
                    let native = QCXChampion.legalActions(view: view).sorted()
                    let swift = actionIDs(game.legalMoves, rankCount: game.rankCount)
                    XCTAssertEqual(native, swift, "P=\(players) seed=\(seed) phase=\(game.phase) step=\(steps)")
                    checked += 1
                    phasesSeen.insert(game.phase.rawValue)
                    if game.legalMoves == [.paradox] { paradoxSeen = true }
                    let legal = game.legalMoves
                    // Mostly random, sometimes the lowest id: drives rounds into paradoxes too.
                    let move = rng.below(3) == 0 ? legal[0] : legal[rng.below(legal.count)]
                    XCTAssertTrue(game.applyHumanMove(move, autoAdvanceBots: false))
                }
            }
        }
        XCTAssertGreaterThan(checked, 5_000)
        XCTAssertEqual(phasesSeen, ["discard", "prediction", "play"])
        XCTAssertTrue(paradoxSeen, "random play should reach at least one forced paradox")
    }

    // MARK: (b) the bot returns a legal action in every phase

    func testChampionReturnsLegalMoveInEveryPhaseFor3To5Players() throws {
        QCXChampion.worldsOverride = 12
        let failuresBefore = QCXChampion.usage.failures
        for players in 3...5 {
            var phases = Set<GamePhase>()
            for seed in 0..<10 {
                var game = QuantumCatGame(
                    seats: Array(repeating: .bot(.qcxChampion), count: players),
                    seed: 31_000 + 17 * seed + players,
                    startingPlayer: seed % players,
                    autoAdvanceBots: false
                )
                var steps = 0
                while !game.isTerminal {
                    steps += 1
                    XCTAssertLessThan(steps, 200)
                    let legal = game.legalMoves
                    let phase = game.phase
                    let decision = try XCTUnwrap(
                        QCXChampion.decide(game: game, player: game.currentPlayer),
                        "no decision P=\(players) seed=\(seed) phase=\(phase)"
                    )
                    XCTAssertTrue(legal.contains(decision.move))
                    // The real gameplay route (botMove -> qcx) must agree.
                    let applied = try XCTUnwrap(game.applyBotMoveForCurrentPlayer())
                    XCTAssertEqual(applied, decision.move)
                    phases.insert(phase)
                }
                XCTAssertNotNil(game.players.first?.score)
            }
            XCTAssertEqual(phases, [.discard, .prediction, .play], "P=\(players)")
        }
        XCTAssertEqual(QCXChampion.usage.failures, failuresBefore)
    }

    func testTwoPlayerTablesFallBackToPreviousChampion() throws {
        var game = QuantumCatGame(seats: [.bot(.qcxChampion), .bot(.qcxChampion)], seed: 5, autoAdvanceBots: false)
        XCTAssertNil(QCXChampion.decide(game: game, player: game.currentPlayer))
        XCTAssertNotNil(game.applyBotMoveForCurrentPlayer())
    }

    func testBundledBudgetLatencyOnFullRounds() throws {
        // Measures the shipped configuration end to end (simulator/Mac CPU).
        let config = QCXChampion.config
        XCTAssertTrue(config.viewLayoutMatchesBinary)
        XCTAssertGreaterThanOrEqual(config.worlds, 64)
        for players in [3, 5] {
            var game = QuantumCatGame(
                seats: Array(repeating: .bot(.qcxChampion), count: players),
                seed: 424_242 + players,
                autoAdvanceBots: false
            )
            var seconds: [Double] = []
            while !game.isTerminal {
                let decision = try XCTUnwrap(QCXChampion.decide(game: game, player: game.currentPlayer))
                seconds.append(decision.seconds)
                XCTAssertEqual(game.applyBotMoveForCurrentPlayer(), decision.move)
            }
            let sorted = seconds.sorted()
            let mean = seconds.reduce(0, +) / Double(max(1, seconds.count))
            print(String(
                format: "QCX_LATENCY players=%d worlds=%d decisions=%d mean=%.3fs p90=%.3fs max=%.3fs",
                players, config.worlds, seconds.count, mean,
                sorted[min(sorted.count - 1, Int(Double(sorted.count) * 0.9))], sorted.last ?? 0
            ))
        }
    }

    // MARK: (c) information safety

    func testDecisionIgnoresOpponentsHiddenCards() throws {
        var compared = 0
        var hiddenChanged = 0
        for players in 3...5 {
            for seed in 0..<12 {
                var rng = TestRNG(state: UInt64(900 + seed * 7 + players))
                var game = humanGame(players: players, seed: 55_000 + seed)
                // Stop at a random decision point (any phase).
                let stopAt = rng.below(players * 2 + 12)
                var step = 0
                while !game.isTerminal, step < stopAt {
                    let legal = game.legalMoves
                    _ = game.applyHumanMove(legal[rng.below(legal.count)], autoAdvanceBots: false)
                    step += 1
                }
                guard !game.isTerminal else { continue }
                let me = game.currentPlayer

                // Consistently re-deal every opponent's hidden cards: pool their
                // hands and face-down discards, shuffle, refill same sizes.
                var pool: [Int] = []
                for p in game.players.indices where p != me {
                    for (rank, n) in game.players[p].hand.enumerated() {
                        pool += Array(repeating: rank, count: n)
                    }
                    if let discarded = game.players[p].discardedRank { pool.append(discarded - 1) }
                }
                for i in stride(from: pool.count - 1, to: 0, by: -1) {
                    pool.swapAt(i, rng.below(i + 1))
                }
                var altered = game
                var cursor = 0
                for p in game.players.indices where p != me {
                    var hand = Array(repeating: 0, count: game.rankCount)
                    var discard: Int?
                    if game.players[p].discardedRank != nil {
                        discard = pool[cursor] + 1
                        cursor += 1
                    }
                    for _ in 0..<game.players[p].hand.reduce(0, +) {
                        hand[pool[cursor]] += 1
                        cursor += 1
                    }
                    altered.overrideHiddenCards(for: p, hand: hand, discardedRank: discard)
                }
                XCTAssertEqual(cursor, pool.count)
                if game.players.indices.contains(where: { p in
                    p != me && (game.players[p].hand != altered.players[p].hand
                        || game.players[p].discardedRank != altered.players[p].discardedRank)
                }) {
                    hiddenChanged += 1
                }
                XCTAssertEqual(game.players[me].hand, altered.players[me].hand)

                XCTAssertEqual(
                    QCXView.build(game: game, player: me),
                    QCXView.build(game: altered, player: me),
                    "view leaked hidden cards P=\(players) seed=\(seed)"
                )
                let original = try XCTUnwrap(QCXChampion.decide(game: game, player: me, worlds: 48, seed: 0xC0FFEE))
                let changed = try XCTUnwrap(QCXChampion.decide(game: altered, player: me, worlds: 48, seed: 0xC0FFEE))
                XCTAssertEqual(original.move, changed.move, "P=\(players) seed=\(seed) phase=\(game.phase)")
                XCTAssertEqual(original.values, changed.values)
                // The production per-decision seed is derived from the view only.
                let a = try XCTUnwrap(QCXChampion.decide(game: game, player: me, worlds: 24))
                let b = try XCTUnwrap(QCXChampion.decide(game: altered, player: me, worlds: 24))
                XCTAssertEqual(a.move, b.move)
                compared += 1
            }
        }
        XCTAssertGreaterThan(compared, 25)
        XCTAssertGreaterThan(hiddenChanged, compared / 2, "the test must actually change hidden hands")
    }

    func testViewContainsOnlyActingPlayersHand() throws {
        let game = humanGame(players: 4, seed: 12_345)
        for me in 0..<4 {
            let view = try XCTUnwrap(QCXView.build(game: game, player: me))
            XCTAssertEqual(Array(view[8..<(8 + game.rankCount)]).map(Int.init), game.players[me].hand)
            XCTAssertEqual(view[17], -1)
        }
    }

    // MARK: cross-language parity (Python qc_py.decide fixtures)

    func testPythonFixturesMatchSwiftViewAndNativeDecision() throws {
        let file = try QCXParity.loadBundled()
        XCTAssertEqual(file.viewLen, QCXView.length)
        XCTAssertGreaterThanOrEqual(file.fixtures.count, 30)
        XCTAssertEqual(Set(file.fixtures.map(\.players)), [3, 4, 5])
        XCTAssertEqual(Set(file.fixtures.map(\.phase)), [1, 2, 3])
        let report = QCXParity.run(file)
        XCTAssertTrue(report.passed, report.summary + "\n" + report.failures.prefix(10).joined(separator: "\n"))
        print("QCX_PARITY " + report.summary)
    }

    // MARK: catalog / defaults

    @MainActor
    func testQcxChampionIsTheDefaultBot() {
        UserDefaults.standard.removeObject(forKey: "quantum-cat.saved-state.v1")
        let store = GameStore()
        XCTAssertEqual(BotKind.allCases.first, .qcxChampion)
        XCTAssertEqual(BotKind.qcxChampion.name, "Champion (qcx PIMC search)")
        XCTAssertTrue(store.botKinds.allSatisfy { $0 == .qcxChampion })
        XCTAssertTrue(store.game.seats.contains(.bot(.qcxChampion)))
        XCTAssertTrue(QCXChampion.supports(players: 3))
        XCTAssertTrue(QCXChampion.supports(players: 4))
        XCTAssertTrue(QCXChampion.supports(players: 5))
        XCTAssertFalse(QCXChampion.supports(players: 2))
    }

    func testReplaceBotKindUpgradesSavedSeats() {
        var game = QuantumCatGame(
            seats: [.human, .bot(.championBeliefPolicy), .bot(.random)],
            seed: 3,
            autoAdvanceBots: false
        )
        game.replaceBotKind(.championBeliefPolicy, with: .qcxChampion)
        XCTAssertEqual(game.seats, [.human, .bot(.qcxChampion), .bot(.random)])
        XCTAssertEqual(game.players[1].seatKind, .bot(.qcxChampion))
    }
}
