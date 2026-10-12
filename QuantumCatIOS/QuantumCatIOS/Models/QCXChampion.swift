import Foundation
import QCXNative

/// Parameters of the bundled qcx champion, written by `qcx/sync_ios.sh` from
/// `qcx/qcx_champion.json`. Unknown keys are ignored and every field has a
/// default, so newer champion files (extra fields such as a bid model) still
/// decode on an older app.
struct QCXChampionConfig: Decodable {
    var id: String = "qcx_pimc"
    var name: String = "qcx PIMC search"
    var displayName: String = "Champion (qcx PIMC search)"
    var worlds: Int = 256
    var util: Int = 0
    var viewLen: Int = Int(QCX_VIEW_LEN)
    var players: [Int] = [3, 4, 5]
    var sourceCommit: String?
    var sourceSHA256: String?
    var evidence: String?

    private enum CodingKeys: String, CodingKey {
        case id, name, worlds, util, players, evidence
        case displayName = "display_name"
        case viewLen = "view_len"
        case sourceCommit = "source_commit"
        case sourceSHA256 = "source_sha256"
    }

    init() {}

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        let fallback = QCXChampionConfig()
        id = (try? c.decode(String.self, forKey: .id)) ?? fallback.id
        name = (try? c.decode(String.self, forKey: .name)) ?? fallback.name
        displayName = (try? c.decode(String.self, forKey: .displayName)) ?? fallback.displayName
        worlds = (try? c.decode(Int.self, forKey: .worlds)) ?? fallback.worlds
        util = (try? c.decode(Int.self, forKey: .util)) ?? fallback.util
        viewLen = (try? c.decode(Int.self, forKey: .viewLen)) ?? fallback.viewLen
        players = (try? c.decode([Int].self, forKey: .players)) ?? fallback.players
        sourceCommit = try? c.decode(String.self, forKey: .sourceCommit)
        sourceSHA256 = try? c.decode(String.self, forKey: .sourceSHA256)
        evidence = try? c.decode(String.self, forKey: .evidence)
    }

    static let bundled: QCXChampionConfig = {
        guard let url = Bundle.main.url(forResource: "qcx_champion", withExtension: "json"),
              let data = try? Data(contentsOf: url),
              let config = try? JSONDecoder().decode(QCXChampionConfig.self, from: data) else {
            return QCXChampionConfig()
        }
        return config
    }()

    /// The native search only reads views of this exact layout.
    var viewLayoutMatchesBinary: Bool { viewLen == Int(QCX_VIEW_LEN) }
}

/// The single place that turns an app game state into the qcx C ABI "view".
///
/// Mirrors `qcx/qc_py.py:build_view` field for field. Only the acting
/// player's information is read: their own hand and own discard, plus public
/// state (bids, board, colour tokens, tricks, current trick, start/leader/led,
/// trick number) and hand sizes derived from public counts. Other players'
/// hands and face-down discards are never touched.
///
/// Layout (ints, padded to 5 players / 9 ranks):
///  0 P, 1 round start, 2 phase, 3 me, 4 trick leader, 5 led colour, 6 trick no,
///  7 cards in current trick, 8..16 my hand, 17 my discard rank idx,
///  18..22 bids, 23..58 board[4][9], 59..63 colour-token bitmasks,
///  64..68 tricks won, 69..73 current-trick rank idx, 74..78 current-trick
///  colour, 79..83 hand sizes.
enum QCXView {
    static let length = Int(QCX_VIEW_LEN)
    static let maxPlayers = 5
    static let maxRanks = 9

    static func phaseCode(_ phase: GamePhase) -> Int32 {
        switch phase {
        case .dealing: 0
        case .discard: 1
        case .prediction: 2
        case .play: 3
        case .scoring: 4
        }
    }

    /// Python's `has_discarded[p]` under the reference (OpenSpiel) seat order,
    /// where discards go 0, 1, ..., P-1. The app lets the round's starting
    /// player discard first instead; that order is invisible information-wise
    /// (discards are face down), and expressing it the reference way keeps the
    /// view consistent with the C++ world sampler, which infers
    /// `has_discarded = q < me` during the discard phase.
    static func countsAsDiscarded(_ p: Int, me: Int, phase: GamePhase) -> Bool {
        phase != .discard || p < me
    }

    static func build(game: QuantumCatGame, player me: Int) -> [Int32]? {
        let players = game.players
        let count = players.count
        let ranks = game.rankCount
        guard (3...maxPlayers).contains(count),
              ranks <= maxRanks,
              players.indices.contains(me),
              game.board.count == 4,
              game.board.allSatisfy({ $0.count == ranks }) else { return nil }

        var v = [Int32](repeating: 0, count: length)
        v[0] = Int32(count)
        v[1] = Int32(game.mlRoundStartPlayer)
        v[2] = phaseCode(game.phase)
        v[3] = Int32(me)
        v[4] = Int32(game.mlStartPlayer)
        v[5] = game.ledSuit.flatMap { Suit.allCases.firstIndex(of: $0) }.map(Int32.init) ?? -1
        v[6] = Int32(game.trickNumber)
        v[7] = Int32(game.currentTrick.compactMap { $0 }.count)
        let hand = players[me].hand
        for r in 0..<ranks { v[8 + r] = Int32(hand.indices.contains(r) ? hand[r] : 0) }
        v[17] = players[me].discardedRank.map { Int32($0 - 1) } ?? -1
        for p in 0..<maxPlayers {
            v[18 + p] = p < count ? Int32(players[p].prediction ?? -1) : -1
            v[69 + p] = -1
            v[74 + p] = -1
        }
        for i in 23..<59 { v[i] = -1 }
        for c in 0..<4 {
            for r in 0..<ranks { v[23 + c * maxRanks + r] = Int32(game.board[c][r]) }
        }
        let initial = QuantumCatGame.cardsPerPlayer(for: count)
        for p in 0..<count {
            let tokens = players[p].colorTokens
            v[59 + p] = Int32((0..<4).reduce(0) { mask, c in
                mask | ((tokens.indices.contains(c) && tokens[c]) ? (1 << c) : 0)
            })
            v[64 + p] = Int32(players[p].tricksWon)
            if game.currentTrick.indices.contains(p), let card = game.currentTrick[p] {
                v[69 + p] = Int32(card.rank - 1)
                v[74 + p] = Int32(Suit.allCases.firstIndex(of: card.suit) ?? 0)
            }
            let played = game.board.reduce(0) { $0 + $1.filter { $0 == p }.count }
            let discarded = countsAsDiscarded(p, me: me, phase: game.phase) ? 1 : 0
            v[79 + p] = Int32(initial - discarded - played)
        }
        return v
    }

    /// Defensive check before handing a view to C++ (which trusts it): the
    /// unseen-card pool must exactly fill the opponents' hidden hands and
    /// discards, every count must be in range, and only seat `me` may act.
    static func isConsistent(_ v: [Int32]) -> Bool {
        guard v.count == length else { return false }
        let count = Int(v[0])
        guard (3...maxPlayers).contains(count) else { return false }
        let ranks = QuantumCatGame.rankCount(for: count)
        let me = Int(v[3])
        guard (0..<count).contains(me), (1...3).contains(Int(v[2])) else { return false }
        guard (0..<count).contains(Int(v[1])), (0..<count).contains(Int(v[4])) else { return false }
        var unseen = Array(repeating: 5, count: ranks)
        for r in 0..<maxRanks {
            let n = Int(v[8 + r])
            if r >= ranks { guard n == 0 else { return false }; continue }
            guard n >= 0 else { return false }
            unseen[r] -= n
        }
        let discard = Int(v[17])
        if discard >= 0 {
            guard discard < ranks else { return false }
            unseen[discard] -= 1
        }
        for c in 0..<4 {
            for r in 0..<maxRanks {
                let owner = Int(v[23 + c * maxRanks + r])
                if r >= ranks { guard owner == -1 else { return false }; continue }
                guard owner == -1 || (0..<count).contains(owner) else { return false }
                if owner >= 0 { unseen[r] -= 1 }
            }
        }
        guard unseen.allSatisfy({ $0 >= 0 }) else { return false }
        let phase = Int(v[2])
        var hidden = 0
        for p in 0..<count where p != me {
            let size = Int(v[79 + p])
            guard size >= 0 else { return false }
            hidden += size + ((phase != 1 || p < me) ? 1 : 0)
        }
        return hidden == unseen.reduce(0, +)
    }
}

struct QCXDecision {
    let move: Move
    let action: Int
    /// Mean search utility per legal action id (empty when only one action was legal).
    let values: [Int: Double]
    let worlds: Int
    let seconds: Double
}

struct QCXUsageSnapshot {
    var decisions = 0
    var failures = 0
    var totalSeconds = 0.0
    var maxSeconds = 0.0
}

/// Swift entry point for the qcx champion (information-set determinized
/// search, `qc_agents.h`). Call from a background queue: a 256-world search
/// is CPU bound.
enum QCXChampion {
    static var config: QCXChampionConfig { QCXChampionConfig.bundled }

    static func supports(players: Int) -> Bool {
        config.players.contains(players) && (3...5).contains(players) && config.viewLayoutMatchesBinary
    }

    // MARK: action encoding (OpenSpiel ids, shared with the C++ engine)

    static func action(for move: Move, rankCount: Int) -> Int {
        switch move {
        case .discard(let rank): rank - 1
        case .prediction(let value): 100 + value
        case .play(let rank, let suit): (Suit.allCases.firstIndex(of: suit) ?? 0) * rankCount + rank - 1
        case .paradox: 999
        }
    }

    static func move(for action: Int, phase: GamePhase, rankCount: Int) -> Move? {
        switch phase {
        case .discard:
            return (0..<rankCount).contains(action) ? .discard(rank: action + 1) : nil
        case .prediction:
            return (101...104).contains(action) ? .prediction(action - 100) : nil
        case .play:
            if action == 999 { return .paradox }
            guard (0..<(4 * rankCount)).contains(action) else { return nil }
            return .play(rank: action % rankCount + 1, suit: Suit.allCases[action / rankCount])
        default:
            return nil
        }
    }

    // MARK: native calls

    static func legalActions(view: [Int32]) -> [Int] {
        precondition(view.count == QCXView.length)
        var out = [Int32](repeating: 0, count: Int(QCX_MAX_ACTIONS))
        let n = view.withUnsafeBufferPointer { v in
            out.withUnsafeMutableBufferPointer { o in
                qc_legal_from_view(v.baseAddress, o.baseAddress)
            }
        }
        return out.prefix(Int(max(0, n))).map(Int.init)
    }

    /// Raw qc_decide with explicit parameters (used by parity fixtures).
    static func decide(view: [Int32], worlds: Int, util: Int, seed: UInt64) -> (action: Int, values: [Int: Double]) {
        precondition(view.count == QCXView.length)
        let legal = legalActions(view: view)
        var values = [Double](repeating: .nan, count: Int(QCX_MAX_ACTIONS))
        let action = view.withUnsafeBufferPointer { v in
            values.withUnsafeMutableBufferPointer { o in
                qc_decide(v.baseAddress, Int32(worlds), Int32(util), seed, o.baseAddress)
            }
        }
        var byAction: [Int: Double] = [:]
        if legal.count > 1 {
            for (index, id) in legal.enumerated() where values[index].isFinite {
                byAction[id] = values[index]
            }
        }
        return (Int(action), byAction)
    }

    /// Deterministic per-decision seed: a hash of the acting player's view, so
    /// the same information set always yields the same decision and nothing
    /// hidden can influence it.
    static func seed(for view: [Int32], salt: UInt64 = 0x51_43_58_5F_43_48_41_4D) -> UInt64 {
        var h: UInt64 = 0xcbf2_9ce4_8422_2325 ^ salt
        for value in view {
            h ^= UInt64(UInt32(bitPattern: value))
            h = h &* 0x0000_0100_0000_01B3
        }
        // splitmix64 finaliser
        h = (h ^ (h >> 30)) &* 0xbf58_476d_1ce4_e5b9
        h = (h ^ (h >> 27)) &* 0x94d0_49bb_1331_11eb
        return h ^ (h >> 31)
    }

    /// Decide for `player` (who must be the current player) from that seat's
    /// information only. Returns nil when unsupported (e.g. 2 players) or the
    /// view fails validation, so callers can fall back safely.
    static func decide(
        game: QuantumCatGame,
        player: Int,
        worlds: Int? = nil,
        seed explicitSeed: UInt64? = nil
    ) -> QCXDecision? {
        guard supports(players: game.players.count),
              !game.isTerminal,
              game.currentPlayer == player,
              [.discard, .prediction, .play].contains(game.phase),
              let view = QCXView.build(game: game, player: player) else { return nil }
        guard QCXView.isConsistent(view) else {
            recordFailure()
            return nil
        }
        let budget = max(1, worlds ?? worldsOverride ?? config.worlds)
        let started = DispatchTime.now().uptimeNanoseconds
        let result = decide(view: view, worlds: budget, util: config.util, seed: explicitSeed ?? seed(for: view))
        let seconds = Double(DispatchTime.now().uptimeNanoseconds - started) / 1e9
        guard let move = move(for: result.action, phase: game.phase, rankCount: game.rankCount),
              game.legalMoves.contains(move) else {
            recordFailure()
            return nil
        }
        record(seconds: seconds)
        return QCXDecision(move: move, action: result.action, values: result.values, worlds: budget, seconds: seconds)
    }

    // MARK: diagnostics

    /// Test/benchmark hook: overrides the bundled world count when set.
    static var worldsOverride: Int? {
        get { lock.lock(); defer { lock.unlock() }; return _worldsOverride }
        set { lock.lock(); _worldsOverride = newValue; lock.unlock() }
    }

    static var usage: QCXUsageSnapshot {
        lock.lock(); defer { lock.unlock() }
        return _usage
    }

    static func resetUsage() {
        lock.lock(); _usage = QCXUsageSnapshot(); lock.unlock()
    }

    private static let lock = NSLock()
    private static var _usage = QCXUsageSnapshot()
    private static var _worldsOverride: Int?

    private static func record(seconds: Double) {
        lock.lock()
        _usage.decisions += 1
        _usage.totalSeconds += seconds
        _usage.maxSeconds = max(_usage.maxSeconds, seconds)
        lock.unlock()
    }

    private static func recordFailure() {
        lock.lock(); _usage.failures += 1; lock.unlock()
    }
}

/// Cross-language parity against Python (`qcx/export_ios_fixtures.py`, run by
/// `qcx/sync_ios.sh`). Each fixture is replayed in the Swift engine from the
/// Python deal and action history; the Swift-built view, the native legal list
/// and the native decision (fixed seed, deterministic RNG) must all match.
enum QCXParity {
    struct Fixture: Decodable {
        let players: Int
        let start: Int
        let hands: [[Int]]
        let history: [Int]
        let me: Int
        let phase: Int
        let view: [Int32]
        let legal: [Int]
        let worlds: Int
        let util: Int
        let seed: UInt64
        let action: Int
    }

    struct FixtureFile: Decodable {
        let viewLen: Int
        let fixtures: [Fixture]
        private enum CodingKeys: String, CodingKey {
            case fixtures
            case viewLen = "view_len"
        }
    }

    struct Report {
        var fixtures = 0
        var viewMismatches = 0
        var legalMismatches = 0
        var actionMismatches = 0
        var swiftRouteMismatches = 0
        var replayFailures = 0
        var failures: [String] = []
        var passed: Bool {
            fixtures > 0 && viewMismatches + legalMismatches + actionMismatches
                + swiftRouteMismatches + replayFailures == 0
        }
        var summary: String {
            "fixtures=\(fixtures) view_mismatches=\(viewMismatches) legal_mismatches=\(legalMismatches) " +
            "action_mismatches=\(actionMismatches) swift_route_mismatches=\(swiftRouteMismatches) " +
            "replay_failures=\(replayFailures) passed=\(passed)"
        }
    }

    static func loadBundled() throws -> FixtureFile {
        guard let url = Bundle.main.url(forResource: "qcx_parity_fixtures", withExtension: "json") else {
            throw CocoaError(.fileNoSuchFile)
        }
        return try JSONDecoder().decode(FixtureFile.self, from: Data(contentsOf: url))
    }

    /// Rebuild the fixture's decision point in the Swift engine.
    static func replay(_ fixture: Fixture) -> QuantumCatGame? {
        let seats = Array(repeating: SeatKind.human, count: fixture.players)
        guard var game = QuantumCatGame.withDealtHands(
            seats: seats, hands: fixture.hands, startingPlayer: fixture.start
        ) else { return nil }
        let count = fixture.players
        if fixture.phase == 1 {
            // Python discards in seat order 0..P-1, the app from the starting
            // player. The acting seat's view does not depend on what anyone
            // else discarded, so earlier seats (app order) take any legal discard.
            while game.phase == .discard, game.currentPlayer != fixture.me {
                guard let move = game.legalMoves.first,
                      game.applyHumanMove(move, autoAdvanceBots: false) else { return nil }
            }
            return game.phase == .discard ? game : nil
        }
        guard fixture.history.count >= count else { return nil }
        while game.phase == .discard {
            let action = fixture.history[game.currentPlayer]
            guard let move = QCXChampion.move(for: action, phase: .discard, rankCount: game.rankCount),
                  game.applyHumanMove(move, autoAdvanceBots: false) else { return nil }
        }
        for action in fixture.history.dropFirst(count) {
            guard let move = QCXChampion.move(for: action, phase: game.phase, rankCount: game.rankCount),
                  game.applyHumanMove(move, autoAdvanceBots: false) else { return nil }
        }
        return game
    }

    static func run(_ file: FixtureFile) -> Report {
        var report = Report()
        for (index, fixture) in file.fixtures.enumerated() {
            report.fixtures += 1
            guard fixture.view.count == QCXView.length else {
                report.viewMismatches += 1
                report.failures.append("#\(index) view length \(fixture.view.count)")
                continue
            }
            let legal = QCXChampion.legalActions(view: fixture.view).sorted()
            if legal != fixture.legal {
                report.legalMismatches += 1
                report.failures.append("#\(index) legal \(legal) != \(fixture.legal)")
            }
            let native = QCXChampion.decide(
                view: fixture.view, worlds: fixture.worlds, util: fixture.util, seed: fixture.seed
            )
            if native.action != fixture.action {
                report.actionMismatches += 1
                report.failures.append("#\(index) action \(native.action) != \(fixture.action)")
            }
            guard let game = replay(fixture), game.currentPlayer == fixture.me else {
                report.replayFailures += 1
                report.failures.append("#\(index) replay failed")
                continue
            }
            if QCXView.build(game: game, player: fixture.me) != fixture.view {
                report.viewMismatches += 1
                report.failures.append("#\(index) swift view differs")
            }
            if fixture.util == QCXChampion.config.util {
                let routed = QCXChampion.decide(
                    game: game, player: fixture.me, worlds: fixture.worlds, seed: fixture.seed
                )
                if routed?.action != fixture.action {
                    report.swiftRouteMismatches += 1
                    report.failures.append("#\(index) swift route \(String(describing: routed?.action)) != \(fixture.action)")
                }
            }
        }
        return report
    }
}
