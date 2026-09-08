import Foundation
import CryptoKit
import CoreML

/// Serial inference keeps Core ML and the policy caches off the UI thread.
enum ChampionInference {
    static let queue = DispatchQueue(label: "com.canal.quantumcat.inference", qos: .userInitiated)
    static let receiptKey = "champion1052.native-search-parity.v1"
    static var verified: Bool { UserDefaults.standard.bool(forKey: receiptKey) }
    private static let lock = NSRecursiveLock()
    private static var search: QuantumCatChampionSearch?
    private static var cached: [Data: Move] = [:]

    private static func key(_ state: QuantumCatMLLiveState) throws -> Data {
        let encoder = JSONEncoder(); encoder.outputFormatting = [.sortedKeys]
        return try encoder.encode(state)
    }

    static func cachedMove(_ state: QuantumCatMLLiveState) -> Move? {
        lock.lock(); defer { lock.unlock() }
        guard let key = try? key(state) else { return nil }
        return cached[key]
    }

    static func move(_ state: QuantumCatMLLiveState) -> Move? {
        lock.lock(); defer { lock.unlock() }
        guard verified, state.numPlayers == 5, state.phase == 3,
              let key = try? key(state) else { return nil }
        if let move = cached[key] { return move }
        do {
            if search == nil {
                guard let url = Bundle.main.url(forResource: "champion_1052_search", withExtension: "mlmodelc") else {
                    throw ChampionSearchError.missingModel
                }
                search = QuantumCatChampionSearch(network: try ChampionSearchNetwork(url: url))
            }
            let started = Date()
            let decision = try search!.decide(state)
            guard let move = QuantumCatMLFeatureEncoder.move(forPythonAction: decision.action, state: state) else {
                throw ChampionSearchError.invalidOutput
            }
            if cached.count >= 256 { cached.removeAll(keepingCapacity: true) }
            cached[key] = move
            print("CHAMPION_1052 round=\(state.matchRound) trick=\(state.trickNumber) calls=\(decision.neuralCalls) seconds=\(Date().timeIntervalSince(started))")
            return move
        } catch {
            print("CHAMPION_1052_FAILURE \(error)")
            return nil
        }
    }
}

/// NumPy RandomState's MT19937, including its shuffle and categorical sampling.
/// Search seeds depend only on the acting player's observation and legal actions.
struct ChampionRandom {
    var words = [UInt32](repeating: 0, count: 624)
    var position = 624
    init(seed: UInt32) {
        words[0] = seed
        for i in 1..<624 {
            words[i] = 1812433253 &* (words[i - 1] ^ (words[i - 1] >> 30)) &+ UInt32(i)
        }
    }
    mutating func next() -> UInt32 {
        if position == 624 {
            for i in 0..<624 {
                let y = (words[i] & 0x80000000) | (words[(i + 1) % 624] & 0x7fffffff)
                words[i] = words[(i + 397) % 624] ^ (y >> 1) ^ (y & 1 == 0 ? 0 : 0x9908b0df)
            }
            position = 0
        }
        var y = words[position]
        position += 1
        y ^= y >> 11
        y ^= (y << 7) & 0x9d2c5680
        y ^= (y << 15) & 0xefc60000
        y ^= y >> 18
        return y
    }
    mutating func interval(_ upper: Int) -> Int {
        var mask = UInt32(upper)
        mask |= mask >> 1; mask |= mask >> 2; mask |= mask >> 4
        mask |= mask >> 8; mask |= mask >> 16
        while true {
            let value = next() & mask
            if value <= upper { return Int(value) }
        }
    }
    mutating func shuffle(_ values: inout [Int]) {
        guard values.count > 1 else { return }
        for i in stride(from: values.count - 1, through: 1, by: -1) {
            values.swapAt(i, interval(i))
        }
    }
    mutating func uniform() -> Double {
        let a = Double(next() >> 5), b = Double(next() >> 6)
        return (a * 67108864 + b) / 9007199254740992
    }
    mutating func choice(_ count: Int) -> Int {
        let p = 1.0 / Double(count)
        var cdf = [Double](), sum = 0.0
        for _ in 0..<count { sum += p; cdf.append(sum) }
        let u = uniform()
        return cdf.firstIndex(where: { $0 / sum > u }) ?? count - 1
    }
}

enum ChampionSearchError: Error {
    case invalidState(String)
    case missingModel
    case invalidOutput
}

/// A hypothetical world, never constructed from the app's actual opponent hands.
struct ChampionWorld {
    var phase: Int
    var current: Int
    var trickNumber: Int
    var start: Int
    var roundStart: Int
    var led: String?
    var trick: [QuantumCatMLPlayedCard?]
    var played: [Int]
    var predictions: [Int]
    var won: [Int]
    var board: [[Int]]
    var tokens: [[Bool]]
    var hands: [[Int]]
    var discards: [Int]
    var trump: Bool
    var totals: [Double]
    var round: Int
    var paradox = [Bool](repeating: false, count: 5)
    var roundScores = [Double](repeating: 0, count: 5)

    init(view: QuantumCatMLLiveState, rng: inout ChampionRandom) throws {
        guard view.numPlayers == 5, view.numRanks == 9, view.numTricks == 7,
              view.phase == 3, view.actingPlayer == view.currentPlayer else {
            throw ChampionSearchError.invalidState("five-player play view required")
        }
        phase = view.phase; current = view.currentPlayer; trickNumber = view.trickNumber
        start = view.startPlayer; roundStart = view.roundStartPlayer; led = view.ledColor
        trick = view.currentTrick; played = view.playedCounts; predictions = view.predictions
        won = view.tricksWon; board = view.board; tokens = view.colorTokens
        trump = view.trumpBroken; totals = view.matchTotals; round = view.matchRound
        hands = Array(repeating: Array(repeating: 0, count: 9), count: 5)
        discards = Array(repeating: -1, count: 5)
        hands[view.actingPlayer] = view.actingHand
        discards[view.actingPlayer] = view.actingDiscardedRank
        var unknown = [Int]()
        for rank in 0..<9 {
            let used = board.reduce(0) { $0 + ($1[rank] >= 0 ? 1 : 0) }
            let count = 5 - used - view.actingHand[rank] - (view.actingDiscardedRank == rank + 1 ? 1 : 0)
            guard count >= 0 else { throw ChampionSearchError.invalidState("card conservation") }
            unknown += Array(repeating: rank, count: count)
        }
        rng.shuffle(&unknown)
        var offset = 0
        for player in 0..<5 where player != view.actingPlayer {
            let count = 8 - played[player]
            guard count >= 0, offset + 1 + count <= unknown.count else {
                throw ChampionSearchError.invalidState("unknown hand size")
            }
            discards[player] = unknown[offset] + 1; offset += 1
            for _ in 0..<count { hands[player][unknown[offset]] += 1; offset += 1 }
        }
        guard offset == unknown.count else { throw ChampionSearchError.invalidState("unused cards") }
    }

    init(round: Int, start: Int, totals: [Double], rng: inout ChampionRandom) {
        self.round = round; self.start = start; roundStart = start; self.totals = totals
        phase = 1; current = 0; trickNumber = 0; led = nil; trump = false
        trick = Array(repeating: nil, count: 5); played = Array(repeating: 0, count: 5)
        predictions = Array(repeating: -1, count: 5); won = Array(repeating: 0, count: 5)
        board = Array(repeating: Array(repeating: -1, count: 9), count: 4)
        tokens = Array(repeating: Array(repeating: true, count: 4), count: 5)
        hands = Array(repeating: Array(repeating: 0, count: 9), count: 5)
        discards = Array(repeating: -1, count: 5)
        var deck = (0..<9).flatMap { Array(repeating: $0, count: 5) }
        for i in 0..<45 {
            deck.swapAt(i, i + rng.choice(45 - i))
            hands[i % 5][deck[i]] += 1
        }
    }

    var view: QuantumCatMLLiveState {
        QuantumCatMLLiveState(
            numPlayers: 5, numRanks: 9, numColors: 4, numTricks: 7,
            cardsPerPlayerInitial: 9, phase: phase, currentPlayer: current,
            actingPlayer: current, trickNumber: trickNumber, startPlayer: start,
            roundStartPlayer: roundStart, ledColor: led, currentTrick: trick,
            playedCounts: played, predictions: predictions, tricksWon: won,
            board: board, colorTokens: tokens, actingHand: hands[current],
            actingDiscardedRank: discards[current], trumpBroken: trump,
            matchTotals: totals, matchRound: round, isTerminal: phase == 4
        )
    }

    static func legal(_ view: QuantumCatMLLiveState) -> [Int] {
        if view.phase == 1 { return view.actingHand.indices.filter { view.actingHand[$0] > 0 } }
        if view.phase == 2 { return [101, 102, 103, 104] }
        guard view.phase == 3 else { return [] }
        var actions = [Int]()
        for color in 0..<4 where view.colorTokens[view.actingPlayer][color] {
            for rank in 0..<9 where view.actingHand[rank] > 0 && view.board[color][rank] == -1 {
                actions.append(color * 9 + rank)
            }
        }
        if view.ledColor == nil, !view.board[0].contains(where: { $0 >= 0 }), actions.contains(where: { $0 >= 9 }) {
            actions.removeAll { $0 < 9 }
        }
        return actions.isEmpty ? [999] : actions
    }

    static func winner(_ cards: [QuantumCatMLPlayedCard?], led: String?) -> Int? {
        guard cards.allSatisfy({ $0 != nil }) else { return nil }
        let color = cards.contains(where: { $0?.suit == "R" }) ? "R" : led
        return cards.indices.filter { color == nil || cards[$0]?.suit == color }
            .max { cards[$0]!.rank < cards[$1]!.rank }
    }

    mutating func apply(_ action: Int) throws {
        guard Self.legal(view).contains(action) else { throw ChampionSearchError.invalidState("illegal simulation action") }
        if action == 999 { paradox[current] = true; finish(); return }
        if phase == 1 {
            hands[current][action] -= 1; discards[current] = action + 1
            if discards.allSatisfy({ $0 > 0 }) { phase = 2; current = roundStart }
            else { current = (current + 1) % 5 }
            return
        }
        if phase == 2 {
            predictions[current] = action - 100
            if predictions.allSatisfy({ $0 >= 1 }) { phase = 3; current = roundStart; start = roundStart }
            else { current = (current + 1) % 5 }
            return
        }
        let color = action / 9, rank = action % 9, suit = Suit.allCases[color].rawValue
        hands[current][rank] -= 1; board[color][rank] = current
        trick[current] = QuantumCatMLPlayedCard(rank: rank + 1, suit: suit)
        played[current] += 1
        if color == 0 { trump = true }
        if let led, led != suit { tokens[current][Suit.allCases.firstIndex(where: { $0.rawValue == led })!] = false }
        if led == nil { led = suit }
        current = (current + 1) % 5
        if current == start {
            guard let winner = Self.winner(trick, led: led) else { throw ChampionSearchError.invalidState("incomplete trick") }
            won[winner] += 1; trickNumber += 1; start = winner; current = winner
            led = nil; trick = Array(repeating: nil, count: 5)
            if trickNumber == 7 { finish() }
        }
    }

    mutating func finish() {
        phase = 4
        for player in 0..<5 {
            let bonus = won[player] == predictions[player]
                ? QuantumCatGame.competitiveLargestCluster(board: board, player: player) : 0
            roundScores[player] = Double(paradox[player] ? -won[player] : won[player] + bonus)
            totals[player] += roundScores[player]
        }
    }
}

/// Full-size Run1052 network, with only unused action padding reduced from 1000 to 36.
/// No weights, architecture, or numerical precision are distilled or quantized.
final class ChampionSearchNetwork {
    let model: MLModel
    init(url: URL) throws {
        let config = MLModelConfiguration(); config.computeUnits = .cpuOnly
        let compiled = url.pathExtension == "mlmodelc" ? url : try MLModel.compileModel(at: url)
        model = try MLModel(contentsOf: compiled, configuration: config)
    }
    func evaluate(_ state: QuantumCatMLLiveState, legal: [Int]) throws -> (policy: [Double], base: Int) {
        let moves = legal.compactMap { QuantumCatMLFeatureEncoder.move(forPythonAction: $0, state: state) }
        let observation = try QuantumCatMLFeatureEncoder.observation(state: state)
        let rows = try QuantumCatMLFeatureEncoder.actionFeatures(state: state, legalMoves: moves)
        let obs = try MLMultiArray(shape: [1, 113], dataType: .float32)
        let features = try MLMultiArray(shape: [1, 36, 214], dataType: .float32)
        let op = obs.dataPointer.assumingMemoryBound(to: Float.self)
        let fp = features.dataPointer.assumingMemoryBound(to: Float.self)
        for i in 0..<113 { op[i] = Float(observation[i]) }
        fp.initialize(repeating: 0, count: 36 * 214)
        for (i, row) in rows.enumerated() { for (j, value) in row.enumerated() { fp[i * 214 + j] = Float(value) } }
        let inputs = try MLDictionaryFeatureProvider(dictionary: ["observation": obs, "action_features": features])
        let result = try model.prediction(from: inputs)
        func values(_ key: String) throws -> [Double] {
            guard let array = result.featureValue(for: key)?.multiArrayValue else { throw ChampionSearchError.invalidOutput }
            return legal.indices.map { array[$0].doubleValue }
        }
        let policy = try values("policy_logits")
        let q = try values("action_values"), risks = try values("action_paradox_logits")
        guard policy.allSatisfy(\.isFinite), q.allSatisfy(\.isFinite), risks.allSatisfy(\.isFinite) else {
            throw ChampionSearchError.invalidOutput
        }
        let best: Int
        if state.phase == 2 { best = policy.indices.max { policy[$0] < policy[$1] } ?? 0 }
        else {
            guard let index = QuantumCatMLPolicy.selectFivePlayerSpecialistActionIndex(
                policyLogits: policy, actionValues: q, actionParadoxLogits: risks, actionFeatures: rows
            ) else { throw ChampionSearchError.invalidOutput }
            best = index
        }
        return (policy, legal[best])
    }
}

struct ChampionSearchDecision: Codable {
    let action: Int
    let baseline: Int
    let seed: UInt32?
    let candidates: [Int]
    let searchScores: [String: [Double]]
    let auditScores: [String: [Double]]
    let neuralCalls: Int
}

final class QuantumCatChampionSearch {
    let network: ChampionSearchNetwork
    private var neuralCalls = 0
    init(network: ChampionSearchNetwork) { self.network = network }

    static func discard(_ state: QuantumCatMLLiveState, legal: [Int]) -> Int {
        let maximum = legal.map { state.actingHand[$0] }.max()!
        var candidates = legal.filter { state.actingHand[$0] == maximum }
        if candidates.count > 1 { candidates.removeAll { $0 == 5 } }
        return candidates.max { left, right in
            func distance(_ rank: Int) -> Int {
                state.actingHand.indices.reduce(0) { $0 + abs($1 - rank) * state.actingHand[$1] }
            }
            let l = distance(left), r = distance(right)
            return l == r ? left > right : l < r
        }!
    }

    func continuation(_ state: QuantumCatMLLiveState) throws -> Int {
        let legal = ChampionWorld.legal(state)
        if legal.count == 1 { return legal[0] }
        if state.phase == 1 { return Self.discard(state, legal: legal) }
        neuralCalls += 1
        let evaluation = try network.evaluate(state, legal: legal)
        return tactical(state, legal: legal, base: evaluation.base)
    }

    private func tactical(_ state: QuantumCatMLLiveState, legal: [Int], base: Int) -> Int {
        let moves = legal.compactMap { QuantumCatMLFeatureEncoder.move(forPythonAction: $0, state: state) }
        guard let index = legal.firstIndex(of: base),
              let selected = QuantumCatGame.competitiveTacticalMove(state: state, legalMoves: moves, baseMove: moves[index]),
              let selectedIndex = moves.firstIndex(of: selected) else { return base }
        return legal[selectedIndex]
    }

    static func seed(_ state: QuantumCatMLLiveState, legal: [Int]) throws -> UInt32 {
        var bytes = Data()
        for value in try QuantumCatMLFeatureEncoder.observation(state: state) {
            var bits = Float(value).bitPattern.littleEndian
            withUnsafeBytes(of: &bits) { bytes.append(contentsOf: $0) }
        }
        for action in legal.sorted() {
            var bits = Int32(action).littleEndian
            withUnsafeBytes(of: &bits) { bytes.append(contentsOf: $0) }
        }
        bytes.append(contentsOf: "\(state.actingPlayer)0".utf8)
        return Array(SHA256.hash(data: bytes).prefix(4)).enumerated().reduce(UInt32(0)) { $0 | UInt32($1.element) << (8 * $1.offset) }
    }

    struct Safety {
        let feasible: Bool
        let buffer: Int
        let margin: Int
        let lostToken: Bool
        let breaksBid: Bool
        let responsibleWin: Bool
    }
    static func safety(_ state: QuantumCatMLLiveState, action: Int) -> Safety {
        let color = action / 9, rank = action % 9, player = state.actingPlayer
        var counts = state.actingHand, board = state.board, tokens = state.colorTokens[player]
        counts[rank] -= 1; board[color][rank] = player
        let ledIndex = Suit.allCases.firstIndex { $0.rawValue == state.ledColor }
        let lost = ledIndex.map { $0 != color && tokens[$0] } ?? false
        if let ledIndex, ledIndex != color { tokens[ledIndex] = false }
        let caps = counts.indices.map { rank in (0..<4).reduce(0) { $0 + (tokens[$1] && board[$1][rank] == -1 ? 1 : 0) } }
        var cards = state.currentTrick
        cards[player] = QuantumCatMLPlayedCard(rank: rank + 1, suit: Suit.allCases[color].rawValue)
        let wins = ChampionWorld.winner(cards, led: state.ledColor ?? cards[state.startPlayer]?.suit) == player
        let breaksBid = wins && state.tricksWon[player] == state.predictions[player]
        return Safety(
            feasible: counts.indices.allSatisfy { counts[$0] <= caps[$0] },
            buffer: counts.indices.reduce(0) { $0 + (counts[$1] > 0 ? max(0, counts[$1] + 1 - caps[$1]) : 0) },
            margin: QuantumCatGame.competitiveAdversarialMargin(counts: counts, caps: caps, playsRemaining: 7 - state.trickNumber - 1),
            lostToken: lost, breaksBid: breaksBid, responsibleWin: lost && wins && !breaksBid
        )
    }
    static func safeCandidates(_ state: QuantumCatMLLiveState, candidates: [Int], baseline: Int) -> [Int] {
        let base = safety(state, action: baseline)
        return candidates.filter { action in
            if action == baseline { return true }
            let row = safety(state, action: action)
            return !(base.feasible && !row.feasible) && row.margin >= base.margin && row.buffer <= base.buffer
                && !(row.lostToken && !base.lostToken && !row.responsibleWin)
                && !(row.breaksBid && !base.breaksBid)
        }
    }

    private func rollout(_ world: ChampionWorld, action: Int, actor: Int, rng: inout ChampionRandom) throws -> [Double] {
        var world = world
        let initialStart = (world.roundStart - world.round + 5) % 5
        try world.apply(action)
        while world.phase != 4 { try world.apply(continuation(world.view)) }
        let ownParadox = world.paradox[actor] ? 1.0 : 0.0
        while world.round < 4 {
            let next = world.round + 1
            world = ChampionWorld(round: next, start: (initialStart + next) % 5, totals: world.totals, rng: &rng)
            while world.phase != 4 { try world.apply(continuation(world.view)) }
        }
        let scores = zip(world.totals, world.roundScores).map { $0 * 1000 + $1 }
        let wins = scores.filter { scores[actor] > $0 }.count
        let losses = scores.filter { scores[actor] < $0 }.count
        return [ownParadox, Double(wins - losses) * 5]
    }

    private func score(_ state: QuantumCatMLLiveState, actions: [Int], rng: inout ChampionRandom) throws -> [String: [Double]] {
        var worlds = [ChampionWorld]()
        for _ in 0..<8 { worlds.append(try ChampionWorld(view: state, rng: &rng)) }
        var result = Dictionary(uniqueKeysWithValues: actions.map { (String($0), [0.0, 0.0]) })
        for world in worlds {
            let saved = rng
            for action in actions {
                rng = saved
                let value = try rollout(world, action: action, actor: state.actingPlayer, rng: &rng)
                result[String(action)]![0] += value[0] / 8
                result[String(action)]![1] += value[1] / 8
            }
        }
        return result
    }

    func decide(_ state: QuantumCatMLLiveState) throws -> ChampionSearchDecision {
        neuralCalls = 0
        let legal = ChampionWorld.legal(state)
        guard !legal.isEmpty else { throw ChampionSearchError.invalidState("terminal decision") }
        if legal.count == 1 || state.phase == 1 {
            let action = state.phase == 1 ? Self.discard(state, legal: legal) : legal[0]
            return ChampionSearchDecision(action: action, baseline: action, seed: nil, candidates: [], searchScores: [:], auditScores: [:], neuralCalls: 0)
        }
        neuralCalls += 1
        let evaluation = try network.evaluate(state, legal: legal)
        let baseline = tactical(state, legal: legal, base: evaluation.base)
        guard state.phase == 3, (4...6).contains(state.trickNumber), baseline == evaluation.base else {
            return ChampionSearchDecision(action: baseline, baseline: baseline, seed: nil, candidates: [], searchScores: [:], auditScores: [:], neuralCalls: neuralCalls)
        }
        let seed = try Self.seed(state, legal: legal)
        var rng = ChampionRandom(seed: seed)
        var candidates = [baseline]
        let top = legal.indices.sorted {
            evaluation.policy[$0] == evaluation.policy[$1] ? legal[$0] < legal[$1] : evaluation.policy[$0] > evaluation.policy[$1]
        }
        for index in top.prefix(2) where candidates.count < 2 && !candidates.contains(legal[index]) { candidates.append(legal[index]) }
        candidates = Self.safeCandidates(state, candidates: candidates.sorted(), baseline: baseline)
        // A singleton cannot change the selected action. Its RNG is local to this
        // decision, so skipping unused rollouts has no effect on subsequent choices.
        if candidates.count == 1 {
            return ChampionSearchDecision(action: baseline, baseline: baseline, seed: seed, candidates: candidates, searchScores: [:], auditScores: [:], neuralCalls: neuralCalls)
        }
        let scores = try score(state, actions: candidates, rng: &rng)
        var selected = candidates[0]
        for action in candidates.dropFirst() {
            let left = scores[String(selected)]!, right = scores[String(action)]!
            if right[0] < left[0] || (right[0] == left[0] && (right[1] > left[1]
                || (right[1] == left[1] && evaluation.policy[legal.firstIndex(of: action)!] > evaluation.policy[legal.firstIndex(of: selected)!]))) { selected = action }
        }
        var audit = [String: [Double]]()
        if selected != baseline {
            audit = try score(state, actions: [baseline, selected], rng: &rng)
            let base = audit[String(baseline)]!, proposal = audit[String(selected)]!
            if proposal[0] > base[0] || proposal[1] < base[1] { selected = baseline }
        }
        return ChampionSearchDecision(action: selected, baseline: baseline, seed: seed, candidates: candidates, searchScores: scores, auditScores: audit, neuralCalls: neuralCalls)
    }
}
