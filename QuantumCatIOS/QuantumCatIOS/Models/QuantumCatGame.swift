import Combine
import Foundation

enum GamePhase: String, Codable {
    case dealing
    case discard
    case prediction
    case play
    case scoring

    var title: String {
        switch self {
        case .dealing: "Deal"
        case .discard: "Discard"
        case .prediction: "Bid"
        case .play: "Play"
        case .scoring: "Score"
        }
    }
}

enum Suit: String, CaseIterable, Identifiable, Codable {
    case red = "R"
    case blue = "B"
    case yellow = "Y"
    case green = "G"

    var id: String { rawValue }
}

struct PlayedCard: Equatable, Codable {
    let rank: Int
    let suit: Suit
}

enum Move: Identifiable, Equatable {
    case discard(rank: Int)
    case prediction(Int)
    case play(rank: Int, suit: Suit)
    case paradox

    var id: String {
        switch self {
        case .discard(let rank): "discard-\(rank)"
        case .prediction(let value): "bid-\(value)"
        case .play(let rank, let suit): "play-\(suit.rawValue)-\(rank)"
        case .paradox: "paradox"
        }
    }

    var primary: String {
        switch self {
        case .discard: "Discard"
        case .prediction: "Bid"
        case .play(_, let suit): suit.rawValue
        case .paradox: "Paradox"
        }
    }

    var value: String {
        switch self {
        case .discard(let rank), .play(let rank, _): "\(rank)"
        case .prediction(let value): "\(value)"
        case .paradox: "!"
        }
    }
}

struct PlayerState: Identifiable, Codable {
    let id: Int
    var seatKind: SeatKind
    var hand: [Int]
    var discardedRank: Int?
    var prediction: Int?
    var tricksWon: Int
    var hasParadoxed: Bool
    var colorTokens: [Bool]
    var score: Double?
}

struct GameLogEntry: Identifiable, Codable {
    let id: UUID
    let text: String
    let kind: String

    init(id: UUID = UUID(), text: String, kind: String) {
        self.id = id
        self.text = text
        self.kind = kind
    }
}

enum BotOnlyRunMode: String, CaseIterable, Identifiable {
    case watch
    case bulk

    var id: String { rawValue }

    var title: String {
        switch self {
        case .watch: "Watch"
        case .bulk: "Simulate"
        }
    }
}

struct BulkSimulationSummary: Identifiable {
    struct PlayerResult: Identifiable {
        let id: Int
        let botName: String
        let wins: Int
        let paradoxes: Int
        let paradoxRate: Double
        let averageScore: Double
        let bestScore: Double
    }

    let id = UUID()
    let games: Int
    let players: [PlayerResult]
    let totalParadoxes: Int
    let gamesWithParadox: Int
    let seatParadoxRate: Double
    let gameParadoxRate: Double
    let mlUsage: [MLPolicyUsageSnapshot]
    let generatedAt: Date
}

struct SeededGenerator: Codable {
    private var state: UInt64

    init(seed: Int) {
        state = UInt64(bitPattern: Int64(seed == 0 ? 0xC0FFEE : seed))
    }

    mutating func next() -> UInt64 {
        state = state &* 6364136223846793005 &+ 1442695040888963407
        return state
    }

    mutating func int(upperBound: Int) -> Int {
        guard upperBound > 1 else { return 0 }
        return Int(next() % UInt64(upperBound))
    }

    mutating func double() -> Double {
        Double(next() % 10_000) / 10_000.0
    }
}

@MainActor
final class GameStore: ObservableObject {
    private static let savedStateKey = "quantum-cat.saved-state.v1"
    private struct PersistedState: Codable {
        var humanSeats: Int
        var botSeats: Int
        var botKinds: [BotKind]
        var game: QuantumCatGame
    }

    @Published var humanSeats: Int = 1 {
        didSet { normalizeSetup() }
    }
    @Published var botSeats: Int = 2 {
        didSet { normalizeSetup() }
    }
    @Published var botKinds: [BotKind] = [.championBeliefPolicy, .championBeliefPolicy, .heuristicTarget, .heuristicAdjacency, .random] {
        didSet { normalizeSetup() }
    }
    @Published var botOnlyRunMode: BotOnlyRunMode = .watch
    @Published var bulkSimulationGames: Int = 100
    @Published private(set) var game: QuantumCatGame
    @Published private(set) var botStatus: String?
    @Published private(set) var bulkSimulationSummary: BulkSimulationSummary?
    private var isNormalizingSetup = false
    private var botTask: Task<Void, Never>?

    init() {
        if ProcessInfo.processInfo.arguments.contains("-uiTestReset") {
            UserDefaults.standard.removeObject(forKey: Self.savedStateKey)
        }

        if ProcessInfo.processInfo.arguments.contains("-uiTestMixedRoster") {
            humanSeats = 2
            botSeats = 3
            botKinds = [.championBeliefPolicy, .setPoolDistill, .strictQHead, .heuristicAdjacency, .random]
            game = QuantumCatGame(
                seats: [.human, .human, .bot(.championBeliefPolicy), .bot(.setPoolDistill), .bot(.strictQHead)],
                seed: 20260612,
                autoAdvanceBots: false
            )
        } else if ProcessInfo.processInfo.arguments.contains("-uiTestBotOnlyBulk") {
            humanSeats = 0
            botSeats = 3
            botOnlyRunMode = .bulk
            bulkSimulationGames = 5
            botKinds = [.championBeliefPolicy, .setPoolDistill, .strictQHead, .heuristicAdjacency, .random]
            game = QuantumCatGame(
                seats: [.bot(.championBeliefPolicy), .bot(.setPoolDistill), .bot(.strictQHead)],
                seed: 20260613,
                autoAdvanceBots: false
            )
        } else if ProcessInfo.processInfo.arguments.contains("-uiTestRun374Benchmark") {
            let seats = Array(repeating: SeatKind.bot(.championBeliefPolicy), count: 5)
            humanSeats = 0
            botSeats = seats.count
            botOnlyRunMode = .bulk
            bulkSimulationGames = 20
            botKinds = Array(repeating: .championBeliefPolicy, count: 5)
            game = QuantumCatGame(
                seats: seats,
                seed: 20263740,
                autoAdvanceBots: true
            )
            bulkSimulationSummary = runBulkSimulation(seats: seats, baseSeed: 20263740, games: bulkSimulationGames)
        } else if let data = UserDefaults.standard.data(forKey: Self.savedStateKey),
           let state = try? JSONDecoder().decode(PersistedState.self, from: data) {
            humanSeats = state.humanSeats
            botSeats = state.botSeats
            botKinds = state.botKinds
            game = state.game
            normalizeSetup()
        } else {
            game = QuantumCatGame(seats: [.human, .bot(.championBeliefPolicy), .bot(.championBeliefPolicy)], seed: 20260606, autoAdvanceBots: false)
        }
        triggerBotAdvance()
    }

    deinit {
        botTask?.cancel()
    }

    var setupIsValid: Bool {
        let total = humanSeats + botSeats
        return (0...5).contains(humanSeats) && (0...5).contains(botSeats) && (2...5).contains(total)
    }

    func newGame() {
        normalizeSetup()
        let seats = Array(repeating: SeatKind.human, count: humanSeats)
            + (0..<botSeats).map { SeatKind.bot(botKinds[$0]) }
        botTask?.cancel()
        botStatus = nil
        bulkSimulationSummary = nil
        let seed = Int(Date().timeIntervalSince1970) % 1_000_000
        if humanSeats == 0, botOnlyRunMode == .bulk {
            game = QuantumCatGame(seats: seats, seed: seed, autoAdvanceBots: true)
            bulkSimulationSummary = runBulkSimulation(seats: seats, baseSeed: seed, games: bulkSimulationGames)
        } else {
            game = QuantumCatGame(seats: seats, seed: seed, autoAdvanceBots: false)
        }
        save()
        triggerBotAdvance()
    }

    func apply(_ move: Move) {
        game.applyHumanMove(move, autoAdvanceBots: false)
        save()
        triggerBotAdvance()
    }

    func save() {
        let state = PersistedState(humanSeats: humanSeats, botSeats: botSeats, botKinds: botKinds, game: game)
        if let data = try? JSONEncoder().encode(state) {
            UserDefaults.standard.set(data, forKey: Self.savedStateKey)
        }
    }

    private func normalizeSetup() {
        guard !isNormalizingSetup else { return }
        isNormalizingSetup = true
        defer { isNormalizingSetup = false }

        var normalizedHumans = min(max(humanSeats, 0), 5)
        var normalizedBots = min(max(botSeats, 0), 5)
        while normalizedHumans + normalizedBots > 5 {
            if normalizedBots > 0 {
                normalizedBots -= 1
            } else {
                normalizedHumans -= 1
            }
        }
        if normalizedHumans == 0, normalizedBots < 2 {
            normalizedBots = 2
        } else if normalizedHumans + normalizedBots < 2 {
            normalizedBots = min(5 - normalizedHumans, 2 - normalizedHumans)
        }
        if humanSeats != normalizedHumans { humanSeats = normalizedHumans }
        if botSeats != normalizedBots { botSeats = normalizedBots }
        while botKinds.count < normalizedBots {
            botKinds.append(.championBeliefPolicy)
        }
        if botKinds.count > 5 {
            botKinds = Array(botKinds.prefix(5))
        }
    }

    private func runBulkSimulation(seats: [SeatKind], baseSeed: Int, games: Int) -> BulkSimulationSummary {
        let gameCount = min(max(games, 1), 10_000)
        var totalScores = Array(repeating: 0.0, count: seats.count)
        var bestScores = Array(repeating: -Double.infinity, count: seats.count)
        var wins = Array(repeating: 0, count: seats.count)
        var paradoxes = Array(repeating: 0, count: seats.count)
        var gamesWithParadox = 0

        QuantumCatMLPolicy.shared.resetUsage()

        for index in 0..<gameCount {
            let simulated = QuantumCatGame(seats: seats, seed: baseSeed + index + 1, autoAdvanceBots: true)
            let scores = simulated.players.map { $0.score ?? 0.0 }
            let playerParadoxes = simulated.players.map(\.hasParadoxed)
            if playerParadoxes.contains(true) {
                gamesWithParadox += 1
            }
            for playerIndex in scores.indices {
                totalScores[playerIndex] += scores[playerIndex]
                bestScores[playerIndex] = max(bestScores[playerIndex], scores[playerIndex])
                if playerParadoxes[playerIndex] {
                    paradoxes[playerIndex] += 1
                }
            }
            if let topScore = scores.max() {
                for playerIndex in scores.indices where scores[playerIndex] == topScore {
                    wins[playerIndex] += 1
                }
            }
        }

        let totalParadoxes = paradoxes.reduce(0, +)
        let seatCount = max(1, gameCount * max(1, seats.count))
        let playerResults = seats.enumerated().map { index, seat in
            let name: String
            if case .bot(let kind) = seat {
                name = kind.name
            } else {
                name = "Human"
            }
            return BulkSimulationSummary.PlayerResult(
                id: index,
                botName: name,
                wins: wins[index],
                paradoxes: paradoxes[index],
                paradoxRate: Double(paradoxes[index]) / Double(max(1, gameCount)),
                averageScore: totalScores[index] / Double(gameCount),
                bestScore: bestScores[index].isFinite ? bestScores[index] : 0
            )
        }

        let summary = BulkSimulationSummary(
            games: gameCount,
            players: playerResults,
            totalParadoxes: totalParadoxes,
            gamesWithParadox: gamesWithParadox,
            seatParadoxRate: Double(totalParadoxes) / Double(seatCount),
            gameParadoxRate: Double(gamesWithParadox) / Double(gameCount),
            mlUsage: QuantumCatMLPolicy.shared.usageSnapshot(),
            generatedAt: Date()
        )
        let coreMLSuccesses = summary.mlUsage.reduce(0) { $0 + $1.successes }
        let coreMLAttempts = summary.mlUsage.reduce(0) { $0 + $1.attempts }
        print(
            "BULK_SIMULATION_SUMMARY games=\(summary.games) players=\(seats.count) " +
            "seat_paradox_rate=\(String(format: "%.4f", summary.seatParadoxRate)) " +
            "game_paradox_rate=\(String(format: "%.4f", summary.gameParadoxRate)) " +
            "coreml_successes=\(coreMLSuccesses) coreml_attempts=\(coreMLAttempts)"
        )
        return summary
    }

    private func triggerBotAdvance() {
        botTask?.cancel()
        botTask = Task { @MainActor [weak self] in
            await self?.advanceVisibleBots()
        }
    }

    private func advanceVisibleBots() async {
        var guardCount = 0
        while !Task.isCancelled, game.activeBotKind != nil, !game.isTerminal {
            guardCount += 1
            guard guardCount <= 200 else {
                botStatus = "Bot guard stopped an unexpected loop."
                return
            }
            let player = game.currentPlayer
            let botName = game.activeBotKind?.name ?? "Bot"
            botStatus = "P\(player) \(botName) is thinking..."
            try? await Task.sleep(nanoseconds: 450_000_000)
            guard !Task.isCancelled else { return }
            let move = game.applyBotMoveForCurrentPlayer()
            botStatus = move.map { "P\(player) played \($0.primary.lowercased()) \($0.value)." } ?? nil
            save()
            try? await Task.sleep(nanoseconds: 180_000_000)
        }
        if !Task.isCancelled {
            botStatus = nil
        }
    }
}

struct QuantumCatGame: Codable {
    private(set) var seats: [SeatKind]
    private(set) var phase: GamePhase = .dealing
    private(set) var players: [PlayerState] = []
    private(set) var rankCount: Int
    private(set) var tricksInRound: Int
    private(set) var currentPlayer: Int = 0
    private(set) var trickNumber: Int = 0
    private(set) var ledSuit: Suit?
    private(set) var board: [[Int]]
    private(set) var currentTrick: [PlayedCard?]
    private(set) var completedTricks: [[PlayedCard?]] = []
    private(set) var log: [GameLogEntry] = []

    private var deck: [Int] = []
    private var rng: SeededGenerator
    private var startPlayer: Int = 0
    private var trumpBroken = false
    private static let blocked = -2

    init(seats: [SeatKind], seed: Int, autoAdvanceBots: Bool = true) {
        self.seats = seats
        let playerCount = seats.count
        self.rankCount = Self.rankCount(for: playerCount)
        self.tricksInRound = Self.tricksInRound(for: playerCount)
        self.board = Array(repeating: Array(repeating: -1, count: Self.rankCount(for: playerCount)), count: Suit.allCases.count)
        self.currentTrick = Array(repeating: nil, count: playerCount)
        self.rng = SeededGenerator(seed: seed)
        self.players = seats.enumerated().map { index, kind in
            PlayerState(
                id: index,
                seatKind: kind,
                hand: Array(repeating: 0, count: Self.rankCount(for: playerCount)),
                discardedRank: nil,
                prediction: nil,
                tricksWon: 0,
                hasParadoxed: false,
                colorTokens: Array(repeating: true, count: Suit.allCases.count),
                score: nil
            )
        }
        deal(seed: seed)
        if autoAdvanceBots {
            advanceBots()
        }
    }

    static func rankCount(for players: Int) -> Int {
        switch players {
        case 2: 5
        case 3: 6
        case 4: 8
        default: 9
        }
    }

    static func cardsPerPlayer(for players: Int) -> Int {
        players == 5 ? 9 : 10
    }

    static func tricksInRound(for players: Int) -> Int {
        switch players {
        case 2: 9
        case 5: 7
        default: 8
        }
    }

    var isTerminal: Bool { phase == .scoring }
    var activeHuman: PlayerState? {
        guard !isTerminal, players.indices.contains(currentPlayer), players[currentPlayer].seatKind.isHuman else { return nil }
        return players[currentPlayer]
    }
    var activeBotKind: BotKind? {
        guard !isTerminal, players.indices.contains(currentPlayer), case .bot(let kind) = players[currentPlayer].seatKind else {
            return nil
        }
        return kind
    }

    var legalMoves: [Move] {
        guard !isTerminal else { return [] }
        switch phase {
        case .discard:
            return players[currentPlayer].hand.enumerated()
                .filter { $0.element > 0 }
                .map { .discard(rank: $0.offset + 1) }
        case .prediction:
            return [1, 2, 3, 4].map(Move.prediction)
        case .play:
            return legalPlays(for: currentPlayer)
        default:
            return []
        }
    }

    mutating func applyHumanMove(_ move: Move, autoAdvanceBots: Bool = true) {
        guard players.indices.contains(currentPlayer), players[currentPlayer].seatKind.isHuman else { return }
        apply(move)
        if autoAdvanceBots {
            advanceBots()
        }
    }

    mutating func applyBotMoveForCurrentPlayer() -> Move? {
        guard activeBotKind != nil else { return nil }
        let move = botMove(for: currentPlayer)
        apply(move)
        return move
    }

    mutating private func deal(seed: Int) {
        deck = (1...rankCount).flatMap { rank in Array(repeating: rank, count: 5) }
        shuffleDeck()
        let count = Self.cardsPerPlayer(for: players.count)
        for _ in 0..<count {
            for playerIndex in players.indices {
                guard !deck.isEmpty else { continue }
                let rank = deck.removeFirst()
                players[playerIndex].hand[rank - 1] += 1
            }
        }
        if players.count == 2 {
            handleTwoPlayerLeftover()
            phase = .play
            currentPlayer = startPlayer
        } else {
            phase = .discard
            currentPlayer = 0
        }
        let humanCount = players.filter { $0.seatKind.isHuman }.count
        log.append(GameLogEntry(text: "New game: \(humanCount) human, \(players.count - humanCount) bot.", kind: "system"))
    }

    mutating private func shuffleDeck() {
        guard deck.count > 1 else { return }
        for index in deck.indices.reversed() {
            let other = rng.int(upperBound: index + 1)
            deck.swapAt(index, other)
        }
    }

    mutating private func handleTwoPlayerLeftover() {
        let blockRows = [3, 2, 1]
        var seen: [Int: Int] = [:]
        for rank in deck.prefix(3) {
            let count = seen[rank, default: 0]
            guard count < blockRows.count else { continue }
            board[blockRows[count]][rank - 1] = Self.blocked
            seen[rank] = count + 1
        }
        deck.removeAll()
    }

    mutating private func advanceBots() {
        var guardCount = 0
        while !isTerminal, players.indices.contains(currentPlayer), !players[currentPlayer].seatKind.isHuman {
            guardCount += 1
            if guardCount > 200 {
                log.append(GameLogEntry(text: "Bot guard stopped an unexpected loop.", kind: "system"))
                return
            }
            let move = botMove(for: currentPlayer)
            apply(move)
        }
    }

    mutating private func apply(_ move: Move) {
        switch move {
        case .discard(let rank):
            players[currentPlayer].hand[rank - 1] -= 1
            players[currentPlayer].discardedRank = rank
            log.append(GameLogEntry(text: "P\(currentPlayer) discarded \(rank).", kind: playerLogKind))
            advanceDiscard()
        case .prediction(let prediction):
            players[currentPlayer].prediction = prediction
            log.append(GameLogEntry(text: "P\(currentPlayer) bid \(prediction).", kind: playerLogKind))
            advancePrediction()
        case .play(let rank, let suit):
            play(rank: rank, suit: suit)
        case .paradox:
            players[currentPlayer].hasParadoxed = true
            log.append(GameLogEntry(text: "P\(currentPlayer) paradoxed.", kind: playerLogKind))
            finishRound()
        }
        if log.count > 80 {
            log.removeFirst(log.count - 80)
        }
    }

    private var playerLogKind: String {
        players[currentPlayer].seatKind.isHuman ? "human" : "bot"
    }

    mutating private func advanceDiscard() {
        if players.allSatisfy({ $0.discardedRank != nil }) {
            phase = .prediction
            currentPlayer = startPlayer
        } else {
            currentPlayer = (currentPlayer + 1) % players.count
        }
    }

    mutating private func advancePrediction() {
        if players.allSatisfy({ $0.prediction != nil }) {
            phase = .play
            trickNumber = 0
            currentPlayer = startPlayer
            ledSuit = nil
            currentTrick = Array(repeating: nil, count: players.count)
        } else {
            currentPlayer = (currentPlayer + 1) % players.count
        }
    }

    mutating private func play(rank: Int, suit: Suit) {
        guard let suitIndex = Suit.allCases.firstIndex(of: suit) else { return }
        players[currentPlayer].hand[rank - 1] -= 1
        board[suitIndex][rank - 1] = currentPlayer
        currentTrick[currentPlayer] = PlayedCard(rank: rank, suit: suit)
        if suit == .red { trumpBroken = true }
        if ledSuit == nil {
            ledSuit = suit
        } else if suit != ledSuit, let ledIndex = Suit.allCases.firstIndex(of: ledSuit!) {
            players[currentPlayer].colorTokens[ledIndex] = false
        }
        log.append(GameLogEntry(text: "P\(currentPlayer) played \(suit.rawValue)\(rank).", kind: playerLogKind))
        currentPlayer = (currentPlayer + 1) % players.count
        if currentPlayer == startPlayer {
            finishTrick()
        }
    }

    mutating private func finishTrick() {
        let winner = trickWinner()
        players[winner].tricksWon += 1
        completedTricks.append(currentTrick)
        trickNumber += 1
        log.append(GameLogEntry(text: "P\(winner) won trick \(trickNumber).", kind: "system"))
        if trickNumber >= tricksInRound {
            finishRound()
            return
        }
        startPlayer = winner
        currentPlayer = winner
        ledSuit = nil
        currentTrick = Array(repeating: nil, count: players.count)
    }

    private func trickWinner() -> Int {
        let plays = currentTrick.enumerated().compactMap { index, card in card.map { (index, $0) } }
        let red = plays.filter { $0.1.suit == .red }
        if let bestRed = red.max(by: { $0.1.rank < $1.1.rank }) { return bestRed.0 }
        let lead = ledSuit ?? plays.first!.1.suit
        return plays.filter { $0.1.suit == lead }.max(by: { $0.1.rank < $1.1.rank })!.0
    }

    mutating private func finishRound() {
        var raw = Array(repeating: 0.0, count: players.count)
        for index in players.indices {
            if players[index].hasParadoxed {
                raw[index] = -Double(players[index].tricksWon)
                continue
            }
            let tricks = players[index].tricksWon
            if players.count == 2 {
                raw[index] = Double(tricks + (tricks <= 4 ? largestCluster(for: index) : 0))
            } else if players[index].prediction == tricks {
                raw[index] = Double(tricks + largestCluster(for: index))
            } else {
                raw[index] = Double(tricks)
            }
        }
        if players.count == 2 {
            let diff = raw[0] - raw[1]
            raw[0] = diff
            raw[1] = -diff
        }
        for index in players.indices {
            players[index].score = raw[index]
        }
        phase = .scoring
        log.append(GameLogEntry(text: "Round finished.", kind: "system"))
    }

    private func legalPlays(for player: Int) -> [Move] {
        var moves: [Move] = []
        for rankIndex in 0..<rankCount where players[player].hand[rankIndex] > 0 {
            for suitIndex in Suit.allCases.indices where players[player].colorTokens[suitIndex] && board[suitIndex][rankIndex] == -1 {
                let suit = Suit.allCases[suitIndex]
                if ledSuit == nil, suit == .red, !redRowOccupied() {
                    continue
                }
                moves.append(.play(rank: rankIndex + 1, suit: suit))
            }
        }
        if moves.isEmpty {
            for rankIndex in 0..<rankCount where players[player].hand[rankIndex] > 0 {
                for suitIndex in Suit.allCases.indices where players[player].colorTokens[suitIndex] && board[suitIndex][rankIndex] == -1 {
                    moves.append(.play(rank: rankIndex + 1, suit: Suit.allCases[suitIndex]))
                }
            }
        }
        return moves.isEmpty ? [.paradox] : moves
    }

    private func redRowOccupied() -> Bool {
        board[0].contains { $0 >= 0 }
    }

    private func largestCluster(for player: Int, board candidateBoard: [[Int]]? = nil) -> Int {
        let source = candidateBoard ?? board
        var seen = Array(repeating: Array(repeating: false, count: rankCount), count: Suit.allCases.count)
        var best = 0
        for suitIndex in Suit.allCases.indices {
            for rankIndex in 0..<rankCount where source[suitIndex][rankIndex] == player && !seen[suitIndex][rankIndex] {
                var stack = [(suitIndex, rankIndex)]
                seen[suitIndex][rankIndex] = true
                var size = 0
                while let item = stack.popLast() {
                    size += 1
                    for delta in [(1, 0), (-1, 0), (0, 1), (0, -1)] {
                        let nextSuit = item.0 + delta.0
                        let nextRank = item.1 + delta.1
                        guard Suit.allCases.indices.contains(nextSuit), (0..<rankCount).contains(nextRank) else { continue }
                        if !seen[nextSuit][nextRank], source[nextSuit][nextRank] == player {
                            seen[nextSuit][nextRank] = true
                            stack.append((nextSuit, nextRank))
                        }
                    }
                }
                best = max(best, size)
            }
        }
        return best
    }

    mutating private func botMove(for player: Int) -> Move {
        guard case .bot(let kind) = players[player].seatKind else { return legalMoves.first ?? .paradox }
        let legal = legalMoves
        guard !legal.isEmpty else { return .paradox }
        if kind == .random {
            return legal[rng.int(upperBound: legal.count)]
        }
        let modelPreferredMove = QuantumCatMLPolicy.shared.chooseMove(kind: kind, game: self, player: player, legalMoves: legal)
        let weights = kind.weights
        let scored = legal.map { move -> (Double, Move) in
            let modelBonus = move == modelPreferredMove ? 4.0 : 0.0
            return (score(move, for: player, weights: weights) + modelBonus + rng.double() * weights.noise, move)
        }
        if phase == .play {
            let ordered = scored.sorted { $0.0 > $1.0 }
            let lookaheadCandidates = ordered.prefix(12)
            if let safeMove = lookaheadCandidates.first(where: { avoidsParadoxAfterMove($0.1) }) {
                return safeMove.1
            }
        }
        return scored.max(by: { $0.0 < $1.0 })?.1 ?? legal[0]
    }

    private func avoidsParadoxAfterMove(_ move: Move) -> Bool {
        var copy = self
        copy.apply(move)
        if copy.players.contains(where: \.hasParadoxed) {
            return false
        }
        if copy.isTerminal {
            return true
        }
        return copy.avoidsParadoxWithGreedyContinuation()
    }

    private func avoidsParadoxWithGreedyContinuation() -> Bool {
        var copy = self
        var guardCount = 0
        while !copy.isTerminal {
            guardCount += 1
            guard guardCount <= 200 else { return false }
            let legal = copy.legalMoves
            if legal == [.paradox] {
                return false
            }
            guard let next = copy.greedyMobilityMove(from: legal) else {
                return false
            }
            copy.apply(next)
        }
        return !copy.players.contains { $0.hasParadoxed }
    }

    private func greedyMobilityMove(from legal: [Move]) -> Move? {
        legal.max { lhs, rhs in
            greedyMobilityScore(lhs) < greedyMobilityScore(rhs)
        }
    }

    private func greedyMobilityScore(_ move: Move) -> Double {
        var copy = self
        copy.apply(move)
        if copy.players.contains(where: \.hasParadoxed) {
            return -10_000
        }
        if copy.isTerminal {
            return 10_000
        }
        return -copy.positionParadoxPressure()
    }

    private func positionParadoxPressure() -> Double {
        guard phase == .play else { return 0.0 }
        var pressure = 0.0
        for playerIndex in players.indices {
            let count = legalPlayCount(for: playerIndex)
            if count == 0 {
                pressure += 120.0
            } else {
                pressure += 18.0 / Double(count)
            }
        }
        return pressure
    }

    private func legalPlayCount(for player: Int) -> Int {
        let moves = legalPlays(for: player)
        return moves == [.paradox] ? 0 : moves.count
    }

    private func score(_ move: Move, for player: Int, weights: BotWeights) -> Double {
        switch move {
        case .discard(let rank):
            let count = players[player].hand[rank - 1]
            return Double(count * 10 - rank)
        case .prediction(let value):
            let guess = sharedPredictionBid(for: player)
            return -Double(abs(value - guess))
        case .play(let rank, let suit):
            let suitIndex = Suit.allCases.firstIndex(of: suit)!
            let rankIndex = rank - 1
            var score = 0.0
            let wantsMore = players[player].prediction == nil || players[player].tricksWon < (players[player].prediction ?? 0)
            if let win = wouldWin(player: player, rank: rank, suit: suit) {
                score += weights.winner * ((win == wantsMore) ? 1.0 : -1.0)
            }
            var trial = board
            trial[suitIndex][rankIndex] = player
            score += weights.adjacency * Double(largestCluster(for: player, board: trial) - largestCluster(for: player))
            let highness = Double(rank) / Double(rankCount)
            score += weights.target * weights.highCard * (wantsMore ? highness : -highness)
            if suit == .red, ledSuit == nil, !redRowOccupied() {
                score -= 0.25
            }
            let completesTrick = currentTrick.enumerated().allSatisfy { index, card in
                index == player || card != nil
            }
            score -= paradoxRiskPenalty(
                board: trial,
                ledSuit: completesTrick ? nil : (ledSuit ?? suit),
                consumed: (player: player, rankIndex: rankIndex)
            )
            return score
        case .paradox:
            return -1_000
        }
    }

    private func sharedPredictionBid(for player: Int) -> Int {
        let hand = players[player].hand
        guard !hand.isEmpty else { return 1 }
        let topRankIndex = min(rankCount, hand.count) - 1
        let secondRankIndex = topRankIndex - 1
        var expectedTricks = Double(hand[topRankIndex])
        if secondRankIndex >= 0 {
            expectedTricks += (player == startPlayer ? 0.5 : 0.25) * Double(hand[secondRankIndex])
        }
        return max(1, min(4, Int(expectedTricks.rounded())))
    }

    private func paradoxRiskPenalty(board candidateBoard: [[Int]], ledSuit candidateLedSuit: Suit?, consumed: (player: Int, rankIndex: Int)) -> Double {
        var penalty = 0.0
        for playerIndex in players.indices {
            let count = potentialPlayCount(for: playerIndex, board: candidateBoard, ledSuit: candidateLedSuit, consumed: consumed)
            if count == 0 {
                penalty += 60.0
            } else if count == 1 {
                penalty += 16.0
            } else if count == 2 {
                penalty += 5.0
            }
        }
        return penalty
    }

    private func potentialPlayCount(for player: Int, board candidateBoard: [[Int]], ledSuit candidateLedSuit: Suit?, consumed: (player: Int, rankIndex: Int)) -> Int {
        func remainingCount(rankIndex: Int) -> Int {
            let used = player == consumed.player && rankIndex == consumed.rankIndex ? 1 : 0
            return max(0, players[player].hand[rankIndex] - used)
        }

        func count(allowClosedRedLead: Bool) -> Int {
            var total = 0
            for rankIndex in 0..<rankCount where remainingCount(rankIndex: rankIndex) > 0 {
                for suitIndex in Suit.allCases.indices where players[player].colorTokens[suitIndex] && candidateBoard[suitIndex][rankIndex] == -1 {
                    let suit = Suit.allCases[suitIndex]
                    if !allowClosedRedLead, candidateLedSuit == nil, suit == .red, !redRowOccupied(board: candidateBoard) {
                        continue
                    }
                    total += 1
                }
            }
            return total
        }

        let normal = count(allowClosedRedLead: false)
        return normal > 0 ? normal : count(allowClosedRedLead: true)
    }

    private func redRowOccupied(board candidateBoard: [[Int]]) -> Bool {
        candidateBoard[0].contains { $0 >= 0 }
    }

    private func wouldWin(player: Int, rank: Int, suit: Suit) -> Bool? {
        var plays = currentTrick
        plays[player] = PlayedCard(rank: rank, suit: suit)
        guard plays.allSatisfy({ $0 != nil }) else { return nil }
        var copy = self
        copy.currentTrick = plays
        copy.ledSuit = ledSuit ?? suit
        return copy.trickWinner() == player
    }
}
