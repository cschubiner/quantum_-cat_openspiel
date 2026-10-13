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

    var sortValue: Int {
        switch self {
        case .discard(let rank):
            rank
        case .prediction(let value):
            100 + value
        case .play(let rank, let suit):
            ((Suit.allCases.firstIndex(of: suit) ?? 0) * 16) + rank - 1
        case .paradox:
            999
        }
    }

    var telemetryKey: String {
        switch self {
        case .discard(let rank): "discard:\(rank)"
        case .prediction(let value): "prediction:\(value)"
        case .play(let rank, let suit): "play:\(suit.rawValue):\(rank)"
        case .paradox: "paradox"
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

struct BotMoveAnimation: Identifiable, Equatable {
    let id = UUID()
    let player: Int
    let botName: String
    let move: Move

    var title: String {
        "P\(player) \(botName)"
    }

    var subtitle: String {
        switch move {
        case .discard:
            "discarded rank \(move.value)"
        case .prediction:
            "bid \(move.value)"
        case .play:
            "played \(move.primary)\(move.value)"
        case .paradox:
            "paradox"
        }
    }
}

struct BotMovePreference: Identifiable, Equatable {
    let move: Move
    let score: Int

    var id: String { move.id }
}

struct BotRecommendation: Identifiable, Equatable {
    let player: Int
    let botKind: BotKind
    let move: Move
    let legalMoveCount: Int
    let rankedMoves: [BotMovePreference]

    var id: String {
        "\(player)-\(botKind.id)-\(move.id)"
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

struct HumanMatchTelemetry: Codable, Identifiable {
    struct Seat: Codable, Equatable {
        let index: Int
        let participantID: String
        let actor: String
        let botKind: String?
        let modelResource: String?
        let modelCheckpointSHA256: String?
        let liveInputContract: String?
    }

    struct MoveRecord: Codable, Equatable {
        let sequence: Int
        let roundIndex: Int
        let phase: String
        let player: Int
        let actor: String
        let action: String
        let legalActions: [String]
        let legalActionCount: Int
        let trickNumber: Int
        let ledSuit: String?
        let championRecommendation: String?
        let agreedWithChampion: Bool?
    }

    struct RoundRecord: Codable, Equatable {
        let index: Int
        let startingPlayer: Int
        let startedAt: Date
        let completedAt: Date
        let moves: [MoveRecord]
        let scores: [Double]
        let paradoxes: [Bool]
        let winner: Int?
    }

    static let schemaVersion = 1

    let id: UUID
    let schemaVersion: Int
    let createdAt: Date
    var completedAt: Date?
    let appVersion: String
    let buildVersion: String
    let seats: [Seat]
    let targetRounds: Int
    var rounds: [RoundRecord]
    var currentRoundIndex: Int
    var currentRoundStartingPlayer: Int
    var currentRoundStartedAt: Date
    var currentRoundMoves: [MoveRecord]

    var isComplete: Bool {
        completedAt != nil && rounds.count >= targetRounds
    }

    var cumulativeScores: [Double] {
        var totals = Array(repeating: 0.0, count: seats.count)
        for round in rounds where round.scores.count == totals.count {
            for index in totals.indices {
                totals[index] += round.scores[index]
            }
        }
        return totals
    }

    var winnerSeats: [Int] {
        let totals = cumulativeScores
        guard let best = totals.max() else { return [] }
        return totals.indices.filter { totals[$0] == best }
    }
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
    /// Bumped whenever a new champion ships. Saved rosters that still name the
    /// previous champion are moved to the new one once, so the app always
    /// plays the latest champion; an explicit later pick of an older bot sticks.
    static let championGenerationKey = "quantum-cat.champion-generation"
    static let currentChampionGeneration = "qcx-pimc-v1"
    private static let telemetryFileName = "human_match_telemetry.jsonl"
    private struct PersistedState: Codable {
        var humanSeats: Int
        var humanParticipantIDs: [String]?
        var botSeats: Int
        var botKinds: [BotKind]
        var game: QuantumCatGame
        var humanMatchTelemetry: HumanMatchTelemetry?
    }

    @Published var humanSeats: Int = 1 {
        didSet { normalizeSetup() }
    }
    @Published var humanParticipantIDs: [String] = (1...5).map { "player-\($0)" } {
        didSet { normalizeSetup() }
    }
    @Published var botSeats: Int = 2 {
        didSet { normalizeSetup() }
    }
    @Published var botKinds: [BotKind] = Array(repeating: .qcxChampion, count: 5) {
        didSet { normalizeSetup() }
    }
    @Published var botOnlyRunMode: BotOnlyRunMode = .watch
    @Published var bulkSimulationGames: Int = 100
    @Published private(set) var game: QuantumCatGame
    @Published private(set) var botStatus: String?
    @Published private(set) var botMoveAnimation: BotMoveAnimation?
    @Published private(set) var bulkSimulationSummary: BulkSimulationSummary?
    @Published private(set) var humanMatchTelemetry: HumanMatchTelemetry?
    @Published private(set) var latestTelemetryExportURL: URL?
    @Published private(set) var telemetryStatus: String?
    private var isNormalizingSetup = false
    private var botTask: Task<Void, Never>?
    private let telemetryDirectoryURL: URL

    init(telemetryDirectoryURL: URL? = nil) {
        self.telemetryDirectoryURL = telemetryDirectoryURL
            ?? FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first
            ?? FileManager.default.temporaryDirectory
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
            humanParticipantIDs = state.humanParticipantIDs ?? (1...5).map { "player-\($0)" }
            botSeats = state.botSeats
            var savedKinds = state.botKinds
            var savedGame = state.game
            if UserDefaults.standard.string(forKey: Self.championGenerationKey) != Self.currentChampionGeneration {
                savedKinds = savedKinds.map { $0 == .championBeliefPolicy ? .qcxChampion : $0 }
                savedGame.replaceBotKind(.championBeliefPolicy, with: .qcxChampion)
            }
            botKinds = savedKinds
            game = savedGame
            humanMatchTelemetry = state.humanMatchTelemetry
            normalizeSetup()
        } else {
            game = QuantumCatGame(seats: [.human, .bot(.qcxChampion), .bot(.qcxChampion)], seed: 20260606, autoAdvanceBots: false)
        }
        UserDefaults.standard.set(Self.currentChampionGeneration, forKey: Self.championGenerationKey)
        prepareTelemetryForCurrentGameIfNeeded(startingNewRound: false)
        triggerBotAdvance()
    }

    deinit {
        botTask?.cancel()
    }

    var setupIsValid: Bool {
        let total = humanSeats + botSeats
        let activeHumanIDs = (0..<humanSeats).map { humanParticipantID(at: $0) }
        return (0...5).contains(humanSeats)
            && (0...5).contains(botSeats)
            && (2...5).contains(total)
            && Set(activeHumanIDs).count == activeHumanIDs.count
    }

    func newGame() {
        finalizeTelemetryRoundIfNeeded()
        normalizeSetup()
        let seats = Array(repeating: SeatKind.human, count: humanSeats)
            + (0..<botSeats).map { SeatKind.bot(botKinds[$0]) }
        let nextStartingPlayer = seats.isEmpty ? 0 : game.roundWinner.map { $0 % seats.count } ?? 0
        botTask?.cancel()
        botStatus = nil
        botMoveAnimation = nil
        bulkSimulationSummary = nil
        let seed = Int(Date().timeIntervalSince1970) % 1_000_000
        if humanSeats == 0, botOnlyRunMode == .bulk {
            game = QuantumCatGame(seats: seats, seed: seed, startingPlayer: nextStartingPlayer, autoAdvanceBots: true)
            bulkSimulationSummary = runBulkSimulation(seats: seats, baseSeed: seed, games: bulkSimulationGames)
        } else {
            game = QuantumCatGame(seats: seats, seed: seed, startingPlayer: nextStartingPlayer, autoAdvanceBots: false)
        }
        prepareTelemetryForCurrentGameIfNeeded(startingNewRound: true)
        save()
        triggerBotAdvance()
    }

    func apply(_ move: Move) {
        guard game.activeHuman != nil else { return }
        // Taking a human move must not launch a new multi-second search merely
        // for telemetry. Recommendations are computed asynchronously by the UI.
        let context = telemetryMoveContext(move: move, recommendation: nil)
        guard game.applyHumanMove(move, autoAdvanceBots: false) else { return }
        appendTelemetryMove(context)
        finalizeTelemetryRoundIfNeeded()
        save()
        triggerBotAdvance()
    }

    func save() {
        let state = PersistedState(
            humanSeats: humanSeats,
            humanParticipantIDs: humanParticipantIDs,
            botSeats: botSeats,
            botKinds: botKinds,
            game: game,
            humanMatchTelemetry: humanMatchTelemetry
        )
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
            botKinds.append(.qcxChampion)
        }
        while humanParticipantIDs.count < 5 {
            humanParticipantIDs.append("player-\(humanParticipantIDs.count + 1)")
        }
        if humanParticipantIDs.count > 5 {
            humanParticipantIDs = Array(humanParticipantIDs.prefix(5))
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
            let context = telemetryMoveContext(move: nil, recommendation: nil)
            botStatus = "P\(player) \(botName) is thinking..."
            try? await Task.sleep(nanoseconds: 450_000_000)
            guard !Task.isCancelled else { return }
            let snapshot = game
            let result: (QuantumCatGame, Move?) = await withCheckedContinuation { continuation in
                ChampionInference.queue.async {
                    var updated = snapshot
                    let move = updated.applyBotMoveForCurrentPlayer()
                    continuation.resume(returning: (updated, move))
                }
            }
            guard !Task.isCancelled else { return }
            game = result.0
            let move = result.1
            if let move {
                appendTelemetryMove(context.with(move: move))
                finalizeTelemetryRoundIfNeeded()
                let event = BotMoveAnimation(player: player, botName: botName, move: move)
                botMoveAnimation = event
                botStatus = "\(event.title) \(event.subtitle)."
            } else {
                botStatus = nil
            }
            save()
            try? await Task.sleep(nanoseconds: 620_000_000)
            if !Task.isCancelled, botMoveAnimation?.player == player, botMoveAnimation?.move == move {
                botMoveAnimation = nil
            }
        }
        if !Task.isCancelled {
            botStatus = nil
            botMoveAnimation = nil
        }
    }

    private struct PendingTelemetryMove {
        let phase: String
        let player: Int
        let actor: String
        let move: Move?
        let legalMoves: [Move]
        let trickNumber: Int
        let ledSuit: String?
        let recommendation: Move?

        func with(move: Move) -> PendingTelemetryMove {
            PendingTelemetryMove(
                phase: phase,
                player: player,
                actor: actor,
                move: move,
                legalMoves: legalMoves,
                trickNumber: trickNumber,
                ledSuit: ledSuit,
                recommendation: recommendation
            )
        }
    }

    private func telemetryMoveContext(
        move: Move?,
        recommendation: BotRecommendation?
    ) -> PendingTelemetryMove {
        PendingTelemetryMove(
            phase: game.phase.rawValue,
            player: game.currentPlayer,
            actor: game.players[game.currentPlayer].seatKind.isHuman ? "human" : "bot",
            move: move,
            legalMoves: game.legalMoves,
            trickNumber: game.trickNumber,
            ledSuit: game.ledSuit?.rawValue,
            recommendation: recommendation?.move
        )
    }

    private func appendTelemetryMove(_ pending: PendingTelemetryMove) {
        guard var telemetry = humanMatchTelemetry, let move = pending.move else { return }
        let record = HumanMatchTelemetry.MoveRecord(
            sequence: telemetry.currentRoundMoves.count,
            roundIndex: telemetry.currentRoundIndex,
            phase: pending.phase,
            player: pending.player,
            actor: pending.actor,
            action: move.telemetryKey,
            legalActions: pending.legalMoves.map(\.telemetryKey),
            legalActionCount: pending.legalMoves.count,
            trickNumber: pending.trickNumber,
            ledSuit: pending.ledSuit,
            championRecommendation: pending.recommendation?.telemetryKey,
            agreedWithChampion: pending.recommendation.map { $0 == move }
        )
        telemetry.currentRoundMoves.append(record)
        humanMatchTelemetry = telemetry
    }

    private func prepareTelemetryForCurrentGameIfNeeded(startingNewRound: Bool) {
        guard game.seats.contains(where: { $0.isHuman }) else {
            humanMatchTelemetry = nil
            telemetryStatus = nil
            return
        }
        let seats = telemetrySeats(for: game.seats)
        let canContinue = humanMatchTelemetry.map { telemetry in
            !telemetry.isComplete
                && telemetry.seats == seats
                && telemetry.rounds.count < telemetry.targetRounds
                && (!startingNewRound || telemetry.currentRoundMoves.isEmpty)
        } ?? false
        if canContinue, var telemetry = humanMatchTelemetry {
            if startingNewRound {
                telemetry.currentRoundIndex = telemetry.rounds.count
                telemetry.currentRoundStartingPlayer = game.currentPlayer
                telemetry.currentRoundStartedAt = Date()
            }
            humanMatchTelemetry = telemetry
        } else if humanMatchTelemetry == nil
                    || humanMatchTelemetry?.isComplete == true
                    || humanMatchTelemetry?.seats != seats
                    || startingNewRound {
            let version = Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String ?? "development"
            let build = Bundle.main.infoDictionary?["CFBundleVersion"] as? String ?? "development"
            humanMatchTelemetry = HumanMatchTelemetry(
                id: UUID(),
                schemaVersion: HumanMatchTelemetry.schemaVersion,
                createdAt: Date(),
                completedAt: nil,
                appVersion: version,
                buildVersion: build,
                seats: seats,
                targetRounds: game.seats.count,
                rounds: [],
                currentRoundIndex: 0,
                currentRoundStartingPlayer: game.currentPlayer,
                currentRoundStartedAt: Date(),
                currentRoundMoves: []
            )
        }
        if let telemetry = humanMatchTelemetry {
            game.setMatchContext(
                totals: telemetry.cumulativeScores,
                roundIndex: telemetry.rounds.count
            )
            telemetryStatus = "Human match \(telemetry.rounds.count + 1)/\(telemetry.targetRounds)"
        }
    }

    private func finalizeTelemetryRoundIfNeeded() {
        guard game.isTerminal, var telemetry = humanMatchTelemetry else { return }
        guard telemetry.rounds.count == telemetry.currentRoundIndex else { return }
        let scores = game.players.map { $0.score ?? 0.0 }
        telemetry.rounds.append(
            HumanMatchTelemetry.RoundRecord(
                index: telemetry.currentRoundIndex,
                startingPlayer: telemetry.currentRoundStartingPlayer,
                startedAt: telemetry.currentRoundStartedAt,
                completedAt: Date(),
                moves: telemetry.currentRoundMoves,
                scores: scores,
                paradoxes: game.players.map(\.hasParadoxed),
                winner: game.roundWinner
            )
        )
        telemetry.currentRoundMoves = []
        if telemetry.rounds.count >= telemetry.targetRounds {
            telemetry.completedAt = Date()
            humanMatchTelemetry = telemetry
            persistCompletedTelemetry(telemetry)
            telemetryStatus = "Human match complete: \(telemetry.rounds.count) rounds"
        } else {
            humanMatchTelemetry = telemetry
            telemetryStatus = "Human match \(telemetry.rounds.count)/\(telemetry.targetRounds) rounds complete"
        }
    }

    private func telemetrySeats(for seats: [SeatKind]) -> [HumanMatchTelemetry.Seat] {
        seats.enumerated().map { index, seat in
            switch seat {
            case .human:
                return HumanMatchTelemetry.Seat(
                    index: index,
                    participantID: humanParticipantID(at: index),
                    actor: "human",
                    botKind: nil,
                    modelResource: nil,
                    modelCheckpointSHA256: nil,
                    liveInputContract: nil
                )
            case .bot(let kind) where kind == .qcxChampion && QCXChampion.supports(players: seats.count):
                let config = QCXChampion.config
                let version = config.sourceSHA256.map { String($0.prefix(16)) } ?? "unknown"
                let inputContract = "qcx-view\(config.viewLen)-worlds\(config.worlds)-util\(config.util)"
                return HumanMatchTelemetry.Seat(
                    index: index,
                    participantID: "bot:\(kind.rawValue):\(version):\(inputContract)",
                    actor: "bot",
                    botKind: kind.rawValue,
                    modelResource: "qcx_champion",
                    modelCheckpointSHA256: config.sourceSHA256,
                    liveInputContract: inputContract
                )
            case .bot(let kind):
                let provenance = QuantumCatMLPolicy.shared.provenance(
                    for: kind,
                    players: seats.count
                )
                let resource = kind == .championBeliefPolicy
                    && seats.count == 5
                    && QuantumCatMLPolicy.shared.usesPlayerCountSpecialist(
                        for: kind,
                        players: seats.count
                    )
                    ? "champion_5p"
                    : kind.coreMLResource
                let version = provenance.checkpointSHA256 ?? "native"
                let inputContract = provenance.liveInputContract ?? "missing"
                return HumanMatchTelemetry.Seat(
                    index: index,
                    // A fail-closed fallback and verified live inference are
                    // different competitors even when they bundle the same
                    // checkpoint. Keep their Elo identities separate.
                    participantID: "bot:\(kind.rawValue):\(version):\(inputContract)",
                    actor: "bot",
                    botKind: kind.rawValue,
                    modelResource: resource,
                    modelCheckpointSHA256: provenance.checkpointSHA256,
                    liveInputContract: inputContract
                )
            }
        }
    }

    private func humanParticipantID(at index: Int) -> String {
        let fallback = "player-\(index + 1)"
        let raw = humanParticipantIDs.indices.contains(index)
            ? humanParticipantIDs[index]
            : fallback
        let normalized = raw
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased()
            .map { character in
                character.isLetter || character.isNumber || character == "-" || character == "_"
                    ? character
                    : "-"
            }
        let compact = String(normalized)
            .split(separator: "-", omittingEmptySubsequences: true)
            .joined(separator: "-")
        return "human:\(compact.isEmpty ? fallback : compact)"
    }

    private func persistCompletedTelemetry(_ telemetry: HumanMatchTelemetry) {
        do {
            try FileManager.default.createDirectory(
                at: telemetryDirectoryURL,
                withIntermediateDirectories: true
            )
            let encoder = JSONEncoder()
            encoder.dateEncodingStrategy = .iso8601
            encoder.outputFormatting = [.sortedKeys]
            let data = try encoder.encode(telemetry)
            let individualURL = telemetryDirectoryURL
                .appendingPathComponent("human_match_\(telemetry.id.uuidString).json")
            try data.write(to: individualURL, options: .atomic)
            let jsonlURL = telemetryDirectoryURL.appendingPathComponent(Self.telemetryFileName)
            var line = data
            line.append(0x0A)
            if FileManager.default.fileExists(atPath: jsonlURL.path) {
                let handle = try FileHandle(forWritingTo: jsonlURL)
                defer { try? handle.close() }
                try handle.seekToEnd()
                try handle.write(contentsOf: line)
            } else {
                try line.write(to: jsonlURL, options: .atomic)
            }
            latestTelemetryExportURL = individualURL
        } catch {
            telemetryStatus = "Telemetry export failed: \(error.localizedDescription)"
        }
    }
}

struct QuantumCatGame: Codable {
    static func wantsAdditionalTrick(prediction: Int?, tricksWon: Int) -> Bool {
        guard let prediction else { return true }
        return tricksWon != prediction
    }

    static func winningWouldBreakExactPrediction(
        prediction: Int?,
        tricksWon: Int,
        wouldWin: Bool
    ) -> Bool {
        guard let prediction else { return false }
        return wouldWin && tricksWon == prediction
    }

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
    private(set) var matchTotals: [Double]?
    private(set) var matchRound: Int?

    private var deck: [Int] = []
    private var rng: SeededGenerator
    private var startPlayer: Int = 0
    private var roundStartPlayer: Int?
    private var trumpBroken = false
    private static let blocked = -2

    init(
        seats: [SeatKind],
        seed: Int,
        startingPlayer: Int = 0,
        autoAdvanceBots: Bool = true,
        matchTotals: [Double]? = nil,
        matchRound: Int = 0
    ) {
        self.seats = seats
        let playerCount = seats.count
        self.rankCount = Self.rankCount(for: playerCount)
        self.tricksInRound = Self.tricksInRound(for: playerCount)
        self.board = Array(repeating: Array(repeating: -1, count: Self.rankCount(for: playerCount)), count: Suit.allCases.count)
        self.currentTrick = Array(repeating: nil, count: playerCount)
        self.rng = SeededGenerator(seed: seed)
        self.startPlayer = playerCount > 0 ? min(max(startingPlayer, 0), playerCount - 1) : 0
        self.roundStartPlayer = self.startPlayer
        self.matchTotals = matchTotals
        self.matchRound = matchRound
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
    var mlStartPlayer: Int { startPlayer }
    var mlRoundStartPlayer: Int { roundStartPlayer ?? startPlayer }
    var mlTrumpBroken: Bool { trumpBroken }
    var mlMatchTotals: [Double] {
        let totals = matchTotals ?? []
        return players.indices.map { totals.indices.contains($0) ? totals[$0] : 0.0 }
    }
    var mlMatchRound: Int { matchRound ?? 0 }
    var mlPlayedCounts: [Int] {
        players.indices.map { player in
            completedTricks.reduce(0) { count, trick in
                count + (trick.indices.contains(player) && trick[player] != nil ? 1 : 0)
            } + (currentTrick.indices.contains(player) && currentTrick[player] != nil ? 1 : 0)
        }
    }

    /// Swap one bot kind for another in this game's seats (champion upgrades).
    mutating func replaceBotKind(_ old: BotKind, with new: BotKind) {
        for index in seats.indices where seats[index] == .bot(old) {
            seats[index] = .bot(new)
        }
        for index in players.indices where players[index].seatKind == .bot(old) {
            players[index].seatKind = .bot(new)
        }
    }

    mutating func setMatchContext(totals: [Double], roundIndex: Int) {
        matchTotals = players.indices.map { totals.indices.contains($0) ? totals[$0] : 0.0 }
        matchRound = max(0, roundIndex)
    }
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
    var roundWinner: Int? {
        let scoredPlayers = players.filter { $0.score != nil }
        guard scoredPlayers.count == players.count else { return nil }
        return scoredPlayers.max { lhs, rhs in
            let lhsScore = lhs.score ?? -.infinity
            let rhsScore = rhs.score ?? -.infinity
            if lhsScore != rhsScore {
                return lhsScore < rhsScore
            }
            if lhs.tricksWon != rhs.tricksWon {
                return lhs.tricksWon < rhs.tricksWon
            }
            return lhs.id > rhs.id
        }?.id
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

    func botRecommendationForCurrentHuman(kind requestedKind: BotKind = .qcxChampion) -> BotRecommendation? {
        guard activeHuman != nil, seats.indices.contains(currentPlayer) else { return nil }
        let legal = legalMoves
        guard !legal.isEmpty else { return nil }

        if requestedKind == .qcxChampion,
           let decision = QCXChampion.decide(game: self, player: currentPlayer),
           legal.contains(decision.move) {
            // Coach percentages come straight from the search's mean utility
            // per legal move (the human's own information only).
            let values = legal.map { move -> Double in
                if let value = decision.values[QCXChampion.action(for: move, rankCount: rankCount)] {
                    return value
                }
                return move == decision.move ? 0.0 : -Double.infinity
            }
            return BotRecommendation(
                player: currentPlayer,
                botKind: .qcxChampion,
                move: decision.move,
                legalMoveCount: legal.count,
                rankedMoves: Self.normalizedMovePreferences(
                    legalMoves: legal,
                    preferenceValues: values,
                    preferredMove: decision.move
                )
            )
        }
        // Two-player tables (or a rejected view) use the previous champion.
        let kind = requestedKind == .qcxChampion ? BotKind.championBeliefPolicy : requestedKind

        var copy = self
        copy.seats[currentPlayer] = .bot(kind)
        copy.players[currentPlayer].seatKind = .bot(kind)
        let move = copy.botMove(for: currentPlayer)
        guard legal.contains(move) else { return nil }

        let preferenceValues: [Double]
        if phase == .play,
           let modelValues = QuantumCatMLPolicy.shared.movePreferenceLogits(
            kind: kind,
            game: copy,
            player: currentPlayer,
            legalMoves: legal
           ) {
            preferenceValues = modelValues
        } else if phase == .prediction,
                  QuantumCatMLPolicy.shared.usesLearnedPrediction(
                    for: kind,
                    players: players.count
                  ),
                  let modelValues = QuantumCatMLPolicy.shared.movePreferenceLogits(
                    kind: kind,
                    game: copy,
                    player: currentPlayer,
                    legalMoves: legal
                  ) {
            preferenceValues = modelValues
        } else if phase == .prediction {
            let expected = Double(copy.sharedPredictionBid(for: currentPlayer))
            preferenceValues = legal.map { action in
                guard case .prediction(let value) = action else {
                    return -Double.infinity
                }
                return -1.4 * abs(Double(value) - expected)
            }
        } else if phase == .discard {
            let hand = players[currentPlayer].hand
            preferenceValues = legal.map { action in
                guard case .discard(let rank) = action,
                      hand.indices.contains(rank - 1) else {
                    return -Double.infinity
                }
                let rankPenalty = rank == 6 ? 0.75 : 0.0
                return Double(hand[rank - 1])
                    + 0.15 * copy.discardDissimilarityScore(hand: hand, rank: rank)
                    - rankPenalty
            }
        } else if let rows = try? QuantumCatMLFeatureEncoder.actionFeatures(
            game: copy,
            player: currentPlayer,
            legalMoves: legal
        ) {
            preferenceValues = rows.map { row in
                guard row.count > 213 else { return 0.0 }
                return row[209] + row[199] - row[211] - row[213]
            }
        } else {
            preferenceValues = legal.map { $0 == move ? 1.0 : 0.0 }
        }

        return BotRecommendation(
            player: currentPlayer,
            botKind: kind,
            move: move,
            legalMoveCount: legal.count,
            rankedMoves: Self.normalizedMovePreferences(
                legalMoves: legal,
                preferenceValues: preferenceValues,
                preferredMove: move
            )
        )
    }

    static func normalizedMovePreferences(
        legalMoves: [Move],
        preferenceValues: [Double],
        preferredMove: Move
    ) -> [BotMovePreference] {
        guard legalMoves.count == preferenceValues.count,
              let preferredIndex = legalMoves.firstIndex(of: preferredMove) else {
            return []
        }

        var adjusted = preferenceValues.map { value in
            value.isFinite ? value : -Double.infinity
        }
        let highest = adjusted.filter(\.isFinite).max() ?? 0.0
        // Public risk/tactical protections can override the raw policy. The
        // coach must rank the move the champion actually plays first.
        if !adjusted[preferredIndex].isFinite || adjusted[preferredIndex] <= highest {
            adjusted[preferredIndex] = highest + 0.35
        }
        let finite = adjusted.filter(\.isFinite)
        let spread = (finite.max() ?? 0.0) - (finite.min() ?? 0.0)
        let temperature = max(1.0, min(6.0, spread / 5.0))
        let maximum = adjusted.max() ?? 0.0
        let weights = adjusted.map { value in
            value.isFinite ? exp(max(-60.0, (value - maximum) / temperature)) : 0.0
        }
        let total = weights.reduce(0.0, +)
        guard total > 0.0 else {
            return [BotMovePreference(move: preferredMove, score: 100)]
        }

        let exact = weights.map { 100.0 * $0 / total }
        var percentages = exact.map { Int($0.rounded(.down)) }
        let remainder = max(0, 100 - percentages.reduce(0, +))
        let remainderOrder = exact.indices.sorted { lhs, rhs in
            let leftFraction = exact[lhs] - Double(percentages[lhs])
            let rightFraction = exact[rhs] - Double(percentages[rhs])
            if leftFraction != rightFraction { return leftFraction > rightFraction }
            if lhs == preferredIndex { return true }
            if rhs == preferredIndex { return false }
            return lhs < rhs
        }
        for index in remainderOrder.prefix(remainder) {
            percentages[index] += 1
        }

        return legalMoves.indices.sorted { lhs, rhs in
            if lhs == preferredIndex { return true }
            if rhs == preferredIndex { return false }
            if percentages[lhs] != percentages[rhs] {
                return percentages[lhs] > percentages[rhs]
            }
            return lhs < rhs
        }.map { index in
            BotMovePreference(move: legalMoves[index], score: percentages[index])
        }
    }

    @discardableResult
    mutating func applyHumanMove(_ move: Move, autoAdvanceBots: Bool = true) -> Bool {
        guard players.indices.contains(currentPlayer),
              players[currentPlayer].seatKind.isHuman,
              legalMoves.contains(move) else { return false }
        apply(move)
        if autoAdvanceBots {
            advanceBots()
        }
        return true
    }

    mutating func applyBotMoveForCurrentPlayer() -> Move? {
        guard activeBotKind != nil else { return nil }
        let move = botMove(for: currentPlayer)
        guard legalMoves.contains(move) else { return nil }
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
            currentPlayer = startPlayer
        }
        let humanCount = players.filter { $0.seatKind.isHuman }.count
        log.append(GameLogEntry(text: "New game: \(humanCount) human, \(players.count - humanCount) bot. P\(startPlayer) starts.", kind: "system"))
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
        // Match OpenSpiel's sorted integer action ids: color-major, then rank.
        for suitIndex in Suit.allCases.indices where players[player].colorTokens[suitIndex] {
            for rankIndex in 0..<rankCount where players[player].hand[rankIndex] > 0 && board[suitIndex][rankIndex] == -1 {
                let suit = Suit.allCases[suitIndex]
                if ledSuit == nil, suit == .red, !redRowOccupied() {
                    continue
                }
                moves.append(.play(rank: rankIndex + 1, suit: suit))
            }
        }
        if moves.isEmpty {
            for suitIndex in Suit.allCases.indices where players[player].colorTokens[suitIndex] {
                for rankIndex in 0..<rankCount where players[player].hand[rankIndex] > 0 && board[suitIndex][rankIndex] == -1 {
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
        guard case .bot(var kind) = players[player].seatKind else { return legalMoves.first ?? .paradox }
        let legal = legalMoves
        guard !legal.isEmpty else { return .paradox }
        if kind == .qcxChampion {
            // The qcx champion decides discard, bid and every play itself from
            // this seat's information only. Two-player tables (unsupported by
            // the C++ engine) or a rejected view fall back to the previous
            // champion so a bot never stalls.
            if let decision = QCXChampion.decide(game: self, player: player),
               legal.contains(decision.move) {
                return decision.move
            }
            kind = .championBeliefPolicy
        }
        if phase == .discard, let sharedDiscard = sharedDiscardMove(for: player, legal: legal) {
            return sharedDiscard
        }
        if phase == .prediction {
            if QuantumCatMLPolicy.shared.usesLearnedPrediction(
                for: kind,
                players: players.count
            ),
               let learnedMove = QuantumCatMLPolicy.shared.chooseMove(
                kind: kind,
                game: self,
                player: player,
                legalMoves: legal
               ),
               legal.contains(learnedMove) {
                return learnedMove
            }
            let sharedBid = sharedPredictionBid(for: player)
            let sharedMove = Move.prediction(sharedBid)
            if legal.contains(sharedMove) {
                return sharedMove
            }
            return legal.min { lhs, rhs in
                predictionGap(lhs, target: sharedBid) < predictionGap(rhs, target: sharedBid)
            } ?? legal[0]
        }
        if kind == .random {
            return legal[rng.int(upperBound: legal.count)]
        }
        if kind == .championBeliefPolicy, players.count == 5, phase == .play,
           let move = ChampionInference.move(QuantumCatMLLiveState(game: self, player: player)),
           legal.contains(move) {
            return move
        }
        let modelPreferredMove = QuantumCatMLPolicy.shared.chooseMove(kind: kind, game: self, player: player, legalMoves: legal)
        let weights = kind.weights
        let scored = legal.map { move -> (Double, Move) in
            let modelBonus = move == modelPreferredMove ? 4.0 : 0.0
            return (score(move, for: player, weights: weights) + modelBonus + rng.double() * weights.noise, move)
        }
        if kind == .championBeliefPolicy, phase == .play {
            let baseMove = modelPreferredMove ?? scored.max(by: { $0.0 < $1.0 })?.1 ?? legal[0]
            if players.count == 5,
               modelPreferredMove != nil,
               QuantumCatMLPolicy.shared.usesPlayerCountSpecialist(
                   for: kind,
                   players: players.count
               ) {
                let state = QuantumCatMLLiveState(game: self, player: player)
                return Self.competitiveTacticalMove(
                    state: state,
                    legalMoves: legal,
                    baseMove: baseMove
                ) ?? baseMove
            }
            let shieldedMove = livenessShieldedMove(base: baseMove, legal: legal, player: player)
            if avoidsParadoxAfterMove(shieldedMove) {
                return shieldedMove
            }
            let rankedMoves = livenessRankedMoves(base: baseMove, legal: legal, player: player).prefix(16)
            if let safeMove = rankedMoves.first(where: { avoidsParadoxAfterMove($0) }) {
                return safeMove
            }
            return shieldedMove
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

    private struct CompetitiveTacticalRow {
        let move: Move
        let action: Int
        let wins: Bool
        let feasible: Bool
        let margin: Int
        let buffer: Int
        let tokens: Int
        let cluster: Int
        let board: [[Int]]
        let winner: Int
    }

    /// Reproduces the verified specialist's public-information tactical guard.
    /// Opponents' private hands and face-down discards never enter this API.
    static func competitiveTacticalMove(
        state: QuantumCatMLLiveState,
        legalMoves: [Move],
        baseMove: Move
    ) -> Move? {
        let player = state.actingPlayer
        guard state.phase == 3,
              state.currentPlayer == player,
              state.predictions.indices.contains(player),
              state.tricksWon.indices.contains(player),
              state.colorTokens.indices.contains(player),
              state.currentTrick.count == state.numPlayers,
              state.currentTrick.filter({ $0 != nil }).count == state.numPlayers - 1,
              state.currentTrick[player] == nil,
              legalMoves.contains(baseMove)
        else { return nil }

        let rows = legalMoves.compactMap {
            competitiveTacticalRow(state: state, move: $0)
        }
        guard !rows.isEmpty else { return nil }

        if state.matchRound == state.numPlayers - 1,
           state.trickNumber == state.numTricks - 1,
           state.matchTotals.count == state.numPlayers,
           let selected = competitiveTerminalMove(
               state: state,
               rows: rows,
               baseMove: baseMove
           ) {
            return selected
        }

        let prediction = state.predictions[player]
        guard prediction >= 0 else { return nil }
        let won = state.tricksWon[player]
        let remaining = max(0, state.numTricks - state.trickNumber)
        let isAbove = won > prediction
        let isUnreachable = won + remaining < prediction
        let wantsWin = won != prediction || isUnreachable
        let preferred = rows.filter { $0.wins == wantsWin }
        let opposite = rows.filter { $0.wins != wantsWin }
        guard !preferred.isEmpty,
              let reference = opposite.max(by: { lhs, rhs in
                  lexicographicKey(
                      [lhs.feasible ? 1 : 0, lhs.margin, -lhs.buffer, lhs.tokens],
                      isLessThan: [rhs.feasible ? 1 : 0, rhs.margin, -rhs.buffer, rhs.tokens]
                  )
              })
        else { return nil }

        let safe = preferred.filter {
            $0.feasible
                && $0.margin >= reference.margin
                && $0.buffer <= reference.buffer
                && $0.tokens >= reference.tokens
        }
        guard !safe.isEmpty else { return nil }
        if safe.contains(where: { $0.move == baseMove }) {
            return baseMove
        }
        return safe.max(by: { lhs, rhs in
            lexicographicKey(
                [lhs.margin, -lhs.buffer, lhs.tokens, isAbove ? 0 : lhs.cluster, -lhs.action],
                isLessThan: [rhs.margin, -rhs.buffer, rhs.tokens, isAbove ? 0 : rhs.cluster, -rhs.action]
            )
        })?.move
    }

    private static func competitiveTerminalMove(
        state: QuantumCatMLLiveState,
        rows: [CompetitiveTacticalRow],
        baseMove: Move
    ) -> Move? {
        struct TerminalRow {
            let row: CompetitiveTacticalRow
            let placement: Double
            let ownTotal: Double
            let ownRoundScore: Double
            let ownBonus: Int
        }
        let player = state.actingPlayer
        let terminalRows: [TerminalRow] = rows.map { row in
            var tricks = state.tricksWon
            tricks[row.winner] += 1
            let scores = tricks.indices.map { seat -> Double in
                let bonus = tricks[seat] == state.predictions[seat]
                    ? competitiveLargestCluster(board: row.board, player: seat)
                    : 0
                return Double(tricks[seat] + bonus)
            }
            let totals = scores.indices.map { state.matchTotals[$0] + scores[$0] }
            let placement = scores.indices.filter { $0 != player }.reduce(0.0) { partial, seat in
                if totals[player] > totals[seat] {
                    return partial + 1.0
                }
                if totals[player] == totals[seat] {
                    if scores[player] > scores[seat] {
                        return partial + 1.0
                    }
                    if scores[player] == scores[seat] {
                        return partial + 0.5
                    }
                }
                return partial
            }
            return TerminalRow(
                row: row,
                placement: placement,
                ownTotal: totals[player],
                ownRoundScore: scores[player],
                ownBonus: tricks[player] == state.predictions[player] ? row.cluster : 0
            )
        }
        guard let bestPlacement = terminalRows.map(\.placement).max() else { return nil }
        let best = terminalRows.filter { abs($0.placement - bestPlacement) < 1e-9 }
        if best.contains(where: { $0.row.move == baseMove }) {
            return baseMove
        }
        return best.max(by: { lhs, rhs in
            if lhs.ownTotal != rhs.ownTotal { return lhs.ownTotal < rhs.ownTotal }
            if lhs.ownRoundScore != rhs.ownRoundScore {
                return lhs.ownRoundScore < rhs.ownRoundScore
            }
            if lhs.ownBonus != rhs.ownBonus { return lhs.ownBonus < rhs.ownBonus }
            return lhs.row.action > rhs.row.action
        })?.row.move
    }

    private static func competitiveTacticalRow(
        state: QuantumCatMLLiveState,
        move: Move
    ) -> CompetitiveTacticalRow? {
        guard case .play(let rank, let suit) = move,
              let color = Suit.allCases.firstIndex(of: suit),
              state.actingHand.indices.contains(rank - 1),
              state.board.indices.contains(color),
              state.board[color].indices.contains(rank - 1),
              state.board[color][rank - 1] == -1
        else { return nil }

        let player = state.actingPlayer
        var board = state.board
        board[color][rank - 1] = player
        var counts = state.actingHand
        counts[rank - 1] -= 1
        guard counts[rank - 1] >= 0 else { return nil }
        var tokens = state.colorTokens[player]
        if let ledColor = state.ledColor,
           let ledIndex = Suit.allCases.firstIndex(where: { $0.rawValue == ledColor }),
           ledIndex != color {
            tokens[ledIndex] = false
        }
        let caps = counts.indices.map { rankIndex in
            board.indices.reduce(0) { partial, suitIndex in
                partial + (tokens[suitIndex] && board[suitIndex][rankIndex] == -1 ? 1 : 0)
            }
        }
        let occupiedRanks = counts.indices.filter { counts[$0] > 0 }
        let feasible = occupiedRanks.allSatisfy { counts[$0] <= caps[$0] }
        let buffer = occupiedRanks.reduce(0) {
            $0 + max(0, counts[$1] + 1 - caps[$1])
        }
        let remaining = state.numTricks - state.trickNumber - 1
        let margin = competitiveAdversarialMargin(
            counts: counts,
            caps: caps,
            playsRemaining: remaining
        )
        guard let winner = competitiveTrickWinner(
            state: state,
            player: player,
            card: QuantumCatMLPlayedCard(rank: rank, suit: suit.rawValue)
        ) else { return nil }
        return CompetitiveTacticalRow(
            move: move,
            action: color * state.numRanks + rank - 1,
            wins: winner == player,
            feasible: feasible,
            margin: margin,
            buffer: buffer,
            tokens: tokens.filter { $0 }.count,
            cluster: competitiveLargestCluster(board: board, player: player),
            board: board,
            winner: winner
        )
    }

    private static func competitiveTrickWinner(
        state: QuantumCatMLLiveState,
        player: Int,
        card: QuantumCatMLPlayedCard
    ) -> Int? {
        var trick = state.currentTrick
        trick[player] = card
        guard trick.allSatisfy({ $0 != nil }),
              let ledColor = state.ledColor ?? trick[state.startPlayer]?.suit
        else { return nil }
        let trump = "R"
        let winningColor = trick.contains(where: { $0?.suit == trump }) ? trump : ledColor
        return trick.indices
            .filter { trick[$0]?.suit == winningColor }
            .max { lhs, rhs in (trick[lhs]?.rank ?? -1) < (trick[rhs]?.rank ?? -1) }
    }

    static func competitiveLargestCluster(board: [[Int]], player: Int) -> Int {
        guard let width = board.first?.count, width > 0 else { return 0 }
        var visited = Array(repeating: Array(repeating: false, count: width), count: board.count)
        var best = 0
        for color in board.indices {
            for rank in 0..<width where board[color][rank] == player && !visited[color][rank] {
                var pending = [(color, rank)]
                visited[color][rank] = true
                var size = 0
                while let (currentColor, currentRank) = pending.popLast() {
                    size += 1
                    for (colorDelta, rankDelta) in [(1, 0), (-1, 0), (0, 1), (0, -1)] {
                        let nextColor = currentColor + colorDelta
                        let nextRank = currentRank + rankDelta
                        guard board.indices.contains(nextColor),
                              (0..<width).contains(nextRank),
                              !visited[nextColor][nextRank],
                              board[nextColor][nextRank] == player
                        else { continue }
                        visited[nextColor][nextRank] = true
                        pending.append((nextColor, nextRank))
                    }
                }
                best = max(best, size)
            }
        }
        return best
    }

    static func competitiveAdversarialMargin(
        counts: [Int],
        caps: [Int],
        playsRemaining: Int
    ) -> Int {
        let matching = zip(counts, caps).reduce(0) { $0 + min($1.0, $1.1) }
        guard matching >= playsRemaining else { return -1 }
        let needed = matching - playsRemaining + 1
        guard needed > 0 else { return -1 }
        let infinity = Int.max / 4
        var costs = Array(repeating: infinity, count: needed + 1)
        costs[0] = 0
        for (count, cap) in zip(counts, caps) {
            let units = min(count, cap)
            guard units > 0 else { continue }
            let surplus = max(0, cap - count)
            for target in stride(from: needed, through: 1, by: -1) {
                let source = max(0, target - units)
                if costs[source] != infinity {
                    costs[target] = min(costs[target], costs[source] + surplus)
                }
            }
        }
        return costs[needed] == infinity ? caps.reduce(0, +) : costs[needed] + needed - 1
    }

    private static func lexicographicKey(_ lhs: [Int], isLessThan rhs: [Int]) -> Bool {
        for (left, right) in zip(lhs, rhs) where left != right {
            return left < right
        }
        return false
    }

    private struct OwnHandFeasibility {
        let feasible: Bool
        let remainingCards: Int
        let totalOpenSlots: Int
        let slotSurplus: Int
        let totalDeficit: Int
        let bufferDeficit: Int
        let tightRankCount: Int
        let deadRankCount: Int
        let singletonCardCount: Int
        let minChoices: Int
    }

    private struct PublicExitLiquidity {
        let openCells: Int
        let playerOpenSlots: [Int]
        let playerMinRankSlots: [Int]
        let playerRemainingCards: [Int]
        let playerLaneSurplus: [Int]
        let totalPlayerOpenSlots: Int
        let minPlayerOpenSlots: Int
        let totalPlayerRemainingCards: Int
        let totalPlayerLaneSurplus: Int
        let minPlayerLaneSurplus: Int
        let lanePressurePlayerCount: Int
    }

    private struct LivenessCertificate {
        let action: Move
        let key: [Int]
        let isParadox: Bool
        let ownFeasible: Bool
        let ownTotalDeficit: Int
        let ownBufferDeficit: Int
        let ownDeadRankCount: Int
        let ownSlotSurplus: Int
        let ownMinChoices: Int
        let publicSlotDamage: Int
        let ownPublicSlotDamage: Int
        let boardOpenCellDamage: Int
        let ownPublicSlotsAfter: Int
        let totalPlayerOpenSlotsAfter: Int
        let minPlayerOpenSlotsAfter: Int
        let ownLaneSurplusAfter: Int
        let totalPlayerLaneSurplusAfter: Int
        let minPlayerLaneSurplusAfter: Int
        let lanePressurePlayerCountAfter: Int
        let ownMinRankSlotsAfter: Int
        let minPlayerRankSlotsAfter: Int
        let lostLedToken: Bool
        let wouldWinNow: Bool?
        let overTargetWouldWin: Bool

        func withKey(_ key: [Int]) -> LivenessCertificate {
            LivenessCertificate(
                action: action,
                key: key,
                isParadox: isParadox,
                ownFeasible: ownFeasible,
                ownTotalDeficit: ownTotalDeficit,
                ownBufferDeficit: ownBufferDeficit,
                ownDeadRankCount: ownDeadRankCount,
                ownSlotSurplus: ownSlotSurplus,
                ownMinChoices: ownMinChoices,
                publicSlotDamage: publicSlotDamage,
                ownPublicSlotDamage: ownPublicSlotDamage,
                boardOpenCellDamage: boardOpenCellDamage,
                ownPublicSlotsAfter: ownPublicSlotsAfter,
                totalPlayerOpenSlotsAfter: totalPlayerOpenSlotsAfter,
                minPlayerOpenSlotsAfter: minPlayerOpenSlotsAfter,
                ownLaneSurplusAfter: ownLaneSurplusAfter,
                totalPlayerLaneSurplusAfter: totalPlayerLaneSurplusAfter,
                minPlayerLaneSurplusAfter: minPlayerLaneSurplusAfter,
                lanePressurePlayerCountAfter: lanePressurePlayerCountAfter,
                ownMinRankSlotsAfter: ownMinRankSlotsAfter,
                minPlayerRankSlotsAfter: minPlayerRankSlotsAfter,
                lostLedToken: lostLedToken,
                wouldWinNow: wouldWinNow,
                overTargetWouldWin: overTargetWouldWin
            )
        }
    }

    private func livenessShieldedMove(base: Move, legal: [Move], player: Int) -> Move {
        let baseRow = livenessCertificate(after: base, base: base, player: player)
        guard let selected = livenessRankedCertificates(base: base, legal: legal, player: player).first else { return base }
        guard selected.action != base, selectedDominates(base: baseRow, candidate: selected) else {
            return base
        }
        return selected.action
    }

    private func livenessRankedMoves(base: Move, legal: [Move], player: Int) -> [Move] {
        livenessRankedCertificates(base: base, legal: legal, player: player).map(\.action)
    }

    private func livenessRankedCertificates(base: Move, legal: [Move], player: Int) -> [LivenessCertificate] {
        let candidates = legal.filter { $0 != .paradox }
        let actions = candidates.isEmpty ? legal : candidates
        return actions
            .map { livenessCertificate(after: $0, base: base, player: player) }
            .sorted { lhs, rhs in livenessKey(lhs.key, isBetterThan: rhs.key) }
    }

    private func selectedDominates(base: LivenessCertificate, candidate: LivenessCertificate) -> Bool {
        if candidate.isParadox {
            return false
        }
        if base.isParadox {
            return true
        }
        let publicNotWorse = candidate.minPlayerOpenSlotsAfter >= base.minPlayerOpenSlotsAfter
            && candidate.totalPlayerOpenSlotsAfter >= base.totalPlayerOpenSlotsAfter
            && candidate.ownPublicSlotsAfter >= base.ownPublicSlotsAfter
            && candidate.minPlayerLaneSurplusAfter >= base.minPlayerLaneSurplusAfter
            && candidate.totalPlayerLaneSurplusAfter >= base.totalPlayerLaneSurplusAfter
            && candidate.lanePressurePlayerCountAfter <= base.lanePressurePlayerCountAfter
            && candidate.publicSlotDamage <= base.publicSlotDamage
            && candidate.ownPublicSlotDamage <= base.ownPublicSlotDamage

        if base.lostLedToken, !candidate.lostLedToken {
            if base.ownFeasible, !candidate.ownFeasible {
                return false
            }
            return true
        }
        if candidate.ownTotalDeficit < base.ownTotalDeficit {
            return true
        }
        if candidate.ownTotalDeficit > base.ownTotalDeficit {
            return false
        }
        if candidate.ownFeasible, !base.ownFeasible {
            return true
        }
        if base.lostLedToken, !candidate.lostLedToken {
            return true
        }
        if candidate.minPlayerLaneSurplusAfter > base.minPlayerLaneSurplusAfter {
            return true
        }
        if candidate.lanePressurePlayerCountAfter < base.lanePressurePlayerCountAfter {
            return true
        }
        if candidate.ownDeadRankCount < base.ownDeadRankCount, publicNotWorse {
            return true
        }
        if candidate.ownBufferDeficit < base.ownBufferDeficit, publicNotWorse {
            return true
        }
        if candidate.minPlayerOpenSlotsAfter - base.minPlayerOpenSlotsAfter >= 0 {
            return true
        }
        let damageDelta = base.publicSlotDamage - candidate.publicSlotDamage
        if damageDelta >= 0 {
            return true
        }
        if base.lostLedToken, !candidate.lostLedToken, damageDelta >= 0 {
            return true
        }
        if base.overTargetWouldWin, !candidate.overTargetWouldWin, damageDelta >= 0 {
            return true
        }
        return false
    }

    private func livenessCertificate(after move: Move, base: Move, player: Int) -> LivenessCertificate {
        let before = publicExitLiquidity()
        let beforeOwn = ownHandFeasibility(for: player)

        if move == .paradox {
            let ownDeficit = max(1, beforeOwn.remainingCards)
            let row = LivenessCertificate(
                action: move,
                key: [],
                isParadox: true,
                ownFeasible: false,
                ownTotalDeficit: ownDeficit,
                ownBufferDeficit: ownDeficit,
                ownDeadRankCount: max(1, beforeOwn.deadRankCount),
                ownSlotSurplus: -ownDeficit,
                ownMinChoices: 0,
                publicSlotDamage: before.totalPlayerOpenSlots,
                ownPublicSlotDamage: before.playerOpenSlots[safe: player] ?? 0,
                boardOpenCellDamage: before.openCells,
                ownPublicSlotsAfter: 0,
                totalPlayerOpenSlotsAfter: 0,
                minPlayerOpenSlotsAfter: 0,
                ownLaneSurplusAfter: 0,
                totalPlayerLaneSurplusAfter: -before.totalPlayerRemainingCards,
                minPlayerLaneSurplusAfter: -(before.playerRemainingCards.max() ?? 0),
                lanePressurePlayerCountAfter: before.lanePressurePlayerCount,
                ownMinRankSlotsAfter: 0,
                minPlayerRankSlotsAfter: 0,
                lostLedToken: false,
                wouldWinNow: nil,
                overTargetWouldWin: false
            )
            return row.withKey(livenessKey(for: row, base: base))
        }

        let actionSuit: Suit?
        if case .play(_, let suit) = move {
            actionSuit = suit
        } else {
            actionSuit = nil
        }
        let wouldWinNow: Bool?
        if case .play(let rank, let suit) = move {
            wouldWinNow = wouldWin(player: player, rank: rank, suit: suit)
        } else {
            wouldWinNow = nil
        }

        var copy = self
        copy.apply(move)
        let after = copy.publicExitLiquidity()
        let own = copy.ownHandFeasibility(for: player)
        let isParadox = copy.players[safe: player]?.hasParadoxed ?? false
        let beforePlayerSlots = before.playerOpenSlots[safe: player] ?? 0
        let afterPlayerSlots = after.playerOpenSlots[safe: player] ?? 0
        let afterPlayerMinRankSlots = after.playerMinRankSlots[safe: player] ?? 0
        let afterPlayerLaneSurplus = after.playerLaneSurplus[safe: player] ?? 0

        var lostLedToken = false
        if let ledSuit, let actionSuit, ledSuit != actionSuit,
           let ledIndex = Suit.allCases.firstIndex(of: ledSuit),
           players[safe: player]?.colorTokens[safe: ledIndex] == true,
           copy.players[safe: player]?.colorTokens[safe: ledIndex] == false {
            lostLedToken = true
        }

        let prediction = players[safe: player]?.prediction
        let tricks = players[safe: player]?.tricksWon ?? 0
        let row = LivenessCertificate(
            action: move,
            key: [],
            isParadox: isParadox,
            ownFeasible: own.feasible && !isParadox,
            ownTotalDeficit: isParadox ? max(1, own.totalDeficit) : own.totalDeficit,
            ownBufferDeficit: own.bufferDeficit,
            ownDeadRankCount: own.deadRankCount,
            ownSlotSurplus: own.slotSurplus,
            ownMinChoices: own.minChoices,
            publicSlotDamage: before.totalPlayerOpenSlots - after.totalPlayerOpenSlots,
            ownPublicSlotDamage: beforePlayerSlots - afterPlayerSlots,
            boardOpenCellDamage: before.openCells - after.openCells,
            ownPublicSlotsAfter: afterPlayerSlots,
            totalPlayerOpenSlotsAfter: after.totalPlayerOpenSlots,
            minPlayerOpenSlotsAfter: after.minPlayerOpenSlots,
            ownLaneSurplusAfter: afterPlayerLaneSurplus,
            totalPlayerLaneSurplusAfter: after.totalPlayerLaneSurplus,
            minPlayerLaneSurplusAfter: after.minPlayerLaneSurplus,
            lanePressurePlayerCountAfter: after.lanePressurePlayerCount,
            ownMinRankSlotsAfter: afterPlayerMinRankSlots,
            minPlayerRankSlotsAfter: after.playerMinRankSlots.min() ?? 0,
            lostLedToken: lostLedToken,
            wouldWinNow: wouldWinNow,
            overTargetWouldWin: Self.winningWouldBreakExactPrediction(
                prediction: prediction,
                tricksWon: tricks,
                wouldWin: wouldWinNow == true
            )
        )
        return row.withKey(livenessKey(for: row, base: base))
    }

    private func livenessKey(for row: LivenessCertificate, base: Move) -> [Int] {
        let baseBonus = row.action == base ? 1 : 0
        let responsibleRedWinSwitch = row.lostLedToken && row.wouldWinNow == true && !row.overTargetWouldWin
        let preservesLaneOrSpendsToWin = !row.lostLedToken || responsibleRedWinSwitch
        return [
            row.isParadox ? 0 : 1,
            row.ownFeasible ? 1 : 0,
            preservesLaneOrSpendsToWin ? 1 : 0,
            -row.ownTotalDeficit,
            -row.ownDeadRankCount,
            row.minPlayerLaneSurplusAfter,
            row.totalPlayerLaneSurplusAfter,
            row.ownLaneSurplusAfter,
            -row.lanePressurePlayerCountAfter,
            row.minPlayerOpenSlotsAfter,
            row.totalPlayerOpenSlotsAfter,
            row.ownPublicSlotsAfter,
            row.minPlayerRankSlotsAfter,
            row.ownMinRankSlotsAfter,
            row.overTargetWouldWin ? 0 : 1,
            -row.publicSlotDamage,
            -row.ownPublicSlotDamage,
            -row.boardOpenCellDamage,
            -row.ownBufferDeficit,
            row.ownMinChoices,
            row.ownSlotSurplus,
            baseBonus,
            -row.action.sortValue
        ]
    }

    private func livenessKey(_ lhs: [Int], isBetterThan rhs: [Int]) -> Bool {
        for index in 0..<min(lhs.count, rhs.count) where lhs[index] != rhs[index] {
            return lhs[index] > rhs[index]
        }
        return lhs.count > rhs.count
    }

    private func ownHandFeasibility(for player: Int) -> OwnHandFeasibility {
        guard players.indices.contains(player) else {
            return OwnHandFeasibility(feasible: false, remainingCards: 0, totalOpenSlots: 0, slotSurplus: -1, totalDeficit: 1, bufferDeficit: 1, tightRankCount: 1, deadRankCount: 1, singletonCardCount: 0, minChoices: 0)
        }
        let hand = players[player].hand
        let tokens = players[player].colorTokens
        var remainingCards = 0
        var totalSlots = 0
        var totalDeficit = 0
        var bufferDeficit = 0
        var tightRankCount = 0
        var deadRankCount = 0
        var singletonCardCount = 0
        var minChoices: Int?

        for rankIndex in 0..<rankCount {
            let count = max(0, hand[rankIndex])
            guard count > 0 else { continue }
            remainingCards += count
            var openSlots = 0
            for suitIndex in Suit.allCases.indices where tokens[safe: suitIndex] == true && board[suitIndex][rankIndex] == -1 {
                openSlots += 1
            }
            totalSlots += openSlots
            totalDeficit += max(0, count - openSlots)
            bufferDeficit += max(0, count + 1 - openSlots)
            if openSlots <= count {
                tightRankCount += 1
            }
            if openSlots <= 0 {
                deadRankCount += 1
            }
            if openSlots == 1 {
                singletonCardCount += count
            }
            minChoices = min(minChoices ?? openSlots, openSlots)
        }

        return OwnHandFeasibility(
            feasible: totalDeficit == 0,
            remainingCards: remainingCards,
            totalOpenSlots: totalSlots,
            slotSurplus: totalSlots - remainingCards,
            totalDeficit: totalDeficit,
            bufferDeficit: bufferDeficit,
            tightRankCount: tightRankCount,
            deadRankCount: deadRankCount,
            singletonCardCount: singletonCardCount,
            minChoices: minChoices ?? 0
        )
    }

    private func publicExitLiquidity() -> PublicExitLiquidity {
        let openBySuit = board.map { row in row.filter { $0 == -1 }.count }
        var playerOpenSlots: [Int] = []
        var playerMinRankSlots: [Int] = []
        var playerRemainingCards: [Int] = []
        var playerLaneSurplus: [Int] = []

        for seat in players.indices {
            let tokens = players[seat].colorTokens
            let openSlots = Suit.allCases.indices.reduce(0) { total, suitIndex in
                total + (tokens[safe: suitIndex] == true ? openBySuit[suitIndex] : 0)
            }
            playerOpenSlots.append(openSlots)

            var rankSlots: [Int] = []
            for rankIndex in 0..<rankCount {
                var slots = 0
                for suitIndex in Suit.allCases.indices where tokens[safe: suitIndex] == true && board[suitIndex][rankIndex] == -1 {
                    slots += 1
                }
                rankSlots.append(slots)
            }
            playerMinRankSlots.append(rankSlots.min() ?? 0)
            let remaining = players[seat].hand.reduce(0, +)
            playerRemainingCards.append(remaining)
            playerLaneSurplus.append(openSlots - remaining)
        }

        let totalOpen = playerOpenSlots.reduce(0, +)
        let totalRemaining = playerRemainingCards.reduce(0, +)
        let totalLaneSurplus = playerLaneSurplus.reduce(0, +)
        return PublicExitLiquidity(
            openCells: openBySuit.reduce(0, +),
            playerOpenSlots: playerOpenSlots,
            playerMinRankSlots: playerMinRankSlots,
            playerRemainingCards: playerRemainingCards,
            playerLaneSurplus: playerLaneSurplus,
            totalPlayerOpenSlots: totalOpen,
            minPlayerOpenSlots: playerOpenSlots.min() ?? 0,
            totalPlayerRemainingCards: totalRemaining,
            totalPlayerLaneSurplus: totalLaneSurplus,
            minPlayerLaneSurplus: playerLaneSurplus.min() ?? 0,
            lanePressurePlayerCount: playerLaneSurplus.filter { $0 < 0 }.count
        )
    }

    private func sharedDiscardMove(for player: Int, legal: [Move]) -> Move? {
        guard phase == .discard, players.indices.contains(player) else { return nil }
        let hand = players[player].hand
        let discardRanks = legal.compactMap { move -> Int? in
            guard case .discard(let rank) = move,
                  (1...hand.count).contains(rank),
                  hand[rank - 1] > 0 else {
                return nil
            }
            return rank
        }
        guard let maxCount = discardRanks.map({ hand[$0 - 1] }).max() else { return nil }
        var candidates = discardRanks.filter { hand[$0 - 1] == maxCount }
        if candidates.count > 1, candidates.contains(6) {
            candidates.removeAll { $0 == 6 }
        }
        guard let selected = candidates.max(by: { lhs, rhs in
            let lhsScore = discardDissimilarityScore(hand: hand, rank: lhs)
            let rhsScore = discardDissimilarityScore(hand: hand, rank: rhs)
            if lhsScore != rhsScore {
                return lhsScore < rhsScore
            }
            return lhs > rhs
        }) else {
            return nil
        }
        return .discard(rank: selected)
    }

    private func discardDissimilarityScore(hand: [Int], rank: Int) -> Double {
        guard (1...hand.count).contains(rank) else { return 0.0 }
        var remaining = hand
        remaining[rank - 1] = max(0, remaining[rank - 1] - 1)
        let total = remaining.reduce(0, +)
        guard total > 0 else { return 0.0 }
        let weightedDistance = remaining.enumerated().reduce(0.0) { partial, item in
            let otherRank = item.offset + 1
            return partial + Double(abs(otherRank - rank) * item.element)
        }
        return weightedDistance / Double(total)
    }

    private func predictionGap(_ move: Move, target: Int) -> Int {
        guard case .prediction(let value) = move else { return Int.max }
        return abs(value - target)
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
            let wantsMore = Self.wantsAdditionalTrick(
                prediction: players[player].prediction,
                tricksWon: players[player].tricksWon
            )
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
        Self.sharedPredictionBid(state: QuantumCatMLLiveState(game: self, player: player))
    }

    static func sharedPredictionBid(state: QuantumCatMLLiveState) -> Int {
        sharedPredictionBid(
            hand: state.actingHand,
            rankCount: state.numRanks,
            isRoundLeader: state.actingPlayer == state.roundStartPlayer
        )
    }

    static func sharedPredictionBid(
        hand: [Int],
        rankCount: Int,
        isRoundLeader: Bool
    ) -> Int {
        guard !hand.isEmpty, rankCount > 0 else { return 1 }
        let topRankIndex = min(rankCount, hand.count) - 1
        let secondRankIndex = topRankIndex - 1
        var expectedTricks = Double(hand[topRankIndex])
        if secondRankIndex >= 0 {
            expectedTricks += (isRoundLeader ? 0.5 : 0.25)
                * Double(hand[secondRankIndex])
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

private extension Array {
    subscript(safe index: Int) -> Element? {
        indices.contains(index) ? self[index] : nil
    }
}

// MARK: - Parity and test support (qcx champion)

extension QuantumCatGame {
    /// A fresh 3-5 player round with explicitly dealt hands (per-rank counts),
    /// used to replay Python-generated parity fixtures in the Swift engine.
    static func withDealtHands(seats: [SeatKind], hands: [[Int]], startingPlayer: Int) -> QuantumCatGame? {
        var game = QuantumCatGame(seats: seats, seed: 1, startingPlayer: startingPlayer, autoAdvanceBots: false)
        guard (3...5).contains(game.players.count),
              hands.count == game.players.count,
              hands.allSatisfy({ $0.count == game.rankCount }),
              hands.allSatisfy({ $0.reduce(0, +) == cardsPerPlayer(for: game.players.count) }) else { return nil }
        for index in game.players.indices {
            game.players[index].hand = hands[index]
        }
        return game
    }

    /// Replace one seat's hidden cards (hand counts and face-down discard).
    /// Only the information-safety tests use this, to prove a decision does
    /// not depend on opponents' hidden cards.
    mutating func overrideHiddenCards(for player: Int, hand: [Int], discardedRank: Int?) {
        guard players.indices.contains(player), hand.count == rankCount else { return }
        players[player].hand = hand
        players[player].discardedRank = discardedRank
    }
}
