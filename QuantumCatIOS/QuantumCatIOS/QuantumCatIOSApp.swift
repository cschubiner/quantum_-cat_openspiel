import SwiftUI
import Darwin

@main
struct QuantumCatIOSApp: App {
    @Environment(\.scenePhase) private var scenePhase
    @StateObject private var store = GameStore()

    init() {
        Self.runDeviceParadoxBenchmarkIfRequested()
    }

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environmentObject(store)
        }
        .onChange(of: scenePhase) { _, phase in
            if phase != .active {
                store.save()
            }
        }
    }

    private static func runDeviceParadoxBenchmarkIfRequested() {
        let arguments = ProcessInfo.processInfo.arguments
        let environment = ProcessInfo.processInfo.environment
        let shouldRunBenchmark = arguments.contains("-deviceParadoxBenchmark")
            || environment["DEVICE_PARADOX_BENCHMARK"] == "1"
        guard shouldRunBenchmark else { return }

        let games = integerArgument(named: "-deviceParadoxBenchmarkGames", in: arguments)
            ?? Int(environment["DEVICE_PARADOX_BENCHMARK_GAMES"] ?? "")
            ?? 20
        let seatCount = integerArgument(named: "-deviceParadoxBenchmarkSeats", in: arguments)
            ?? Int(environment["DEVICE_PARADOX_BENCHMARK_SEATS"] ?? "")
            ?? 5
        let seats = Array(repeating: SeatKind.bot(.championBeliefPolicy), count: seatCount)
        var paradoxEvents = 0
        var gamesWithParadox = 0
        let phaseChecks = benchmarkSharedPhaseChecks()

        QuantumCatMLPolicy.shared.resetUsage()

        for index in 0..<games {
            let game = QuantumCatGame(seats: seats, seed: 20263740 + index, autoAdvanceBots: true)
            let playerParadoxes = game.players.map(\.hasParadoxed)
            paradoxEvents += playerParadoxes.filter { $0 }.count
            if playerParadoxes.contains(true) {
                gamesWithParadox += 1
            }
        }

        let seatParadoxRate = Double(paradoxEvents) / Double(max(1, games * seatCount))
        let gameParadoxRate = Double(gamesWithParadox) / Double(max(1, games))
        let usage = QuantumCatMLPolicy.shared.usageSnapshot()
        let coreMLAttempts = usage.reduce(0) { $0 + $1.attempts }
        let coreMLSuccesses = usage.reduce(0) { $0 + $1.successes }
        let coreMLFailures = usage.reduce(0) { $0 + $1.failures }

        let result = """
        {
          "kind": "championBeliefPolicy",
          "players": \(seatCount),
          "games": \(games),
          "seat_rate": \(String(format: "%.4f", seatParadoxRate)),
          "game_rate": \(String(format: "%.4f", gameParadoxRate)),
          "games_with_paradox": \(gamesWithParadox),
          "paradox_events": \(paradoxEvents),
          "shared_bid_check": \(phaseChecks.bid ? "true" : "false"),
          "shared_discard_check": \(phaseChecks.discard ? "true" : "false"),
          "coreml_successes": \(coreMLSuccesses),
          "coreml_attempts": \(coreMLAttempts),
          "coreml_failures": \(coreMLFailures)
        }
        """
        if let documents = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first {
            try? result.write(
                to: documents.appendingPathComponent("device_paradox_benchmark.json"),
                atomically: true,
                encoding: .utf8
            )
        }
        print(
            "DEVICE_PARADOX_BENCHMARK kind=championBeliefPolicy " +
            "players=\(seatCount) games=\(games) " +
            "seat_rate=\(String(format: "%.4f", seatParadoxRate)) " +
            "game_rate=\(String(format: "%.4f", gameParadoxRate)) " +
            "games_with_paradox=\(gamesWithParadox) " +
            "paradox_events=\(paradoxEvents) " +
            "shared_bid_check=\(phaseChecks.bid) " +
            "shared_discard_check=\(phaseChecks.discard) " +
            "coreml_successes=\(coreMLSuccesses) " +
            "coreml_attempts=\(coreMLAttempts) " +
            "coreml_failures=\(coreMLFailures)"
        )
        fflush(stdout)
        sleep(1)
        exit(coreMLFailures == 0 && phaseChecks.bid && phaseChecks.discard ? 0 : 2)
    }

    private static func benchmarkSharedPhaseChecks() -> (bid: Bool, discard: Bool) {
        var discardOK = false
        for seed in 20260620..<(20260620 + 2_000) {
            var discardGame = QuantumCatGame(
                seats: [.bot(.championBeliefPolicy), .bot(.random), .bot(.random)],
                seed: seed,
                autoAdvanceBots: false
            )
            let hand = discardGame.players[0].hand
            let maxCount = hand.max() ?? 0
            let tiedRanks = hand.enumerated()
                .filter { $0.element == maxCount }
                .map { $0.offset + 1 }
            guard tiedRanks.count > 1, tiedRanks.contains(6) else { continue }
            let discardMove = discardGame.applyBotMoveForCurrentPlayer()
            if case .discard(let rank) = discardMove {
                discardOK = rank != 6
            }
            break
        }

        var bidGame = QuantumCatGame(
            seats: [.bot(.championBeliefPolicy), .bot(.random), .bot(.random)],
            seed: 20260620,
            autoAdvanceBots: false
        )
        var guardCount = 0
        while bidGame.phase != .prediction && !bidGame.isTerminal && guardCount < 20 {
            guardCount += 1
            _ = bidGame.applyBotMoveForCurrentPlayer()
        }
        let bidMove = bidGame.applyBotMoveForCurrentPlayer()
        let bidOK: Bool
        if case .prediction(let value) = bidMove {
            bidOK = (1...4).contains(value)
        } else {
            bidOK = false
        }
        return (bidOK, discardOK)
    }

    private static func integerArgument(named name: String, in arguments: [String]) -> Int? {
        guard let index = arguments.firstIndex(of: name),
              arguments.indices.contains(index + 1),
              let value = Int(arguments[index + 1]) else {
            return nil
        }
        return value
    }
}
