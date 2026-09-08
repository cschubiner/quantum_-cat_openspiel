import SwiftUI
import Darwin

@main
struct QuantumCatIOSApp: App {
    @Environment(\.scenePhase) private var scenePhase
    @StateObject private var store = GameStore()

    init() {
        Self.runDeviceChampionSearchParityIfRequested()
        Self.runDeviceModelMoveParityIfRequested()
        ChampionInference.queue.async {
            Self.runDeviceParadoxBenchmarkIfRequested()
        }
    }

    private static func runDeviceChampionSearchParityIfRequested() {
        guard ProcessInfo.processInfo.arguments.contains("-deviceChampionSearchParity") else { return }
        // Return from app initialization promptly to avoid the launch watchdog.
        ChampionInference.queue.async {
            UserDefaults.standard.set(false, forKey: ChampionInference.receiptKey)
            do {
                guard let fixture = Bundle.main.url(forResource: "champion_1052_search_parity", withExtension: "json"),
                      let model = Bundle.main.url(forResource: "champion_1052_search", withExtension: "mlmodelc") else {
                    throw ChampionSearchError.missingModel
                }
                let passed = try ChampionSearchParity.run(fixtureURL: fixture, modelURL: model)
                UserDefaults.standard.set(passed, forKey: ChampionInference.receiptKey)
                UserDefaults.standard.synchronize()
                print("DEVICE_CHAMPION_1052_SEARCH_PARITY passed=\(passed) enabled=\(ChampionInference.verified)")
                fflush(stdout)
                exit(passed ? 0 : 3)
            } catch {
                print("DEVICE_CHAMPION_1052_SEARCH_PARITY passed=false error=\(error)")
                fflush(stdout)
                exit(3)
            }
        }
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

    private static func runDeviceModelMoveParityIfRequested() {
        let arguments = ProcessInfo.processInfo.arguments
        let environment = ProcessInfo.processInfo.environment
        let shouldRunParity = arguments.contains("-deviceModelMoveParity")
            || environment["DEVICE_MODEL_MOVE_PARITY"] == "1"
        guard shouldRunParity else { return }

        let players = integerArgument(named: "-deviceModelMoveParityPlayers", in: arguments)
            ?? Int(environment["DEVICE_MODEL_MOVE_PARITY_PLAYERS"] ?? "")
            ?? 3
        let fixtureResource: String
        switch players {
        case 5:
            fixtureResource = "champion_5p_move_parity"
        case 4:
            fixtureResource = "champion_4p_move_parity"
        default:
            fixtureResource = "champion_move_parity"
        }
        let result = QuantumCatMLPolicy.shared.runMoveParityFixtures(
            resource: fixtureResource,
            players: players
        )
        let learnedPredictionEnabled = QuantumCatMLPolicy.shared.usesLearnedPrediction(
            for: .championBeliefPolicy,
            players: players
        )
        let checkpointSHA256 = QuantumCatMLPolicy.shared.provenance(
            for: .championBeliefPolicy,
            players: players
        ).checkpointSHA256 ?? "missing"
        let parityPassed = result.passed && (players != 5 || learnedPredictionEnabled)
        let error = result.error ?? "none"
        print(
            "DEVICE_MODEL_MOVE_PARITY " +
            "players=\(players) " +
            "fixtures=\(result.fixtureCount) " +
            "mismatches=\(result.mismatchCount) " +
            "max_abs_diff=\(String(format: "%.8f", result.maxAbsDiff)) " +
            "live_input_mismatches=\(result.liveInputMismatchCount) " +
            "max_live_input_abs_diff=\(String(format: "%.8f", result.maxLiveInputAbsDiff)) " +
            "checkpoint_sha256=\(checkpointSHA256) " +
            "learned_prediction_enabled=\(learnedPredictionEnabled) " +
            "passed=\(parityPassed) " +
            "error=\(error)"
        )
        fflush(stdout)
        sleep(1)
        exit(parityPassed ? 0 : 3)
    }

    private static func runDeviceParadoxBenchmarkIfRequested() {
        let arguments = ProcessInfo.processInfo.arguments
        let environment = ProcessInfo.processInfo.environment
        let shouldRunBenchmark = arguments.contains("-deviceParadoxBenchmark")
            || environment["DEVICE_PARADOX_BENCHMARK"] == "1"
        guard shouldRunBenchmark else { return }

        let games = integerArgument(named: "-deviceParadoxBenchmarkGames", in: arguments)
            ?? Int(environment["DEVICE_PARADOX_BENCHMARK_GAMES"] ?? "")
            ?? 100
        let seatCount = integerArgument(named: "-deviceParadoxBenchmarkSeats", in: arguments)
            ?? Int(environment["DEVICE_PARADOX_BENCHMARK_SEATS"] ?? "")
            ?? 5
        let seats = Array(repeating: SeatKind.bot(.championBeliefPolicy), count: seatCount)
        var paradoxEvents = 0
        var gamesWithParadox = 0
        var bidCounts = Array(repeating: 0, count: 5)
        let phaseChecks = benchmarkSharedPhaseChecks(seatCount: seatCount)

        QuantumCatMLPolicy.shared.resetUsage()

        for index in 0..<games {
            let game = QuantumCatGame(seats: seats, seed: 20263740 + index, autoAdvanceBots: true)
            let playerParadoxes = game.players.map(\.hasParadoxed)
            paradoxEvents += playerParadoxes.filter { $0 }.count
            if playerParadoxes.contains(true) {
                gamesWithParadox += 1
            }
            for player in game.players {
                if let prediction = player.prediction,
                   bidCounts.indices.contains(prediction) {
                    bidCounts[prediction] += 1
                }
            }
        }

        let seatParadoxRate = Double(paradoxEvents) / Double(max(1, games * seatCount))
        let gameParadoxRate = Double(gamesWithParadox) / Double(max(1, games))
        let usage = QuantumCatMLPolicy.shared.usageSnapshot()
        let coreMLAttempts = usage.reduce(0) { $0 + $1.attempts }
        let coreMLSuccesses = usage.reduce(0) { $0 + $1.successes }
        let coreMLFailures = usage.reduce(0) { $0 + $1.failures }
        let provenance = QuantumCatMLPolicy.shared.provenance(
            for: .championBeliefPolicy,
            players: seatCount
        )
        let liveInputContract = provenance.liveInputContract ?? "missing"
        let checkpointSHA256 = provenance.checkpointSHA256 ?? "missing"
        let liveInferenceVerified = QuantumCatMLPolicy.shared.liveInferenceIsVerified(
            for: .championBeliefPolicy,
            players: seatCount
        )
        let learnedPredictionEnabled = QuantumCatMLPolicy.shared.usesLearnedPrediction(
            for: .championBeliefPolicy,
            players: seatCount
        )
        let searchUsage = ChampionInference.usage

        let result = """
        {
          "kind": "championBeliefPolicy",
          "players": \(seatCount),
          "games": \(games),
          "seat_rate": \(String(format: "%.4f", seatParadoxRate)),
          "game_rate": \(String(format: "%.4f", gameParadoxRate)),
          "games_with_paradox": \(gamesWithParadox),
          "paradox_events": \(paradoxEvents),
          "bid_histogram": {"1": \(bidCounts[1]), "2": \(bidCounts[2]), "3": \(bidCounts[3]), "4": \(bidCounts[4])},
          "shared_bid_check": \(phaseChecks.bid ? "true" : "false"),
          "shared_discard_check": \(phaseChecks.discard ? "true" : "false"),
          "live_input_contract": "\(liveInputContract)",
          "checkpoint_sha256": "\(checkpointSHA256)",
          "live_inference_verified": \(liveInferenceVerified ? "true" : "false"),
          "learned_prediction_enabled": \(learnedPredictionEnabled ? "true" : "false"),
          "coreml_successes": \(coreMLSuccesses),
          "coreml_attempts": \(coreMLAttempts),
          "coreml_failures": \(coreMLFailures)
          ,"search_enabled": \(ChampionInference.verified)
          ,"search_expanded_decisions": \(searchUsage.expanded)
          ,"search_failures": \(searchUsage.failures)
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
            "bid_histogram=1:\(bidCounts[1]),2:\(bidCounts[2]),3:\(bidCounts[3]),4:\(bidCounts[4]) " +
            "shared_bid_check=\(phaseChecks.bid) " +
            "shared_discard_check=\(phaseChecks.discard) " +
            "live_input_contract=\(liveInputContract) " +
            "checkpoint_sha256=\(checkpointSHA256) " +
            "live_inference_verified=\(liveInferenceVerified) " +
            "learned_prediction_enabled=\(learnedPredictionEnabled) " +
            "coreml_successes=\(coreMLSuccesses) " +
            "coreml_attempts=\(coreMLAttempts) " +
            "coreml_failures=\(coreMLFailures) " +
            "search_enabled=\(ChampionInference.verified) " +
            "search_expanded_decisions=\(searchUsage.expanded) " +
            "search_failures=\(searchUsage.failures)"
        )
        fflush(stdout)
        sleep(1)
        let inferencePassed = coreMLAttempts > 0
            && coreMLSuccesses == coreMLAttempts
            && coreMLFailures == 0
            && liveInferenceVerified
            && (seatCount != 5 || learnedPredictionEnabled)
            && searchUsage.failures == 0
        exit(inferencePassed && phaseChecks.bid && phaseChecks.discard ? 0 : 2)
    }

    private static func benchmarkSharedPhaseChecks(seatCount: Int) -> (bid: Bool, discard: Bool) {
        let seats: [SeatKind] = [.bot(.championBeliefPolicy)]
            + Array(repeating: SeatKind.bot(.random), count: max(0, seatCount - 1))
        // The shared Python setup rule protects rank six at every table size;
        // it does not switch to the highest rank in four- or five-player games.
        let protectedDiscardRank = 6
        var discardOK = false
        for seed in 20260620..<(20260620 + 2_000) {
            var discardGame = QuantumCatGame(
                seats: seats,
                seed: seed,
                autoAdvanceBots: false
            )
            let hand = discardGame.players[0].hand
            let maxCount = hand.max() ?? 0
            let tiedRanks = hand.enumerated()
                .filter { $0.element == maxCount }
                .map { $0.offset + 1 }
            guard tiedRanks.count > 1, tiedRanks.contains(protectedDiscardRank) else { continue }
            let discardMove = discardGame.applyBotMoveForCurrentPlayer()
            if case .discard(let rank) = discardMove {
                discardOK = rank != protectedDiscardRank
            }
            break
        }

        var bidGame = QuantumCatGame(
            seats: seats,
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
