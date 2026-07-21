import Foundation

@main
struct EngineSmoke {
    static func main() {
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
            var sawBoardPlay = false
            var sawP1Human = game.currentPlayer == 1 && game.activeHuman != nil
            while !game.isTerminal {
                steps += 1
                guard steps < 260 else {
                    fail("\(config.0): exceeded step guard")
                }
                guard let move = game.legalMoves.first else {
                    fail("\(config.0): no legal move in phase \(game.phase.rawValue)")
                }
                if case .play = move {
                    sawBoardPlay = true
                }
                game.applyHumanMove(move)
                if game.currentPlayer == 1 && game.activeHuman != nil {
                    sawP1Human = true
                }
            }
            guard game.phase == .scoring else {
                fail("\(config.0): did not enter scoring")
            }
            guard game.players.allSatisfy({ $0.score != nil }) else {
                fail("\(config.0): missing final scores")
            }
            guard sawBoardPlay else {
                fail("\(config.0): never exercised play phase")
            }
            if config.1.prefix(2).allSatisfy({ $0.isHuman }) {
                guard sawP1Human else {
                    fail("\(config.0): never handed off to P1 human")
                }
            }
            let encoded: Data
            do {
                encoded = try JSONEncoder().encode(game)
                let restored = try JSONDecoder().decode(QuantumCatGame.self, from: encoded)
                guard restored.phase == game.phase,
                      restored.players.count == game.players.count,
                      restored.board == game.board,
                      restored.log.count == game.log.count else {
                    fail("\(config.0): restored game state did not match encoded state")
                }
            } catch {
                fail("\(config.0): persistence encode/decode failed: \(error)")
            }
            print("PASS \(config.0): \(steps) human actions, scores=\(game.players.map { $0.score ?? 0 })")
        }
    }

    private static func fail(_ message: String) -> Never {
        fputs("FAIL \(message)\n", stderr)
        exit(1)
    }
}
