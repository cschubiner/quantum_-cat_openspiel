import Foundation

struct SearchParityPayload: Decodable {
    struct Continuation: Decodable {
        let state: QuantumCatMLLiveState
        let legal: [Int]
        let action: Int
        let observation: [Double]
    }
    struct World: Decodable {
        let state: QuantumCatMLLiveState
        let seed: UInt32
        let legal: [Int]
        let hands: [[Int]]
        let discards: [Int]
        let candidates: [Int]
        let safe: [Int]
        let baseline: Int
    }
    struct Deal: Decodable {
        let seed: UInt32
        let round: Int
        let hands: [[Int]]
        let state: QuantumCatMLLiveState
        let observation: [Double]
    }
    struct Search: Decodable {
        let state: QuantumCatMLLiveState
        let legal: [Int]
        let action: Int
        let baseline: Int
        let seed: UInt32
        let candidates: [Int]
        let searchScores: [String: [Double]]
        let auditScores: [String: [Double]]
    }
    let randomSeed: UInt32
    let randomWords: [UInt32]
    let deals: [Deal]
    let worlds: [World]
    let continuations: [Continuation]
    let searches: [Search]
}

enum ChampionSearchParity {
    static func run(fixtureURL: URL, modelURL: URL) throws -> Bool {
        let payload = try JSONDecoder().decode(SearchParityPayload.self, from: Data(contentsOf: fixtureURL))
        let search = QuantumCatChampionSearch(network: try ChampionSearchNetwork(url: modelURL))
        var failures = [String]()
        func check(_ ok: Bool, _ message: String) { if !ok { failures.append(message); print("FAIL \(message)") } }
        check(payload.continuations.count >= 100 && payload.worlds.count >= 20 && payload.searches.count >= 5, "nonempty coverage")
        check(Set(payload.searches.map { $0.state.matchRound }) == Set(0..<5), "all five search rounds")
        var rng = ChampionRandom(seed: payload.randomSeed)
        check(payload.randomWords.allSatisfy { $0 == rng.next() }, "MT19937 words")
        for (i, row) in payload.deals.enumerated() {
            var rng = ChampionRandom(seed: row.seed)
            let world = ChampionWorld(round: row.round, start: row.round, totals: [1, 2, 3, 4, 5], rng: &rng)
            check(world.hands == row.hands, "deal \(i) hands")
            check(world.view == row.state, "deal \(i) state")
        }
        for (i, row) in payload.worlds.enumerated() {
            var rng = ChampionRandom(seed: row.seed)
            let world = try ChampionWorld(view: row.state, rng: &rng)
            check(try QuantumCatChampionSearch.seed(row.state, legal: row.legal) == row.seed, "world \(i) seed")
            check(world.hands == row.hands && world.discards == row.discards, "world \(i) sampler")
            check(QuantumCatChampionSearch.safeCandidates(row.state, candidates: row.candidates, baseline: row.baseline) == row.safe, "world \(i) safety")
        }
        for (i, row) in payload.continuations.enumerated() {
            check(ChampionWorld.legal(row.state) == row.legal, "continuation \(i) legal")
            let obs = try QuantumCatMLFeatureEncoder.observation(state: row.state)
            check(zip(obs, row.observation).allSatisfy { Float($0) == Float($1) }, "continuation \(i) observation")
            let action = try search.continuation(row.state)
            check(action == row.action, "continuation \(i) expected=\(row.action) actual=\(action)")
        }
        print("SEARCH_COMPONENT_PARITY failures=\(failures.count)")
        fflush(stdout)
        guard failures.isEmpty else { return false }
        for (i, row) in payload.searches.enumerated() {
            let start = Date()
            let actual = try search.decide(row.state)
            check(actual.action == row.action, "root \(i) expected=\(row.action) actual=\(actual.action)")
            check(actual.baseline == row.baseline && actual.candidates == row.candidates && actual.seed == row.seed, "root \(i) inputs")
            if row.candidates.count > 1 {
                check(actual.searchScores == row.searchScores, "root \(i) search scores actual=\(actual.searchScores) expected=\(row.searchScores)")
                check(actual.auditScores == row.auditScores, "root \(i) audit scores actual=\(actual.auditScores) expected=\(row.auditScores)")
            }
            print("ROOT \(i) action=\(actual.action) calls=\(actual.neuralCalls) seconds=\(Date().timeIntervalSince(start))")
            fflush(stdout)
        }
        print("SEARCH_PARITY continuations=\(payload.continuations.count) worlds=\(payload.worlds.count) roots=\(payload.searches.count) failures=\(failures.count)")
        return failures.isEmpty
    }
}
