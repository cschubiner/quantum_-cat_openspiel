import Foundation

@main struct SearchParityCLI {
    static func main() throws {
        let passed = try ChampionSearchParity.run(
            fixtureURL: URL(fileURLWithPath: CommandLine.arguments[1]),
            modelURL: URL(fileURLWithPath: CommandLine.arguments[2])
        )
        exit(passed ? 0 : 1)
    }
}
