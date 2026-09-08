import Foundation

struct QuantumCatMLPlayedCard: Codable, Equatable {
    let rank: Int
    let suit: String
}

/// The exact information available to the acting player at inference time.
/// Opponent hands and face-down discards are intentionally absent.
struct QuantumCatMLLiveState: Codable, Equatable {
    let numPlayers: Int
    let numRanks: Int
    let numColors: Int
    let numTricks: Int
    let cardsPerPlayerInitial: Int
    let phase: Int
    let currentPlayer: Int
    let actingPlayer: Int
    let trickNumber: Int
    let startPlayer: Int
    let roundStartPlayer: Int
    let ledColor: String?
    let currentTrick: [QuantumCatMLPlayedCard?]
    let playedCounts: [Int]
    let predictions: [Int]
    let tricksWon: [Int]
    let board: [[Int]]
    let colorTokens: [[Bool]]
    let actingHand: [Int]
    let actingDiscardedRank: Int
    let trumpBroken: Bool
    let matchTotals: [Double]
    let matchRound: Int
    let isTerminal: Bool

    init(game: QuantumCatGame, player: Int) {
        numPlayers = game.players.count
        numRanks = game.rankCount
        numColors = Suit.allCases.count
        numTricks = game.tricksInRound
        cardsPerPlayerInitial = QuantumCatGame.cardsPerPlayer(for: game.players.count)
        phase = Self.phaseIndex(game.phase)
        currentPlayer = game.currentPlayer
        actingPlayer = player
        trickNumber = game.trickNumber
        startPlayer = game.mlStartPlayer
        roundStartPlayer = game.mlRoundStartPlayer
        ledColor = game.ledSuit?.rawValue
        currentTrick = game.currentTrick.map { card in
            card.map { QuantumCatMLPlayedCard(rank: $0.rank, suit: $0.suit.rawValue) }
        }
        playedCounts = game.mlPlayedCounts
        predictions = game.players.map { $0.prediction ?? -1 }
        tricksWon = game.players.map(\.tricksWon)
        board = game.board
        colorTokens = game.players.map(\.colorTokens)
        actingHand = game.players.indices.contains(player)
            ? game.players[player].hand
            : Array(repeating: 0, count: game.rankCount)
        actingDiscardedRank = game.players.indices.contains(player)
            ? game.players[player].discardedRank ?? -1
            : -1
        trumpBroken = game.mlTrumpBroken
        matchTotals = game.mlMatchTotals
        matchRound = game.mlMatchRound
        isTerminal = game.isTerminal
    }

    private static func phaseIndex(_ phase: GamePhase) -> Int {
        switch phase {
        case .dealing: 0
        case .discard: 1
        case .prediction: 2
        case .play: 3
        case .scoring: 4
        }
    }
}

private extension Array {
    subscript(safe index: Int) -> Element? {
        indices.contains(index) ? self[index] : nil
    }
}

enum QuantumCatMLFeatureEncoderError: Error, Equatable {
    case unsupportedContract(String)
    case invalidState(String)
    case featureSize(expected: Int, actual: Int)
}

enum QuantumCatMLFeatureEncoder {
    static let observationSize = 76
    static let unifiedObservationSize = 113
    static let actionFeatureSize = 214
    static let contract = "python-observation76-action214-v1"
    static let unifiedContract = "python-observation113-action214-padded345-v2"

    private static let maxRankFeatures = 6
    private static let maxColorFeatures = 4
    private static let numDistinctActions = 1000

    static func observationSize(forPlayers players: Int) -> Int? {
        switch players {
        case 3: 76
        case 4: 97
        case 5: 113
        default: nil
        }
    }

    static func supports(game: QuantumCatGame) -> Bool {
        guard observationSize(forPlayers: game.players.count) != nil else {
            return false
        }
        return game.rankCount == QuantumCatGame.rankCount(for: game.players.count)
            && Suit.allCases.count == 4
            && game.tricksInRound == QuantumCatGame.tricksInRound(for: game.players.count)
    }

    private struct Flexibility {
        let legalCount: Double
        let legalRankCount: Double
        let legalColorCount: Double
        let deadCardCount: Double
        let singletonCardCount: Double
        let totalCards: Double
    }

    private struct Pressure {
        let totalSlots: Double
        let rankDeficit: Double
        let tightRankCount: Double
        let minRankSurplus: Double
        let maxRankDeficit: Double
        let bufferDeficit: Double
        let noExit: Bool
        let deadRankCount: Double
    }

    private struct ExitLiquidity {
        let openCells: Int
        let playerOpenSlots: [Int]
        let playerRemainingCards: [Int]
        let playerLaneSurplus: [Int]
        let totalPlayerOpenSlots: Int
        let minPlayerOpenSlots: Int
        let totalPlayerLaneSurplus: Int
        let minPlayerLaneSurplus: Int
        let lanePressurePlayerCount: Int
    }

    private struct FeasibilityProfile {
        let minColors: Double
        let meanColors: Double
        let zeroExitFraction: Double
        let oneExitFraction: Double
        let twoOrLessExitFraction: Double
        let sumLogColors: Double
        let legalCountNormalized: Double
    }

    private struct ClusterProfile {
        let components: Double
        let frontier: Double
    }

    private enum ActionType: String {
        case discard
        case prediction
        case play
        case paradox
        case other
    }

    /// Absolute source/z/percentile columns from the Python 214-feature contract.
    private static let legalContextColumns: [(source: Int, z: Int, percentile: Int)] = [
        (47, 100, 101), (49, 102, 103), (74, 104, 105),
        (76, 106, 107), (77, 108, 109), (78, 110, 111),
        (86, 112, 113), (87, 114, 115), (88, 116, 117),
        (90, 118, 119), (95, 120, 121), (146, 122, 123),
        (147, 124, 125), (151, 126, 127), (166, 128, 129),
        (167, 130, 131), (168, 132, 133), (165, 134, 135),
        (174, 187, 188), (175, 189, 190), (177, 191, 192),
        (180, 193, 194), (182, 195, 196), (184, 197, 198),
        (199, 208, 209), (201, 210, 211), (202, 212, 213),
    ]

    static func liveState(game: QuantumCatGame, player: Int) throws -> QuantumCatMLLiveState {
        let state = QuantumCatMLLiveState(game: game, player: player)
        try validate(state)
        return state
    }

    static func observation(game: QuantumCatGame, player: Int) throws -> [Double] {
        try observation(state: liveState(game: game, player: player))
    }

    static func observation(
        game: QuantumCatGame,
        player: Int,
        paddedTo targetSize: Int
    ) throws -> [Double] {
        try observation(
            state: liveState(game: game, player: player),
            paddedTo: targetSize
        )
    }

    static func actionFeatures(
        game: QuantumCatGame,
        player: Int,
        legalMoves: [Move]
    ) throws -> [[Double]] {
        try actionFeatures(
            state: liveState(game: game, player: player),
            legalMoves: legalMoves
        )
    }

    static func observation(state: QuantumCatMLLiveState) throws -> [Double] {
        try validate(state)
        var values: [Double] = []
        for seat in 0..<state.numPlayers {
            values.append(!state.isTerminal && state.currentPlayer == seat ? 1.0 : 0.0)
        }
        for value in 0..<5 {
            values.append(state.phase == value ? 1.0 : 0.0)
        }
        for color in 0..<state.numColors {
            values.append(state.ledColor == colorName(color) ? 1.0 : 0.0)
        }
        values.append(state.ledColor == nil ? 1.0 : 0.0)
        values.append(Double(state.trickNumber))
        for seat in 0..<state.numPlayers {
            values.append(!state.isTerminal && state.startPlayer == seat ? 1.0 : 0.0)
        }
        for seat in 0..<state.numPlayers {
            let card = state.currentTrick[safe: seat] ?? nil
            values.append(card.map { Double($0.rank) } ?? -1.0)
            values.append(card.flatMap { colorIndex($0.suit) }.map(Double.init) ?? -1.0)
        }
        for seat in 0..<state.numPlayers {
            values.append(Double(state.predictions[safe: seat] ?? -1))
        }
        for seat in 0..<state.numPlayers {
            values.append(Double(state.tricksWon[safe: seat] ?? 0))
        }
        for color in 0..<state.numColors {
            for rank in 0..<state.numRanks {
                values.append(Double(state.board[safe: color]?[safe: rank] ?? -1))
            }
        }
        for seat in 0..<state.numPlayers {
            for color in 0..<state.numColors {
                values.append(state.colorTokens[safe: seat]?[safe: color] == true ? 1.0 : 0.0)
            }
        }
        for rank in 0..<state.numRanks {
            values.append(Double(state.actingHand[safe: rank] ?? 0))
        }
        values.append(Double(state.actingDiscardedRank))
        for seat in 0..<state.numPlayers {
            values.append((state.matchTotals[safe: seat] ?? 0.0) / 20.0)
        }
        values.append(Double(state.matchRound))
        guard let expectedSize = observationSize(forPlayers: state.numPlayers),
              values.count == expectedSize else {
            throw QuantumCatMLFeatureEncoderError.featureSize(
                expected: observationSize(forPlayers: state.numPlayers) ?? -1,
                actual: values.count
            )
        }
        return values
    }

    static func observation(
        state: QuantumCatMLLiveState,
        paddedTo targetSize: Int
    ) throws -> [Double] {
        var values = try observation(state: state)
        guard targetSize >= values.count,
              targetSize <= unifiedObservationSize else {
            throw QuantumCatMLFeatureEncoderError.featureSize(
                expected: targetSize,
                actual: values.count
            )
        }
        values.append(contentsOf: repeatElement(0.0, count: targetSize - values.count))
        return values
    }

    static func actionFeatures(
        state: QuantumCatMLLiveState,
        legalMoves: [Move]
    ) throws -> [[Double]] {
        try validate(state)
        guard !legalMoves.isEmpty else {
            throw QuantumCatMLFeatureEncoderError.invalidState("empty_legal_moves")
        }
        var rows = try legalMoves.map {
            try actionFeatureVector(state: state, move: $0, legalMoves: legalMoves)
        }
        addLegalSetContext(to: &rows)
        return rows
    }

    static func move(forPythonAction action: Int, state: QuantumCatMLLiveState) -> Move? {
        if action == 999 { return .paradox }
        switch state.phase {
        case 1:
            guard (0..<state.numRanks).contains(action) else { return nil }
            return .discard(rank: action + 1)
        case 2:
            guard (101...104).contains(action) else { return nil }
            return .prediction(action - 100)
        case 3:
            guard (0..<(state.numColors * state.numRanks)).contains(action) else { return nil }
            let color = action / state.numRanks
            let rank = action % state.numRanks
            guard Suit.allCases.indices.contains(color) else { return nil }
            return .play(rank: rank + 1, suit: Suit.allCases[color])
        default:
            return nil
        }
    }

    private static func validate(_ state: QuantumCatMLLiveState) throws {
        guard observationSize(forPlayers: state.numPlayers) != nil,
              state.numRanks == QuantumCatGame.rankCount(for: state.numPlayers),
              state.numColors == 4,
              state.numTricks == QuantumCatGame.tricksInRound(for: state.numPlayers),
              state.cardsPerPlayerInitial == QuantumCatGame.cardsPerPlayer(
                  for: state.numPlayers
              ) else {
            throw QuantumCatMLFeatureEncoderError.unsupportedContract(
                "players=\(state.numPlayers),ranks=\(state.numRanks),colors=\(state.numColors),tricks=\(state.numTricks)"
            )
        }
        guard (0..<state.numPlayers).contains(state.actingPlayer),
              state.actingHand.count == state.numRanks,
              state.board.count == state.numColors,
              state.board.allSatisfy({ $0.count == state.numRanks }),
              state.colorTokens.count == state.numPlayers,
              state.colorTokens.allSatisfy({ $0.count == state.numColors }),
              state.currentTrick.count == state.numPlayers,
              state.predictions.count == state.numPlayers,
              state.tricksWon.count == state.numPlayers,
              state.playedCounts.count == state.numPlayers else {
            throw QuantumCatMLFeatureEncoderError.invalidState("dimensions")
        }
    }

    private static func actionFeatureVector(
        state: QuantumCatMLLiveState,
        move: Move,
        legalMoves: [Move]
    ) throws -> [Double] {
        let type = actionType(move, phase: state.phase)
        let player = state.actingPlayer
        let numPlayers = state.numPlayers
        let numRanks = state.numRanks
        let numColors = state.numColors
        let numTricks = max(1, state.numTricks)
        let hand = state.actingHand.map(Double.init)
        let prediction = state.predictions[safe: player] ?? -1
        let tricks = state.tricksWon[safe: player] ?? 0
        let wantsMore = prediction < 0 || tricks != prediction
        let totalHand = hand.reduce(0.0, +)

        var rankIndex = -1
        var colorIndexValue = -1
        var predictionValue = -1
        switch move {
        case .discard(let rank): rankIndex = rank - 1
        case .prediction(let value): predictionValue = value
        case .play(let rank, let suit):
            rankIndex = rank - 1
            colorIndexValue = Suit.allCases.firstIndex(of: suit) ?? -1
        case .paradox: break
        }

        var features: [Double] = [legalMoves.contains(move) ? 1.0 : 0.0]
        for value in 0..<5 { features.append(state.phase == value ? 1.0 : 0.0) }
        for candidate in [ActionType.discard, .prediction, .play, .paradox, .other] {
            features.append(type == candidate ? 1.0 : 0.0)
        }
        for rank in 0..<maxRankFeatures { features.append(rankIndex == rank ? 1.0 : 0.0) }
        for color in 0..<maxColorFeatures { features.append(colorIndexValue == color ? 1.0 : 0.0) }
        features.append(state.ledColor == nil ? 1.0 : 0.0)
        for color in 0..<maxColorFeatures {
            features.append(state.ledColor == colorName(color) ? 1.0 : 0.0)
        }

        let rank = rankIndex >= 0 ? rankIndex + 1 : 0
        let color = colorIndexValue >= 0 ? colorName(colorIndexValue) : nil
        let handCount = hand[safe: rankIndex] ?? 0.0
        let handCountAfter = (type == .discard || type == .play)
            ? max(0.0, handCount - 1.0)
            : handCount
        let largestBefore = largestCluster(board: state.board, player: player)
        var largestAfter = largestBefore
        var adjacencyGain = 0.0
        var boardFree = 0.0
        let clusterBefore = clusterProfile(board: state.board, player: player)
        var clusterAfter = clusterBefore
        if type == .play,
           state.board.indices.contains(colorIndexValue),
           state.board[colorIndexValue].indices.contains(rankIndex) {
            boardFree = state.board[colorIndexValue][rankIndex] == -1 ? 1.0 : 0.0
            var afterBoard = state.board
            afterBoard[colorIndexValue][rankIndex] = player
            largestAfter = largestCluster(board: afterBoard, player: player)
            adjacencyGain = Double(largestAfter - largestBefore)
            clusterAfter = clusterProfile(board: afterBoard, player: player)
        }

        var followsLed = 0.0
        var offLedLosesToken = 0.0
        if type == .play, let led = state.ledColor, let color {
            followsLed = color == led ? 1.0 : 0.0
            offLedLosesToken = color == led ? 0.0 : 1.0
        }
        let wouldComplete = type == .play
            && state.currentTrick.compactMap { $0 }.count + 1 >= numPlayers
        let wouldWin = type == .play && color != nil
            ? wouldWinAfterPlay(state: state, player: player, rank: rank, color: color!)
            : nil
        let winAlignsTarget = wouldWin.map { $0 == wantsMore ? 1.0 : 0.0 } ?? 0.0
        var outcomeTricks = tricks
        if wouldWin == true { outcomeTricks += 1 }
        let outcomePredictionGap = prediction >= 0 && wouldWin != nil
            ? Double(prediction - outcomeTricks) / Double(numTricks)
            : 0.0
        let outcomeHitsPrediction = prediction >= 0 && wouldWin != nil && outcomeTricks == prediction
        let outcomeOvershootsPrediction = prediction >= 0 && wouldWin != nil && outcomeTricks > prediction
        let outcomeStillShortPrediction = prediction >= 0 && wouldWin != nil && outcomeTricks < prediction
        let wouldEndRound = wouldWin != nil && state.trickNumber + 1 >= numTricks
        let remainingTricksAfter = max(0, numTricks - state.trickNumber - (wouldComplete ? 1 : 0))
        let winsNeededAfter = prediction >= 0 && wouldWin != nil
            ? max(0, prediction - outcomeTricks)
            : 0
        let canStillHitAfter = prediction >= 0 && wouldWin != nil
            && outcomeTricks <= prediction
            && winsNeededAfter >= 0 && winsNeededAfter <= remainingTricksAfter
        let mustWinAllRemainingAfter = canStillHitAfter && remainingTricksAfter > 0
            && winsNeededAfter == remainingTricksAfter
        let hitWithFutureTricks = outcomeHitsPrediction && remainingTricksAfter > 0
        var endRoundScoreEstimate = 0.0
        if wouldEndRound {
            endRoundScoreEstimate = Double(outcomeTricks)
            if outcomeHitsPrediction { endRoundScoreEstimate += Double(largestAfter) }
            endRoundScoreEstimate /= Double(max(1, numTricks + numRanks))
        }

        let sameRankLegal = type == .play ? legalMoves.reduce(0) { count, candidate in
            guard case .play(let candidateRank, _) = candidate else { return count }
            return count + (candidateRank - 1 == rankIndex ? 1 : 0)
        } : 0
        let sameColorLegal = type == .play ? legalMoves.reduce(0) { count, candidate in
            guard case .play(_, let candidateSuit) = candidate else { return count }
            return count + ((Suit.allCases.firstIndex(of: candidateSuit) ?? -1) == colorIndexValue ? 1 : 0)
        } : 0

        let allTokensBefore = normalizedTokens(state.colorTokens, players: numPlayers, colors: numColors)
        let playerTokensBefore = allTokensBefore[player]
        let colorTokenAvailable = colorIndexValue >= 0
            && playerTokensBefore.indices.contains(colorIndexValue)
            && playerTokensBefore[colorIndexValue] ? 1.0 : 0.0

        var postHand = hand
        var postBoard = state.board
        var postTokens = playerTokensBefore
        var postRedBroken = state.trumpBroken || state.board[0].contains(where: { $0 >= 0 })
        if (type == .discard || type == .play), postHand.indices.contains(rankIndex) {
            postHand[rankIndex] = max(0.0, postHand[rankIndex] - 1.0)
        }
        if type == .play,
           postBoard.indices.contains(colorIndexValue),
           postBoard[colorIndexValue].indices.contains(rankIndex) {
            postBoard[colorIndexValue][rankIndex] = player
            if colorIndexValue == 0 { postRedBroken = true }
            if let led = state.ledColor, let played = color,
               played != led, let ledIndex = colorIndex(led),
               postTokens.indices.contains(ledIndex) {
                postTokens[ledIndex] = false
            }
        }

        let postLeadFlex = handFlexibility(
            hand: postHand, board: postBoard, tokens: postTokens,
            numRanks: numRanks, leading: true, redBroken: postRedBroken
        )
        var lostTokenIndex = -1
        var tokenLossNewlyLosesLed = 0.0
        if type == .play, let led = state.ledColor, let color, color != led,
           let ledIndex = colorIndex(led) {
            lostTokenIndex = ledIndex
            if playerTokensBefore.indices.contains(ledIndex), postTokens.indices.contains(ledIndex),
               playerTokensBefore[ledIndex], !postTokens[ledIndex] {
                tokenLossNewlyLosesLed = 1.0
            }
        }
        var noLossLeadFlex = postLeadFlex
        if tokenLossNewlyLosesLed > 0.5, postTokens.indices.contains(lostTokenIndex) {
            var noLossTokens = postTokens
            noLossTokens[lostTokenIndex] = true
            noLossLeadFlex = handFlexibility(
                hand: postHand, board: postBoard, tokens: noLossTokens,
                numRanks: numRanks, leading: true, redBroken: postRedBroken
            )
        }
        let tokenLossLegalLeadDamage = max(0.0, noLossLeadFlex.legalCount - postLeadFlex.legalCount)
        let tokenLossDeadCardDamage = max(0.0, postLeadFlex.deadCardCount - noLossLeadFlex.deadCardCount)
        let tokenLossSingletonCardDamage = max(
            0.0, postLeadFlex.singletonCardCount - noLossLeadFlex.singletonCardCount
        )
        let tokenLossRankDamage = max(0.0, noLossLeadFlex.legalRankCount - postLeadFlex.legalRankCount)
        let tokenLossCreatesNoExit = tokenLossNewlyLosesLed > 0.5
            && noLossLeadFlex.legalCount > 0 && postLeadFlex.legalCount <= 0 ? 1.0 : 0.0

        var allTokensAfter = allTokensBefore
        allTokensAfter[player] = postTokens
        let remainingCardsBefore = state.playedCounts.map { max(0, numTricks - $0) }
        var remainingCardsAfter = remainingCardsBefore
        if type == .play { remainingCardsAfter[player] = max(0, remainingCardsAfter[player] - 1) }
        let beforeExit = exitLiquidity(
            board: state.board, tokens: allTokensBefore, remainingCards: remainingCardsBefore
        )
        let afterExit = exitLiquidity(
            board: postBoard, tokens: allTokensAfter, remainingCards: remainingCardsAfter
        )
        let boardCellDenominator = Double(max(1, numColors * numRanks))
        let publicSlotDenominator = Double(max(1, numPlayers)) * boardCellDenominator
        let pressureCountDenominator = Double(max(1, numPlayers))
        let beforeOwnSlots = beforeExit.playerOpenSlots[player]
        let afterOwnSlots = afterExit.playerOpenSlots[player]
        let afterOwnLaneSurplus = afterExit.playerLaneSurplus[player]

        let exitPublicSlotDamage: Double
        let exitOwnPublicSlotDamage: Double
        let exitBoardOpenCellDamage: Double
        let exitMinPlayerOpenSlotsAfter: Double
        let exitTotalPlayerOpenSlotsAfter: Double
        let exitOwnLaneSurplusAfter: Double
        let exitMinPlayerLaneSurplusAfter: Double
        let exitTotalPlayerLaneSurplusAfter: Double
        let exitLaneSurplusDamage: Double
        let exitMinLaneSurplusDamage: Double
        let exitLanePressurePlayerCountAfter: Double
        if type == .paradox {
            exitPublicSlotDamage = Double(beforeExit.totalPlayerOpenSlots) / publicSlotDenominator
            exitOwnPublicSlotDamage = Double(beforeOwnSlots) / boardCellDenominator
            exitBoardOpenCellDamage = Double(beforeExit.openCells) / boardCellDenominator
            exitMinPlayerOpenSlotsAfter = 0.0
            exitTotalPlayerOpenSlotsAfter = 0.0
            exitOwnLaneSurplusAfter = 0.0
            exitMinPlayerLaneSurplusAfter = 0.0
            exitTotalPlayerLaneSurplusAfter = 0.0
            exitLaneSurplusDamage = Double(beforeExit.totalPlayerLaneSurplus) / publicSlotDenominator
            exitMinLaneSurplusDamage = Double(beforeExit.minPlayerLaneSurplus) / boardCellDenominator
            exitLanePressurePlayerCountAfter = 1.0
        } else {
            exitPublicSlotDamage = Double(max(
                0, beforeExit.totalPlayerOpenSlots - afterExit.totalPlayerOpenSlots
            )) / publicSlotDenominator
            exitOwnPublicSlotDamage = Double(max(0, beforeOwnSlots - afterOwnSlots)) / boardCellDenominator
            exitBoardOpenCellDamage = Double(max(0, beforeExit.openCells - afterExit.openCells)) / boardCellDenominator
            exitMinPlayerOpenSlotsAfter = Double(afterExit.minPlayerOpenSlots) / boardCellDenominator
            exitTotalPlayerOpenSlotsAfter = Double(afterExit.totalPlayerOpenSlots) / publicSlotDenominator
            exitOwnLaneSurplusAfter = Double(afterOwnLaneSurplus) / boardCellDenominator
            exitMinPlayerLaneSurplusAfter = Double(afterExit.minPlayerLaneSurplus) / boardCellDenominator
            exitTotalPlayerLaneSurplusAfter = Double(afterExit.totalPlayerLaneSurplus) / publicSlotDenominator
            exitLaneSurplusDamage = Double(max(
                0, beforeExit.totalPlayerLaneSurplus - afterExit.totalPlayerLaneSurplus
            )) / publicSlotDenominator
            exitMinLaneSurplusDamage = Double(max(
                0, beforeExit.minPlayerLaneSurplus - afterExit.minPlayerLaneSurplus
            )) / boardCellDenominator
            exitLanePressurePlayerCountAfter = Double(afterExit.lanePressurePlayerCount) / pressureCountDenominator
        }

        let clusterDenominator = Double(max(1, numColors * numRanks))
        let clusterConnectsComponents = type == .play && clusterBefore.components > 0
            && clusterAfter.components < clusterBefore.components ? 1.0 : 0.0
        let clusterDeadEndAfter = type == .play && largestAfter > largestBefore
            && clusterAfter.frontier <= 0 ? 1.0 : 0.0
        let lowCutoff = max(1, min(numRanks, 2))
        let highStart = max(0, numRanks - 2)
        let postCards = max(1.0, postHand.reduce(0.0, +))
        let lowCardCount = postHand.prefix(lowCutoff).reduce(0.0, +)
        let highCardCount = postHand[highStart..<numRanks].reduce(0.0, +)
        var lowLegalLeadCount = 0.0
        for lowRank in 0..<lowCutoff {
            let count = postHand[lowRank]
            guard count > 0 else { continue }
            var feasible = (0..<min(postBoard.count, postTokens.count)).filter {
                postTokens[$0] && postBoard[$0][lowRank] == -1
            }
            if !postRedBroken, feasible.contains(where: { $0 != 0 }) {
                feasible.removeAll(where: { $0 == 0 })
            }
            lowLegalLeadCount += count * Double(feasible.count)
        }
        let postHitGate = hitWithFutureTricks ? 1.0 : 0.0
        let postHitLowCardFraction = postHitGate * lowCardCount / postCards
        let postHitHighCardFraction = postHitGate * highCardCount / postCards
        let postHitLegalLeadCount = postHitGate * postLeadFlex.legalCount
            / Double(max(1, numRanks * maxColorFeatures))
        let postHitDeadCardFraction = postHitGate * postLeadFlex.deadCardCount / postCards
        let postHitLowLegalLeadRatio = postHitGate * lowLegalLeadCount / max(1.0, postLeadFlex.legalCount)
        let postHitSurvivalMargin = postHitGate * (lowCardCount - Double(remainingTricksAfter))
            / Double(max(1, numTricks))
        let postHitForcedPressure = postHitGate * Double(remainingTricksAfter) / postCards

        let pressureBefore = rankPressure(
            hand: hand, board: state.board, tokens: playerTokensBefore, numRanks: numRanks
        )
        let pressureAfter = rankPressure(
            hand: postHand, board: postBoard, tokens: postTokens, numRanks: numRanks
        )
        let ownFuture = feasibilityProfile(
            hand: postHand, board: postBoard, tokens: postTokens,
            numRanks: numRanks, numColors: numColors, leading: false, redBroken: true
        )
        let ownFutureLead = feasibilityProfile(
            hand: postHand, board: postBoard, tokens: postTokens,
            numRanks: numRanks, numColors: numColors,
            leading: true, redBroken: postRedBroken
        )
        let safeFlexBefore = safeFlexScore(pressureBefore, numRanks: numRanks)
        let safeFlexAfter = safeFlexScore(pressureAfter, numRanks: numRanks)
        let futureTokenFragility = futureTokenLossFragility(
            hand: postHand, board: postBoard, tokens: postTokens,
            numRanks: numRanks, basePressure: pressureAfter
        )

        var discardHandBeforeFraction = 0.0
        var discardHandAfterFraction = 0.0
        var discardRankOpenSlotsFraction = 0.0
        var discardRankSlotSurplusBefore = 0.0
        var discardRankSlotSurplusAfter = 0.0
        var discardRankDeficitRelief = 0.0
        var discardRankBufferRelief = 0.0
        var discardSafeFlexDelta = 0.0
        var discardDeadRankRelief = 0.0
        var discardTightRankRelief = 0.0
        var discardRemovesSingleton = 0.0
        var discardFromDuplicate = 0.0
        var discardNoExitAfter = 0.0
        if type == .discard, (0..<numRanks).contains(rankIndex) {
            let rankOpenSlots = Double((0..<min(postBoard.count, playerTokensBefore.count)).filter {
                playerTokensBefore[$0] && postBoard[$0][rankIndex] == -1
            }.count)
            let rankDeficitBefore = max(0.0, handCount - rankOpenSlots)
            let rankDeficitAfter = max(0.0, handCountAfter - rankOpenSlots)
            let rankBufferBefore = max(0.0, handCount + 1.0 - rankOpenSlots)
            let rankBufferAfter = max(0.0, handCountAfter + 1.0 - rankOpenSlots)
            discardHandBeforeFraction = handCount / max(1.0, totalHand)
            discardHandAfterFraction = handCountAfter / max(1.0, totalHand - 1.0)
            discardRankOpenSlotsFraction = rankOpenSlots / Double(maxColorFeatures)
            discardRankSlotSurplusBefore = (rankOpenSlots - handCount) / Double(maxColorFeatures)
            discardRankSlotSurplusAfter = (rankOpenSlots - handCountAfter) / Double(maxColorFeatures)
            discardRankDeficitRelief = max(0.0, rankDeficitBefore - rankDeficitAfter) / max(1.0, totalHand)
            discardRankBufferRelief = max(0.0, rankBufferBefore - rankBufferAfter) / max(1.0, totalHand)
            discardSafeFlexDelta = safeFlexAfter - safeFlexBefore
            discardDeadRankRelief = max(0.0, pressureBefore.deadRankCount - pressureAfter.deadRankCount)
                / Double(max(1, numRanks))
            discardTightRankRelief = max(0.0, pressureBefore.tightRankCount - pressureAfter.tightRankCount)
                / Double(max(1, numRanks))
            discardRemovesSingleton = handCount <= 1.0 ? 1.0 : 0.0
            discardFromDuplicate = handCount > 1.0 ? 1.0 : 0.0
            discardNoExitAfter = pressureAfter.noExit ? 1.0 : 0.0
        }

        let predictionHighCards = hand[max(0, numRanks - 2)..<numRanks].reduce(0.0, +)
        let predictionDuplicatePressure = hand.reduce(0.0) { $0 + max(0.0, $1 - 1.0) }
        let predictionExpectedTricks = predictionTopRankScore(state: state)
        var predictionActionGap = 0.0
        var predictionActionAbsoluteGap = 0.0
        var predictionActionUnder = 0.0
        var predictionActionOver = 0.0
        var predictionActionIsMin = 0.0
        var predictionActionIsMax = 0.0
        if type == .prediction, predictionValue > 0 {
            predictionActionGap = (Double(predictionValue) - predictionExpectedTricks) / 4.0
            predictionActionAbsoluteGap = abs(predictionActionGap)
            predictionActionUnder = Double(predictionValue) < predictionExpectedTricks ? 1.0 : 0.0
            predictionActionOver = Double(predictionValue) > predictionExpectedTricks ? 1.0 : 0.0
            predictionActionIsMin = predictionValue == 1 ? 1.0 : 0.0
            predictionActionIsMax = predictionValue == 4 ? 1.0 : 0.0
        }

        var numeric: [Double] = [
            Double(state.phase) / 4.0,
            Double(state.trickNumber) / Double(numTricks),
            Double(state.currentTrick.compactMap { $0 }.count) / Double(max(1, numPlayers)),
            Double(tricks) / Double(numTricks),
            prediction >= 0 ? Double(prediction) / 4.0 : 0.0,
            prediction >= 0 ? 1.0 : 0.0,
            wantsMore ? 1.0 : 0.0,
            prediction >= 0 && tricks >= prediction ? 1.0 : 0.0,
            prediction >= 0 ? Double(prediction - tricks) / 4.0 : 0.0,
            Double(rank) / Double(max(1, numRanks)),
            handCount / max(1.0, totalHand),
            handCountAfter / max(1.0, totalHand),
            totalHand / Double(max(1, state.cardsPerPlayerInitial)),
            max(0.0, handCount - 1.0) / max(1.0, totalHand),
            boardFree,
            colorTokenAvailable,
            followsLed,
            offLedLosesToken,
            wouldComplete ? 1.0 : 0.0,
            wouldWin.map { $0 ? 1.0 : -1.0 } ?? 0.0,
            winAlignsTarget,
            adjacencyGain / Double(max(1, numRanks)),
            Double(largestBefore) / Double(max(1, numRanks)),
            Double(largestAfter) / Double(max(1, numRanks)),
            color == "R" ? 1.0 : 0.0,
            type == .play && color == "R" && state.ledColor == nil && !state.trumpBroken ? 1.0 : 0.0,
            state.trumpBroken ? 1.0 : 0.0,
            Double(legalMoves.count) / Double(numDistinctActions),
            Double(sameRankLegal) / Double(max(1, legalMoves.count)),
            Double(sameColorLegal) / Double(max(1, legalMoves.count)),
            predictionValue > 0 ? Double(predictionValue) / 4.0 : 0.0,
            type == .paradox ? 1.0 : 0.0,
        ]
        numeric.append(contentsOf: actionFlexibility(
            state: state, type: type, rankIndex: rankIndex, colorIndex: colorIndexValue
        ))
        numeric.append(contentsOf: [
            rankIndex == 6 ? 1.0 : 0.0,
            rankIndex == 7 ? 1.0 : 0.0,
            rankIndex == 8 ? 1.0 : 0.0,
            wouldWin != nil ? 1.0 : 0.0,
            outcomeHitsPrediction ? 1.0 : 0.0,
            outcomeOvershootsPrediction ? 1.0 : 0.0,
            outcomeStillShortPrediction ? 1.0 : 0.0,
            outcomePredictionGap,
            wouldEndRound ? 1.0 : 0.0,
            endRoundScoreEstimate,
            Double(remainingTricksAfter) / Double(numTricks),
            Double(winsNeededAfter) / Double(numTricks),
            canStillHitAfter ? 1.0 : 0.0,
            mustWinAllRemainingAfter ? 1.0 : 0.0,
            hitWithFutureTricks ? 1.0 : 0.0,
            postHitLowCardFraction,
            postHitHighCardFraction,
            postHitLegalLeadCount,
            postHitDeadCardFraction,
            postHitLowLegalLeadRatio,
            postHitSurvivalMargin,
            postHitForcedPressure,
            tokenLossNewlyLosesLed,
            tokenLossLegalLeadDamage / Double(max(1, numRanks * maxColorFeatures)),
            tokenLossDeadCardDamage / postCards,
            tokenLossSingletonCardDamage / postCards,
            tokenLossRankDamage / Double(max(1, numRanks)),
            tokenLossCreatesNoExit,
            clusterAfter.frontier / clusterDenominator,
            (clusterAfter.frontier - clusterBefore.frontier) / clusterDenominator,
            (clusterAfter.components - clusterBefore.components) / Double(max(1, numRanks)),
            clusterConnectsComponents,
            clusterDeadEndAfter,
        ])
        numeric.append(contentsOf: Array(repeating: 0.0, count: 36))
        numeric.append(contentsOf: [
            pressureAfter.totalSlots / Double(max(1, numRanks * maxColorFeatures)),
            pressureAfter.rankDeficit / postCards,
            pressureAfter.tightRankCount / Double(max(1, numRanks)),
            (pressureAfter.rankDeficit - pressureBefore.rankDeficit) / max(1.0, totalHand),
            safeFlexAfter,
            pressureAfter.minRankSurplus / Double(maxColorFeatures),
            pressureAfter.maxRankDeficit / postCards,
            pressureAfter.bufferDeficit / postCards,
            pressureAfter.noExit ? 1.0 : 0.0,
            pressureAfter.deadRankCount / Double(max(1, numRanks)),
        ])
        numeric.append(contentsOf: futureTokenFragility)
        numeric.append(contentsOf: [
            predictionHighCards / max(1.0, totalHand),
            predictionDuplicatePressure / max(1.0, totalHand),
            predictionExpectedTricks / 4.0,
            predictionActionGap,
            predictionActionAbsoluteGap,
            predictionActionUnder,
            predictionActionOver,
            predictionActionIsMin,
            predictionActionIsMax,
            discardHandBeforeFraction,
            discardHandAfterFraction,
            discardRankOpenSlotsFraction,
            discardRankSlotSurplusBefore,
            discardRankSlotSurplusAfter,
            discardRankDeficitRelief,
            discardRankBufferRelief,
            discardSafeFlexDelta,
            discardDeadRankRelief,
            discardTightRankRelief,
            discardRemovesSingleton,
            discardFromDuplicate,
            discardNoExitAfter,
            exitPublicSlotDamage,
            exitOwnPublicSlotDamage,
            exitBoardOpenCellDamage,
            exitMinPlayerOpenSlotsAfter,
            exitTotalPlayerOpenSlotsAfter,
            exitOwnLaneSurplusAfter,
            exitMinPlayerLaneSurplusAfter,
            exitTotalPlayerLaneSurplusAfter,
            exitLaneSurplusDamage,
            exitMinLaneSurplusDamage,
            exitLanePressurePlayerCountAfter,
            tokenLossNewlyLosesLed,
            wouldWin == true && prediction >= 0 && tricks == prediction ? 1.0 : 0.0,
        ])
        numeric.append(contentsOf: Array(repeating: 0.0, count: 12))
        numeric.append(contentsOf: [
            ownFuture.minColors / Double(max(1, numColors)),
            ownFuture.meanColors / Double(max(1, numColors)),
            ownFuture.zeroExitFraction,
            ownFuture.oneExitFraction,
            ownFuture.twoOrLessExitFraction,
            ownFuture.sumLogColors,
            ownFutureLead.minColors / Double(max(1, numColors)),
            ownFutureLead.zeroExitFraction,
            ownFutureLead.legalCountNormalized,
        ])
        numeric.append(contentsOf: Array(repeating: 0.0, count: 6))
        features.append(contentsOf: numeric)
        guard features.count == actionFeatureSize else {
            throw QuantumCatMLFeatureEncoderError.featureSize(
                expected: actionFeatureSize, actual: features.count
            )
        }
        return features
    }

    private static func addLegalSetContext(to rows: inout [[Double]]) {
        guard !rows.isEmpty else { return }
        // Python materializes every action row as np.float32 before deriving
        // legal-set ranks. Quantize here too: exact-equality percentiles must
        // see the same ties as the training/inference contract.
        for rowIndex in rows.indices {
            rows[rowIndex] = rows[rowIndex].map { Double(Float($0)) }
        }
        for columns in legalContextColumns {
            let values = rows.map { Float($0[columns.source]) }
            let mean = values.reduce(Float(0), +) / Float(values.count)
            let variance = values.reduce(Float(0)) {
                let delta = $1 - mean
                return $0 + delta * delta
            } / Float(values.count)
            let standardDeviation = sqrt(max(Float(0), variance))
            for rowIndex in rows.indices {
                let z: Float = standardDeviation < 1e-6
                    ? 0.0
                    : (values[rowIndex] - mean) / standardDeviation
                rows[rowIndex][columns.z] = Double(z)
                let less = values.filter { $0 < values[rowIndex] }.count
                let equal = values.filter { $0 == values[rowIndex] }.count
                rows[rowIndex][columns.percentile] = Double(Float(
                    (Double(less) + 0.5 * Double(equal)) / Double(values.count)
                ))
            }
        }
    }

    private static func actionType(_ move: Move, phase: Int) -> ActionType {
        if move == .paradox { return .paradox }
        switch phase {
        case 1: return .discard
        case 2: return .prediction
        case 3: return .play
        default: return .other
        }
    }

    private static func actionFlexibility(
        state: QuantumCatMLLiveState,
        type: ActionType,
        rankIndex: Int,
        colorIndex: Int
    ) -> [Double] {
        let handBefore = state.actingHand.map(Double.init)
        let boardBefore = state.board
        let tokensBefore = state.colorTokens[state.actingPlayer]
        let redBrokenBefore = state.trumpBroken || boardBefore[0].contains(where: { $0 >= 0 })
        let leadBefore = handFlexibility(
            hand: handBefore, board: boardBefore, tokens: tokensBefore,
            numRanks: state.numRanks, leading: true, redBroken: redBrokenBefore
        )
        var handAfter = handBefore
        var boardAfter = boardBefore
        var tokensAfter = tokensBefore
        var redBrokenAfter = redBrokenBefore
        if (type == .discard || type == .play), handAfter.indices.contains(rankIndex) {
            handAfter[rankIndex] = max(0.0, handAfter[rankIndex] - 1.0)
        }
        if type == .play, boardAfter.indices.contains(colorIndex),
           boardAfter[colorIndex].indices.contains(rankIndex) {
            boardAfter[colorIndex][rankIndex] = state.actingPlayer
            if colorIndex == 0 { redBrokenAfter = true }
            if let led = state.ledColor, colorName(colorIndex) != led,
               let ledIndex = self.colorIndex(led), tokensAfter.indices.contains(ledIndex) {
                tokensAfter[ledIndex] = false
            }
        }
        let leadAfter = handFlexibility(
            hand: handAfter, board: boardAfter, tokens: tokensAfter,
            numRanks: state.numRanks, leading: true, redBroken: redBrokenAfter
        )
        let nonLeadAfter = handFlexibility(
            hand: handAfter, board: boardAfter, tokens: tokensAfter,
            numRanks: state.numRanks, leading: false, redBroken: true
        )
        let actionDenominator = Double(max(1, state.numRanks * maxColorFeatures))
        let cardDenominator = max(1.0, leadBefore.totalCards)
        return [
            leadBefore.legalCount / actionDenominator,
            leadAfter.legalCount / actionDenominator,
            (leadAfter.legalCount - leadBefore.legalCount) / actionDenominator,
            nonLeadAfter.legalCount / actionDenominator,
            leadAfter.deadCardCount / cardDenominator,
            (leadAfter.deadCardCount - leadBefore.deadCardCount) / cardDenominator,
            leadAfter.legalRankCount / Double(max(1, state.numRanks)),
            leadAfter.legalColorCount / Double(maxColorFeatures),
            handAfter.reduce(0.0, +) > 0 && leadAfter.legalCount <= 0 ? 1.0 : 0.0,
        ]
    }

    private static func handFlexibility(
        hand: [Double], board: [[Int]], tokens: [Bool], numRanks: Int,
        leading: Bool, redBroken: Bool
    ) -> Flexibility {
        var legalCount = 0
        var legalRanks = Set<Int>()
        var legalColors = Set<Int>()
        var deadCards = 0
        var singletonCards = 0
        for rank in 0..<numRanks {
            let count = Int(hand[safe: rank] ?? 0.0)
            guard count > 0 else { continue }
            var colors = (0..<min(board.count, tokens.count)).filter {
                tokens[$0] && board[$0][rank] == -1
            }
            if leading, !redBroken, colors.contains(where: { $0 != 0 }) {
                colors.removeAll(where: { $0 == 0 })
            }
            if colors.isEmpty {
                deadCards += count
            } else {
                legalCount += colors.count * count
                legalRanks.insert(rank)
                legalColors.formUnion(colors)
                if colors.count == 1 { singletonCards += count }
            }
        }
        return Flexibility(
            legalCount: Double(legalCount),
            legalRankCount: Double(legalRanks.count),
            legalColorCount: Double(legalColors.count),
            deadCardCount: Double(deadCards),
            singletonCardCount: Double(singletonCards),
            totalCards: Double(max(1, Int(hand.reduce(0.0, +))))
        )
    }

    private static func rankPressure(
        hand: [Double], board: [[Int]], tokens: [Bool], numRanks: Int
    ) -> Pressure {
        var totalSlots = 0.0
        var rankDeficit = 0.0
        var tightRankCount = 0.0
        var minRankSurplus: Double?
        var maxRankDeficit = 0.0
        var bufferDeficit = 0.0
        var deadRankCount = 0.0
        var remainingCards = 0
        for rank in 0..<numRanks {
            let count = Int(hand[safe: rank] ?? 0.0)
            guard count > 0 else { continue }
            remainingCards += count
            let available = (0..<min(board.count, tokens.count)).filter {
                tokens[$0] && board[$0][rank] == -1
            }.count
            totalSlots += Double(available)
            let deficit = Double(max(0, count - available))
            rankDeficit += deficit
            maxRankDeficit = max(maxRankDeficit, deficit)
            let surplus = Double(available - count)
            minRankSurplus = min(minRankSurplus ?? surplus, surplus)
            bufferDeficit += Double(max(0, count + 1 - available))
            if available <= 0 { deadRankCount += 1.0 }
            if available <= count { tightRankCount += 1.0 }
        }
        return Pressure(
            totalSlots: totalSlots,
            rankDeficit: rankDeficit,
            tightRankCount: tightRankCount,
            minRankSurplus: minRankSurplus ?? 0.0,
            maxRankDeficit: maxRankDeficit,
            bufferDeficit: bufferDeficit,
            noExit: remainingCards > 0 && totalSlots <= 0,
            deadRankCount: deadRankCount
        )
    }

    private static func safeFlexScore(_ pressure: Pressure, numRanks: Int) -> Double {
        (0.20 * pressure.totalSlots - pressure.tightRankCount - 30.0 * pressure.rankDeficit)
            / max(1.0, 30.0 * Double(numRanks))
    }

    private static func exitLiquidity(
        board: [[Int]], tokens: [[Bool]], remainingCards: [Int]
    ) -> ExitLiquidity {
        let openByColor = board.map { $0.filter { $0 == -1 }.count }
        let playerOpenSlots = tokens.map { row in
            openByColor.indices.reduce(0) { total, color in
                total + (row[safe: color] == true ? openByColor[color] : 0)
            }
        }
        let normalizedRemaining = tokens.indices.map { remainingCards[safe: $0] ?? 0 }
        let laneSurplus = tokens.indices.map { playerOpenSlots[$0] - normalizedRemaining[$0] }
        return ExitLiquidity(
            openCells: openByColor.reduce(0, +),
            playerOpenSlots: playerOpenSlots,
            playerRemainingCards: normalizedRemaining,
            playerLaneSurplus: laneSurplus,
            totalPlayerOpenSlots: playerOpenSlots.reduce(0, +),
            minPlayerOpenSlots: playerOpenSlots.min() ?? 0,
            totalPlayerLaneSurplus: laneSurplus.reduce(0, +),
            minPlayerLaneSurplus: laneSurplus.min() ?? 0,
            lanePressurePlayerCount: laneSurplus.filter { $0 < 0 }.count
        )
    }

    private static func feasibilityProfile(
        hand: [Double], board: [[Int]], tokens: [Bool], numRanks: Int,
        numColors: Int, leading: Bool, redBroken: Bool
    ) -> FeasibilityProfile {
        let totalCards = hand.reduce(0.0, +)
        let colorDenominator = Double(max(1, numColors))
        if totalCards <= 0 {
            return FeasibilityProfile(
                minColors: colorDenominator, meanColors: colorDenominator,
                zeroExitFraction: 0.0, oneExitFraction: 0.0,
                twoOrLessExitFraction: 0.0, sumLogColors: 1.0,
                legalCountNormalized: 1.0
            )
        }
        var minColors = colorDenominator
        var totalFeasible = 0.0
        var zeroExitCards = 0.0
        var oneExitCards = 0.0
        var twoOrLessExitCards = 0.0
        var sumLogColors = 0.0
        let logDenominator = log1p(colorDenominator)
        for rank in 0..<numRanks {
            let count = hand[safe: rank] ?? 0.0
            guard count > 0 else { continue }
            var colors = (0..<min(min(board.count, tokens.count), numColors)).filter {
                tokens[$0] && board[$0][rank] == -1
            }
            if leading, !redBroken, colors.contains(where: { $0 != 0 }) {
                colors.removeAll(where: { $0 == 0 })
            }
            let feasible = Double(colors.count)
            minColors = min(minColors, feasible)
            totalFeasible += count * feasible
            if feasible <= 0 { zeroExitCards += count }
            if feasible <= 1 { oneExitCards += count }
            if feasible <= 2 { twoOrLessExitCards += count }
            sumLogColors += count * log1p(feasible)
        }
        return FeasibilityProfile(
            minColors: minColors,
            meanColors: totalFeasible / max(1.0, totalCards),
            zeroExitFraction: zeroExitCards / max(1.0, totalCards),
            oneExitFraction: oneExitCards / max(1.0, totalCards),
            twoOrLessExitFraction: twoOrLessExitCards / max(1.0, totalCards),
            sumLogColors: sumLogColors / max(1e-8, totalCards * logDenominator),
            legalCountNormalized: totalFeasible / max(1.0, totalCards * colorDenominator)
        )
    }

    private static func futureTokenLossFragility(
        hand: [Double], board: [[Int]], tokens: [Bool], numRanks: Int,
        basePressure: Pressure
    ) -> [Double] {
        let tokenCount = tokens.filter { $0 }.count
        let postCards = max(1.0, hand.reduce(0.0, +))
        let baseScore = safeFlexScore(basePressure, numRanks: numRanks)
        guard tokenCount > 0 else { return [0.0, 0.0, 0.0, 0.0, baseScore, 0.0] }
        var maxRankDeficit = 0.0
        var maxBufferDeficit = 0.0
        var maxDeadRankCount = 0.0
        var noExitCount = 0.0
        var worstScore = baseScore
        for color in tokens.indices where tokens[color] {
            var reduced = tokens
            reduced[color] = false
            let pressure = rankPressure(hand: hand, board: board, tokens: reduced, numRanks: numRanks)
            maxRankDeficit = max(maxRankDeficit, pressure.rankDeficit)
            maxBufferDeficit = max(maxBufferDeficit, pressure.bufferDeficit)
            maxDeadRankCount = max(maxDeadRankCount, pressure.deadRankCount)
            if pressure.noExit { noExitCount += 1.0 }
            worstScore = min(worstScore, safeFlexScore(pressure, numRanks: numRanks))
        }
        return [
            maxRankDeficit / postCards,
            maxBufferDeficit / postCards,
            noExitCount / Double(max(1, tokenCount)),
            maxDeadRankCount / Double(max(1, numRanks)),
            worstScore,
            max(0.0, baseScore - worstScore),
        ]
    }

    private static func normalizedTokens(
        _ tokens: [[Bool]], players: Int, colors: Int
    ) -> [[Bool]] {
        (0..<players).map { player in
            (0..<colors).map { color in tokens[safe: player]?[safe: color] ?? false }
        }
    }

    private static func largestCluster(board: [[Int]], player: Int) -> Int {
        guard let rankCount = board.first?.count else { return 0 }
        var visited = Array(repeating: Array(repeating: false, count: rankCount), count: board.count)
        var best = 0
        for color in board.indices {
            for rank in 0..<rankCount where board[color][rank] == player && !visited[color][rank] {
                var stack = [(color, rank)]
                visited[color][rank] = true
                var size = 0
                while let cell = stack.popLast() {
                    size += 1
                    for delta in [(1, 0), (-1, 0), (0, 1), (0, -1)] {
                        let nextColor = cell.0 + delta.0
                        let nextRank = cell.1 + delta.1
                        guard board.indices.contains(nextColor), (0..<rankCount).contains(nextRank),
                              !visited[nextColor][nextRank], board[nextColor][nextRank] == player else { continue }
                        visited[nextColor][nextRank] = true
                        stack.append((nextColor, nextRank))
                    }
                }
                best = max(best, size)
            }
        }
        return best
    }

    private static func clusterProfile(board: [[Int]], player: Int) -> ClusterProfile {
        guard let rankCount = board.first?.count else { return ClusterProfile(components: 0, frontier: 0) }
        var visited = Array(repeating: Array(repeating: false, count: rankCount), count: board.count)
        var components = 0
        var frontier = Set<String>()
        for color in board.indices {
            for rank in 0..<rankCount where board[color][rank] == player && !visited[color][rank] {
                components += 1
                var stack = [(color, rank)]
                visited[color][rank] = true
                while let cell = stack.popLast() {
                    for delta in [(1, 0), (-1, 0), (0, 1), (0, -1)] {
                        let nextColor = cell.0 + delta.0
                        let nextRank = cell.1 + delta.1
                        guard board.indices.contains(nextColor), (0..<rankCount).contains(nextRank) else { continue }
                        if board[nextColor][nextRank] == -1 {
                            frontier.insert("\(nextColor):\(nextRank)")
                        } else if board[nextColor][nextRank] == player && !visited[nextColor][nextRank] {
                            visited[nextColor][nextRank] = true
                            stack.append((nextColor, nextRank))
                        }
                    }
                }
            }
        }
        return ClusterProfile(components: Double(components), frontier: Double(frontier.count))
    }

    private static func wouldWinAfterPlay(
        state: QuantumCatMLLiveState, player: Int, rank: Int, color: String
    ) -> Bool? {
        var plays = state.currentTrick
        guard plays.indices.contains(player) else { return nil }
        plays[player] = QuantumCatMLPlayedCard(rank: rank, suit: color)
        guard plays.allSatisfy({ $0 != nil }) else { return nil }
        let concrete = plays.enumerated().compactMap { index, card in card.map { (index, $0) } }
        let red = concrete.filter { $0.1.suit == "R" }
        if let winner = stableHighest(red) { return winner == player }
        let led = state.ledColor ?? color
        let ledPlays = concrete.filter { $0.1.suit == led }
        let pool = ledPlays.isEmpty ? concrete : ledPlays
        return stableHighest(pool) == player
    }

    private static func stableHighest(_ plays: [(Int, QuantumCatMLPlayedCard)]) -> Int? {
        var winner: (Int, Int)?
        for (seat, card) in plays {
            if winner == nil || card.rank > winner!.1 { winner = (seat, card.rank) }
        }
        return winner?.0
    }

    private static func predictionTopRankScore(state: QuantumCatMLLiveState) -> Double {
        guard state.numRanks > 0, !state.actingHand.isEmpty else { return 0.0 }
        let top = min(state.numRanks, state.actingHand.count) - 1
        let second = top - 1
        var score = Double(state.actingHand[top])
        if second >= 0 {
            score += (state.actingPlayer == state.roundStartPlayer ? 0.50 : 0.25)
                * Double(state.actingHand[second])
        }
        return score
    }

    private static func colorName(_ index: Int) -> String? {
        ["R", "B", "Y", "G"][safe: index]
    }

    private static func colorIndex(_ name: String) -> Int? {
        ["R", "B", "Y", "G"].firstIndex(of: name)
    }
}
