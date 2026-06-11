import SwiftUI

struct GameTableView: View {
    @EnvironmentObject private var store: GameStore

    var body: some View {
        ZStack(alignment: .top) {
            ScrollView {
                VStack(spacing: 14) {
                    header
                    activeTurnPanel
                    currentTrickPanel
                    boardPanel
                    if let summary = store.bulkSimulationSummary {
                        bulkSimulationPanel(summary)
                    }
                    scoreboard
                    logPanel
                }
                .padding(14)
                .padding(.bottom, 24)
            }
            .transaction { transaction in
                transaction.animation = nil
            }

            if let animation = store.botMoveAnimation {
                botMoveOverlay(animation)
                    .padding(.horizontal, 14)
                    .padding(.top, 8)
                    .transition(.opacity)
                    .allowsHitTesting(false)
                    .zIndex(2)
            }
        }
        .background(
            LinearGradient(
                colors: [Color(red: 0.03, green: 0.07, blue: 0.05), Color(red: 0.05, green: 0.16, blue: 0.12)],
                startPoint: .top,
                endPoint: .bottom
            )
            .ignoresSafeArea()
        )
        .navigationTitle("Quantum Cat")
        .navigationBarTitleDisplayMode(.inline)
    }

    private var header: some View {
        HStack(spacing: 10) {
            ForEach(GamePhase.allCasesForDisplay, id: \.self) { phase in
                Text(phase.title)
                    .font(.caption.weight(.black))
                    .foregroundStyle(store.game.phase == phase ? .yellow : .secondary)
                    .padding(.horizontal, 10)
                    .padding(.vertical, 7)
                    .background(
                        Capsule()
                            .fill(store.game.phase == phase ? Color.yellow.opacity(0.17) : Color.white.opacity(0.06))
                    )
                    .overlay(
                        Capsule()
                            .stroke(store.game.phase == phase ? Color.yellow.opacity(0.7) : Color.white.opacity(0.08), lineWidth: 1)
                    )
            }
            Spacer()
            Text(store.game.isTerminal ? "Finished" : "P\(store.game.currentPlayer)")
                .font(.subheadline.weight(.black))
                .foregroundStyle(.white)
        }
    }

    private var activeTurnPanel: some View {
        VStack(spacing: 14) {
            HStack(alignment: .top) {
                VStack(alignment: .leading, spacing: 4) {
                    Text(store.game.phase.rawValue.uppercased())
                        .font(.caption.weight(.black))
                        .foregroundStyle(.yellow)
                    Text(turnTitle)
                        .font(.system(size: 34, weight: .black, design: .serif))
                        .foregroundStyle(.white)
                    Text(turnSubtitle)
                        .font(.subheadline)
                        .foregroundStyle(.white.opacity(0.78))
                }
                Spacer()
                VStack(alignment: .trailing, spacing: 6) {
                    stat("Trick", "\(store.game.trickNumber)/\(store.game.tricksInRound)")
                    stat("Lead", store.game.ledSuit?.rawValue ?? "none")
                }
            }

            turnActionViewport
        }
        .padding(16)
        .accessibilityIdentifier("active-turn-panel")
        .background(
            RoundedRectangle(cornerRadius: 10)
                .fill(Color(red: 0.04, green: 0.17, blue: 0.12))
                .overlay(RoundedRectangle(cornerRadius: 10).stroke(Color.yellow.opacity(0.35), lineWidth: 1))
        )
    }

    private var turnActionViewport: some View {
        Group {
            if let human = store.game.activeHuman {
                ScrollView(.vertical) {
                    VStack(alignment: .leading, spacing: 14) {
                        handView(for: human)
                        if store.game.phase == .play {
                            legalMovesView
                        }
                    }
                    .frame(maxWidth: .infinity, alignment: .topLeading)
                }
                .scrollIndicators(.visible)
                .accessibilityIdentifier("turn-action-viewport")
            } else if store.game.isTerminal {
                Text("Final returns are shown below.")
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding()
                    .background(Color.white.opacity(0.07), in: RoundedRectangle(cornerRadius: 8))
            } else {
                Text(store.botStatus ?? "Bots are thinking.")
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding()
                    .background(Color.white.opacity(0.07), in: RoundedRectangle(cornerRadius: 8))
                    .accessibilityIdentifier("bot-status")
            }
        }
        .frame(height: turnActionViewportHeight, alignment: .top)
        .clipped()
        .transaction { transaction in
            transaction.animation = nil
        }
    }

    private var turnActionViewportHeight: CGFloat {
        if store.game.isTerminal { return 74 }
        switch store.game.phase {
        case .discard:
            return 184
        case .prediction:
            return 154
        case .play:
            return 324
        default:
            return 112
        }
    }

    private var turnTitle: String {
        if store.game.isTerminal { return "Round finished" }
        guard let human = store.game.activeHuman else { return "Bot turn" }
        switch store.game.phase {
        case .discard: return "P\(human.id), discard one rank"
        case .prediction: return "P\(human.id), choose your bid"
        case .play: return "P\(human.id), choose a play"
        default: return "Your turn"
        }
    }

    private var turnSubtitle: String {
        if store.game.activeHuman == nil, !store.game.isTerminal {
            return "Watch the bot resolve its move."
        }
        return switch store.game.phase {
        case .discard: "Pass the phone to the active player and discard face down."
        case .prediction: "Bid how many tricks this seat expects to take."
        case .play: "Tap a legal move or the matching board cell."
        case .scoring: "Scores are final for this round."
        default: "Dealing automatically."
        }
    }

    private func stat(_ title: String, _ value: String) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(title.uppercased()).font(.caption2.weight(.black)).foregroundStyle(.secondary)
            Text(value).font(.headline.weight(.black)).foregroundStyle(.white)
        }
        .padding(8)
        .frame(minWidth: 72, alignment: .leading)
        .background(Color.white.opacity(0.08), in: RoundedRectangle(cornerRadius: 8))
    }

    private func handView(for player: PlayerState) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Hand").font(.caption.weight(.black)).foregroundStyle(.yellow)
            LazyVGrid(columns: Array(repeating: GridItem(.flexible(), spacing: 8), count: min(store.game.rankCount, 6)), spacing: 8) {
                ForEach(0..<store.game.rankCount, id: \.self) { index in
                    let rank = index + 1
                    let move = handMove(for: rank)
                    Button {
                        if let move {
                            store.apply(move)
                        }
                    } label: {
                        handRankCard(rank: rank, held: player.hand[index], isSelectable: move != nil)
                    }
                    .buttonStyle(.plain)
                    .disabled(move == nil)
                    .accessibilityLabel(handRankAccessibilityLabel(rank: rank, held: player.hand[index], move: move))
                    .accessibilityIdentifier("hand-rank-\(rank)")
                }
            }
        }
    }

    private func handMove(for rank: Int) -> Move? {
        switch store.game.phase {
        case .discard:
            let move = Move.discard(rank: rank)
            return store.game.legalMoves.contains(move) ? move : nil
        case .prediction:
            let move = Move.prediction(rank)
            return store.game.legalMoves.contains(move) ? move : nil
        default:
            return nil
        }
    }

    private func handRankCard(rank: Int, held: Int, isSelectable: Bool) -> some View {
        VStack {
            Text("\(rank)")
                .font(.system(size: 26, weight: .black, design: .serif))
            Text("\(held) held")
                .font(.caption2.weight(.semibold))
                .foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity, minHeight: 70)
        .background((held > 0 || isSelectable) ? Color.white.opacity(0.92) : Color.gray.opacity(0.45), in: RoundedRectangle(cornerRadius: 8))
        .overlay(
            RoundedRectangle(cornerRadius: 8)
                .stroke(isSelectable ? Color.green.opacity(0.78) : Color.clear, lineWidth: 2)
        )
        .foregroundStyle(Color(red: 0.11, green: 0.12, blue: 0.11))
    }

    private func handRankAccessibilityLabel(rank: Int, held: Int, move: Move?) -> String {
        if let move {
            switch move {
            case .discard:
                return "Discard \(rank)"
            case .prediction(let value):
                return "Bid \(value)"
            default:
                break
            }
        }
        return "Rank \(rank), \(held) held"
    }

    private var legalMovesView: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Legal moves").font(.caption.weight(.black)).foregroundStyle(.yellow)
            LazyVGrid(columns: [GridItem(.adaptive(minimum: 88), spacing: 8)], spacing: 8) {
                ForEach(store.game.legalMoves) { move in
                    Button {
                        store.apply(move)
                    } label: {
                        VStack(spacing: 3) {
                            Text(move.primary.uppercased())
                                .font(.caption2.weight(.black))
                            Text(move.value)
                                .font(.system(size: 28, weight: .black, design: .serif))
                        }
                        .frame(maxWidth: .infinity, minHeight: 68)
                    }
                    .buttonStyle(.plain)
                    .accessibilityLabel("\(move.primary) \(move.value)")
                    .accessibilityIdentifier("legal-move-\(move.id)")
                    .foregroundStyle(.white)
                    .background(moveColor(move), in: RoundedRectangle(cornerRadius: 8))
                    .overlay(RoundedRectangle(cornerRadius: 8).stroke(Color.yellow.opacity(0.35), lineWidth: 1))
                }
            }
        }
    }

    private var currentTrickPanel: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Current trick").font(.caption.weight(.black)).foregroundStyle(.yellow)
            LazyVGrid(columns: [GridItem(.adaptive(minimum: 104), spacing: 8)], spacing: 8) {
                ForEach(store.game.players) { player in
                    currentTrickCard(for: player)
                }
            }
        }
        .panelStyle()
    }

    private func currentTrickCard(for player: PlayerState) -> some View {
        let card = store.game.currentTrick[player.id]
        let isLedSuit = card?.suit == store.game.ledSuit && store.game.ledSuit != nil
        let isAnimatingBotMove = store.botMoveAnimation?.player == player.id
        return HStack {
            Text("P\(player.id)")
                .font(.caption.weight(.black))
            Spacer()
            if let card {
                Text("\(card.suit.rawValue)\(card.rank)").font(.headline.weight(.black))
            } else {
                Text("-").foregroundStyle(.secondary)
            }
        }
        .padding(10)
        .background(
            RoundedRectangle(cornerRadius: 8)
                .fill(isAnimatingBotMove ? playerColor(player.id).opacity(0.24) : isLedSuit ? ledSuitHighlightColor.opacity(0.14) : Color.white.opacity(0.07))
                .shadow(color: isAnimatingBotMove ? playerColor(player.id).opacity(0.55) : isLedSuit ? ledSuitHighlightColor.opacity(0.45) : .clear, radius: isAnimatingBotMove ? 18 : 14)
        )
        .overlay(
            RoundedRectangle(cornerRadius: 8)
                .stroke(isAnimatingBotMove ? playerColor(player.id).opacity(0.92) : isLedSuit ? ledSuitHighlightColor.opacity(0.82) : Color.clear, lineWidth: isAnimatingBotMove ? 3 : 2)
                .allowsHitTesting(false)
        )
        .scaleEffect(isAnimatingBotMove ? 1.035 : 1)
        .animation(.spring(response: 0.26, dampingFraction: 0.72), value: store.botMoveAnimation?.id)
    }

    private func botMoveOverlay(_ event: BotMoveAnimation) -> some View {
        HStack(spacing: 10) {
            Text("P\(event.player)")
                .font(.caption.weight(.black))
                .foregroundStyle(.white)
                .frame(width: 34, height: 34)
                .background(playerColor(event.player), in: Circle())
                .overlay(Circle().stroke(Color.white.opacity(0.28), lineWidth: 1))

            VStack(alignment: .leading, spacing: 2) {
                Text(event.title)
                    .font(.caption.weight(.black))
                    .foregroundStyle(.white.opacity(0.78))
                Text(event.subtitle)
                    .font(.headline.weight(.black))
                    .foregroundStyle(.white)
            }

            Spacer(minLength: 8)

            moveToken(event.move, player: event.player)
        }
        .padding(10)
        .background(
            RoundedRectangle(cornerRadius: 10)
                .fill(Color(red: 0.04, green: 0.12, blue: 0.09).opacity(0.96))
                .shadow(color: playerColor(event.player).opacity(0.42), radius: 16, x: 0, y: 6)
        )
        .overlay(
            RoundedRectangle(cornerRadius: 10)
                .stroke(playerColor(event.player).opacity(0.72), lineWidth: 1.5)
        )
        .accessibilityElement(children: .combine)
        .accessibilityLabel("\(event.title) \(event.subtitle)")
        .accessibilityIdentifier("bot-move-animation")
    }

    private func moveToken(_ move: Move, player: Int) -> some View {
        VStack(spacing: 0) {
            Text(move.primary.uppercased())
                .font(.caption2.weight(.black))
            Text(move.value)
                .font(.system(size: 22, weight: .black, design: .serif))
        }
        .foregroundStyle(.white)
        .frame(width: 58, height: 48)
        .background(moveTokenColor(move, player: player), in: RoundedRectangle(cornerRadius: 8))
        .overlay(RoundedRectangle(cornerRadius: 8).stroke(Color.white.opacity(0.2), lineWidth: 1))
    }

    private var boardPanel: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text("Public board").font(.caption.weight(.black)).foregroundStyle(.yellow)
                Spacer()
                if let ledSuit = store.game.ledSuit {
                    ledSuitBadge(ledSuit)
                }
                Text(store.game.phase == .play ? "Green legal · gold this trick" : "Claims unlock during play")
                    .font(.caption2.weight(.semibold))
                    .foregroundStyle(.secondary)
            }
            Grid(horizontalSpacing: 5, verticalSpacing: 5) {
                GridRow {
                    Text("")
                    ForEach(1...store.game.rankCount, id: \.self) { rank in
                        Text("\(rank)").font(.caption.weight(.black)).foregroundStyle(.secondary)
                    }
                }
                ForEach(Array(Suit.allCases.enumerated()), id: \.offset) { suitIndex, suit in
                    GridRow {
                        Text(suit.rawValue)
                            .font(.headline.weight(.black))
                            .frame(width: 34, height: 42)
                            .background(suitColor(suit), in: RoundedRectangle(cornerRadius: 7))
                            .overlay(ledSuitStroke(for: suit, cornerRadius: 7, lineWidth: 2))
                            .shadow(color: ledSuitShadowColor(for: suit), radius: 12)
                        ForEach(0..<store.game.rankCount, id: \.self) { rankIndex in
                            boardCell(suitIndex: suitIndex, rankIndex: rankIndex)
                        }
                    }
                }
            }
        }
        .panelStyle(light: true)
    }

    private func ledSuitBadge(_ suit: Suit) -> some View {
        HStack(spacing: 5) {
            Circle()
                .fill(suitColor(suit))
                .frame(width: 8, height: 8)
            Text("Led suit: \(suit.rawValue)")
        }
        .font(.caption2.weight(.black))
        .foregroundStyle(.white)
        .padding(.horizontal, 9)
        .padding(.vertical, 6)
        .background(
            Capsule()
                .fill(ledSuitHighlightColor.opacity(0.24))
                .shadow(color: ledSuitHighlightColor.opacity(0.34), radius: 10)
        )
        .overlay(Capsule().stroke(ledSuitHighlightColor.opacity(0.72), lineWidth: 1))
        .allowsHitTesting(false)
        .accessibilityLabel("Led suit \(suit.rawValue)")
    }

    private func boardCell(suitIndex: Int, rankIndex: Int) -> some View {
        let owner = store.game.board[suitIndex][rankIndex]
        let suit = Suit.allCases[suitIndex]
        let rank = rankIndex + 1
        let move = Move.play(rank: rankIndex + 1, suit: Suit.allCases[suitIndex])
        let legal = store.game.legalMoves.contains(move)
        let currentTrickPlayer = currentTrickPlayer(for: suit, rank: rank)
        return Button {
            if legal { store.apply(move) }
        } label: {
            ZStack(alignment: .topLeading) {
                ZStack(alignment: .bottomTrailing) {
                    Text("\(rank)")
                        .font(.system(size: 19, weight: .black, design: .serif))
                        .frame(maxWidth: .infinity, minHeight: 42)
                    if owner >= 0 {
                        Text("P\(owner)")
                            .font(.caption2.weight(.black))
                            .padding(3)
                    }
                }
                if currentTrickPlayer != nil {
                    Circle()
                        .fill(currentTrickHighlightColor)
                        .frame(width: 8, height: 8)
                        .padding(5)
                        .shadow(color: currentTrickHighlightColor.opacity(0.85), radius: 5)
                        .allowsHitTesting(false)
                }
            }
        }
        .contentShape(RoundedRectangle(cornerRadius: 7))
        .buttonStyle(.plain)
        .accessibilityIdentifier("board-cell-\(Suit.allCases[suitIndex].rawValue)-\(rankIndex + 1)")
        .accessibilityLabel(boardCellAccessibilityLabel(suit: suit, rank: rank, owner: owner, currentTrickPlayer: currentTrickPlayer))
        .foregroundStyle(owner >= 0 ? .white : Color(red: 0.13, green: 0.12, blue: 0.1))
        .background(boardCellColor(owner: owner), in: RoundedRectangle(cornerRadius: 7))
        .overlay(
            RoundedRectangle(cornerRadius: 7)
                .fill(isLedSuit(suit) && owner < 0 ? ledSuitHighlightColor.opacity(0.11) : Color.clear)
                .allowsHitTesting(false)
        )
        .overlay(
            RoundedRectangle(cornerRadius: 7)
                .stroke(legal ? Color.green : Color.black.opacity(0.1), lineWidth: legal ? 2 : 1)
                .allowsHitTesting(false)
        )
        .overlay(currentTrickStroke(isCurrentTrick: currentTrickPlayer != nil, cornerRadius: 7))
        .overlay(ledSuitStroke(for: suit, cornerRadius: 7, lineWidth: legal ? 1.5 : 2))
        .shadow(color: ledSuitShadowColor(for: suit), radius: isLedSuit(suit) ? 10 : 0)
        .shadow(color: currentTrickPlayer != nil ? currentTrickHighlightColor.opacity(0.72) : .clear, radius: currentTrickPlayer != nil ? 8 : 0)
        .disabled(!legal)
    }

    private func currentTrickPlayer(for suit: Suit, rank: Int) -> Int? {
        store.game.currentTrick.enumerated().first { _, card in
            card?.suit == suit && card?.rank == rank
        }?.offset
    }

    private func boardCellAccessibilityLabel(suit: Suit, rank: Int, owner: Int, currentTrickPlayer: Int?) -> String {
        if let currentTrickPlayer {
            return "P\(currentTrickPlayer) played \(suit.rawValue)\(rank) this trick"
        }
        if owner >= 0 {
            return "P\(owner) claimed \(suit.rawValue)\(rank)"
        }
        return "\(suit.rawValue)\(rank)"
    }

    private var scoreboard: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Scoreboard").font(.caption.weight(.black)).foregroundStyle(.yellow)
            ForEach(store.game.players) { player in
                HStack(spacing: 10) {
                    Text("P\(player.id)")
                        .font(.caption.weight(.black))
                        .frame(width: 34, height: 34)
                        .foregroundStyle(.white)
                        .background(playerColor(player.id), in: Circle())
                        .overlay(Circle().stroke(player.id == store.game.currentPlayer && !store.game.isTerminal ? Color.green : Color.white.opacity(0.15), lineWidth: player.id == store.game.currentPlayer && !store.game.isTerminal ? 2 : 1))
                    VStack(alignment: .leading, spacing: 2) {
                        Text(playerSeatLabel(player))
                            .font(.headline.weight(.black))
                        Text(player.hasParadoxed ? "paradox" : player.seatKind.isHuman ? "human" : "bot")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                    Spacer()
                    metric("Tricks", "\(player.tricksWon)")
                    metric("Bid", player.prediction.map(String.init) ?? "-")
                    metric("Score", player.score.map { String(format: "%.0f", $0) } ?? "-")
                }
                .padding(10)
                .background(Color.white.opacity(player.id == store.game.currentPlayer && !store.game.isTerminal ? 0.12 : 0.06), in: RoundedRectangle(cornerRadius: 8))
            }
        }
        .panelStyle()
    }

    private func bulkSimulationPanel(_ summary: BulkSimulationSummary) -> some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Text("Bulk simulation")
                    .font(.caption.weight(.black))
                    .foregroundStyle(.yellow)
                Spacer()
                Text("\(summary.games) games")
                    .font(.caption.weight(.black))
                    .foregroundStyle(.secondary)
            }
            HStack(spacing: 8) {
                metric("Any paradox", percent(summary.gameParadoxRate))
                metric("Seat paradox", percent(summary.seatParadoxRate))
                metric("Core ML", coreMLUsageText(summary.mlUsage))
            }
            .padding(10)
            .background(Color.white.opacity(0.06), in: RoundedRectangle(cornerRadius: 8))
            ForEach(summary.players) { result in
                HStack(spacing: 10) {
                    Text("P\(result.id)")
                        .font(.caption.weight(.black))
                        .frame(width: 34, height: 34)
                        .foregroundStyle(.white)
                        .background(playerColor(result.id), in: Circle())
                    VStack(alignment: .leading, spacing: 2) {
                        Text(result.botName)
                            .font(.headline.weight(.black))
                        Text("\(result.wins) top-score games · \(percent(result.paradoxRate)) paradox")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                    Spacer()
                    metric("Avg", String(format: "%.1f", result.averageScore))
                    metric("Best", String(format: "%.0f", result.bestScore))
                }
                .padding(10)
                .background(Color.white.opacity(0.06), in: RoundedRectangle(cornerRadius: 8))
            }
        }
        .panelStyle()
        .accessibilityIdentifier("bulk-simulation-summary")
    }

    private func percent(_ value: Double) -> String {
        String(format: "%.1f%%", value * 100)
    }

    private func coreMLUsageText(_ usage: [MLPolicyUsageSnapshot]) -> String {
        let successes = usage.reduce(0) { $0 + $1.successes }
        let attempts = usage.reduce(0) { $0 + $1.attempts }
        guard attempts > 0 else { return "fallback" }
        return "\(successes)/\(attempts)"
    }

    private func metric(_ title: String, _ value: String) -> some View {
        VStack(spacing: 2) {
            Text(title).font(.caption2.weight(.black)).foregroundStyle(.secondary)
            Text(value).font(.headline.weight(.black))
        }
        .frame(width: 54)
    }

    private func playerSeatLabel(_ player: PlayerState) -> String {
        if player.seatKind.isHuman {
            return player.id == 0 ? "You" : "Human"
        }
        return player.seatKind.label
    }

    private var logPanel: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Recent moves").font(.caption.weight(.black)).foregroundStyle(.yellow)
            ForEach(store.game.log.reversed().prefix(12)) { entry in
                Text(entry.text)
                    .font(.caption.weight(.semibold))
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(8)
                    .background(Color.white.opacity(0.06), in: RoundedRectangle(cornerRadius: 7))
            }
        }
        .panelStyle()
    }

    private func moveColor(_ move: Move) -> Color {
        switch move {
        case .play(_, let suit): suitColor(suit)
        case .paradox: Color(red: 0.45, green: 0.13, blue: 0.16)
        default: Color(red: 0.08, green: 0.16, blue: 0.13)
        }
    }

    private func moveTokenColor(_ move: Move, player: Int) -> Color {
        switch move {
        case .play(_, let suit):
            suitColor(suit)
        case .paradox:
            Color(red: 0.45, green: 0.13, blue: 0.16)
        default:
            playerColor(player)
        }
    }

    private func suitColor(_ suit: Suit) -> Color {
        switch suit {
        case .red: Color(red: 0.74, green: 0.20, blue: 0.17)
        case .blue: Color(red: 0.15, green: 0.46, blue: 0.68)
        case .yellow: Color(red: 0.68, green: 0.49, blue: 0.07)
        case .green: Color(red: 0.24, green: 0.56, blue: 0.39)
        }
    }

    private var ledSuitHighlightColor: Color {
        Color(red: 0.32, green: 0.77, blue: 1.0)
    }

    private var currentTrickHighlightColor: Color {
        Color(red: 1.0, green: 0.78, blue: 0.18)
    }

    private func isLedSuit(_ suit: Suit) -> Bool {
        store.game.phase == .play && store.game.ledSuit == suit
    }

    private func ledSuitStroke(for suit: Suit, cornerRadius: CGFloat, lineWidth: CGFloat) -> some View {
        RoundedRectangle(cornerRadius: cornerRadius)
            .stroke(isLedSuit(suit) ? ledSuitHighlightColor.opacity(0.82) : Color.clear, lineWidth: lineWidth)
            .allowsHitTesting(false)
    }

    private func currentTrickStroke(isCurrentTrick: Bool, cornerRadius: CGFloat) -> some View {
        RoundedRectangle(cornerRadius: cornerRadius)
            .stroke(isCurrentTrick ? currentTrickHighlightColor : Color.clear, lineWidth: 3)
            .overlay(
                RoundedRectangle(cornerRadius: cornerRadius)
                    .stroke(isCurrentTrick ? Color.white.opacity(0.72) : Color.clear, lineWidth: 1)
                    .padding(2)
            )
            .allowsHitTesting(false)
    }

    private func ledSuitShadowColor(for suit: Suit) -> Color {
        isLedSuit(suit) ? ledSuitHighlightColor.opacity(0.38) : .clear
    }

    private func playerColor(_ player: Int) -> Color {
        switch player {
        case 0: Color(red: 0.10, green: 0.45, blue: 0.86)
        case 1: Color(red: 0.86, green: 0.25, blue: 0.43)
        case 2: Color(red: 0.53, green: 0.34, blue: 0.88)
        case 3: Color(red: 0.00, green: 0.58, blue: 0.53)
        default: Color(red: 0.88, green: 0.48, blue: 0.12)
        }
    }

    private func boardCellColor(owner: Int) -> Color {
        if owner == -2 { return Color.gray.opacity(0.45) }
        if owner >= 0 { return playerColor(owner).opacity(0.86) }
        return Color(red: 0.92, green: 0.86, blue: 0.75)
    }
}

private extension GamePhase {
    static let allCasesForDisplay: [GamePhase] = [.dealing, .discard, .prediction, .play, .scoring]
}

private extension View {
    func panelStyle(light: Bool = false) -> some View {
        padding(12)
            .background(
                RoundedRectangle(cornerRadius: 10)
                    .fill(light ? Color(red: 0.90, green: 0.84, blue: 0.72) : Color.black.opacity(0.28))
                    .overlay(RoundedRectangle(cornerRadius: 10).stroke(Color.yellow.opacity(0.25), lineWidth: 1))
            )
    }
}
