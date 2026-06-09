import SwiftUI

struct ContentView: View {
    @EnvironmentObject private var store: GameStore
    @State private var showingSetup = false

    var body: some View {
        NavigationStack {
            GameTableView()
                .toolbar {
                    ToolbarItem(placement: .topBarLeading) {
                        Button {
                            showingSetup = true
                        } label: {
                            Label("\(store.humanSeats)H \(store.botSeats)B", systemImage: "person.3.fill")
                        }
                        .accessibilityLabel("Table setup")
                        .accessibilityIdentifier("setup-button")
                    }
                    ToolbarItem(placement: .topBarTrailing) {
                        Button {
                            store.newGame()
                        } label: {
                            Label("New game", systemImage: "shuffle")
                        }
                        .disabled(!store.setupIsValid)
                        .accessibilityLabel("New game")
                        .accessibilityIdentifier("new-game-button")
                    }
                }
                .sheet(isPresented: $showingSetup) {
                    GameSetupView()
                        .environmentObject(store)
                        .presentationDetents([.medium, .large])
                        .presentationDragIndicator(.visible)
                }
        }
        .preferredColorScheme(.dark)
    }
}
