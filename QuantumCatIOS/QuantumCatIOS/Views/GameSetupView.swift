import SwiftUI

struct GameSetupView: View {
    @Environment(\.dismiss) private var dismiss
    @EnvironmentObject private var store: GameStore

    var body: some View {
        NavigationStack {
            List {
                Section {
                    Stepper(value: $store.humanSeats, in: 0...5) {
                        seatRow(title: "Humans", value: store.humanSeats, icon: "person.2.fill")
                    }
                    .accessibilityIdentifier("human-stepper")

                    Stepper(value: $store.botSeats, in: 0...5) {
                        seatRow(title: "Bots", value: store.botSeats, icon: "cpu.fill")
                    }
                    .accessibilityIdentifier("bot-stepper")

                    if !store.setupIsValid {
                        Text("Choose 2 to 5 total seats.")
                            .foregroundStyle(.yellow)
                            .font(.caption.weight(.semibold))
                    }
                } header: {
                    Text("Seats")
                } footer: {
                    Text(store.humanSeats == 0 ? "Bot-only tables can be watched live or simulated in bulk." : "Use pass-and-play for every human seat. Bot seats move automatically.")
                }

                if store.humanSeats == 0 {
                    Section {
                        Picker("Mode", selection: $store.botOnlyRunMode) {
                            ForEach(BotOnlyRunMode.allCases) { mode in
                                Text(mode.title).tag(mode)
                            }
                        }
                        .pickerStyle(.segmented)
                        .accessibilityIdentifier("bot-only-mode-picker")

                        if store.botOnlyRunMode == .bulk {
                            Stepper(value: $store.bulkSimulationGames, in: 1...10_000, step: 25) {
                                seatRow(title: "Games", value: store.bulkSimulationGames, icon: "number.square.fill")
                            }
                            .accessibilityIdentifier("bulk-games-stepper")
                        }
                    } header: {
                        Text("Bot-only run")
                    } footer: {
                        Text(store.botOnlyRunMode == .watch ? "Watch mode animates bot turns on the table." : "Simulate mode runs completed bot-only games immediately and reports aggregate results.")
                    }
                }

                if store.botSeats > 0 {
                    Section("Bot roster") {
                        ForEach(0..<store.botSeats, id: \.self) { index in
                            VStack(alignment: .leading, spacing: 8) {
                                Picker("Bot \(index + 1)", selection: $store.botKinds[index]) {
                                    ForEach(BotKind.allCases) { kind in
                                        Text(kind.name).tag(kind)
                                    }
                                }
                                .pickerStyle(.menu)
                                .accessibilityIdentifier("bot-picker-\(index)")

                                Text("Selected: \(store.botKinds[index].name)")
                                    .font(.caption.weight(.semibold))
                                    .foregroundStyle(.primary)
                                    .accessibilityIdentifier("bot-selection-\(index)")

                                Text(store.botKinds[index].mobileStatus)
                                    .font(.caption2)
                                    .foregroundStyle(.secondary)
                                    .fixedSize(horizontal: false, vertical: true)
                            }
                        }
                    }
                }

                Section("Model artifacts") {
                    ForEach(ModelArtifactStatus.bundledArtifacts) { artifact in
                        HStack(spacing: 12) {
                            Image(systemName: artifact.isPresent ? "checkmark.circle.fill" : "exclamationmark.triangle.fill")
                                .foregroundStyle(artifact.isPresent ? .green : .yellow)
                                .frame(width: 24)
                            VStack(alignment: .leading, spacing: 2) {
                                Text(artifact.title)
                                    .font(.subheadline.weight(.semibold))
                                Text(artifact.detail)
                                    .font(.caption)
                                    .foregroundStyle(.secondary)
                            }
                            Spacer()
                            Text(artifact.displayState)
                                .font(.caption.weight(.black))
                                .foregroundStyle(artifact.isPresent ? .green : .yellow)
                        }
                    }
                }
            }
            .navigationTitle("Table setup")
            .accessibilityIdentifier("setup-sheet")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Done") {
                        dismiss()
                    }
                    .accessibilityIdentifier("setup-done-button")
                }
                ToolbarItem(placement: .confirmationAction) {
                    Button {
                        store.newGame()
                        dismiss()
                    } label: {
                        Label("New game", systemImage: "shuffle")
                    }
                    .disabled(!store.setupIsValid)
                    .accessibilityIdentifier("setup-new-game-button")
                }
            }
        }
    }

    private func seatRow(title: String, value: Int, icon: String) -> some View {
        HStack(spacing: 12) {
            Image(systemName: icon)
                .font(.headline)
                .frame(width: 28, height: 28)
                .foregroundStyle(.yellow)
            Text(title)
                .font(.headline)
            Spacer()
            Text("\(value)")
                .font(.title3.weight(.black))
                .monospacedDigit()
        }
    }
}
