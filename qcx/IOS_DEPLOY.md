# Shipping a qcx champion to the iOS app

The iOS app (`QuantumCatIOS/`, bundle `com.canal.quantumcat`) runs the qcx C++
search natively. Its default bot for 3, 4 and 5 players is `BotKind.qcxChampion`,
shown as **Champion (qcx PIMC search)**. Two-player tables fall back to the
previous Run1052 ML bot, because the C++ engine only implements 3 to 5 players.

## One command

From the repository root, with the iPhone unlocked, awake, and on the same Wi-Fi:

```sh
bash qcx/sync_ios.sh --test --install
```

This command:

1. Copies `qcx/qc_core.h`, `qcx/qc_agents.h` and `qcx/qc_capi.cpp` byte for byte into
   `QuantumCatIOS/QuantumCatIOS/Native/`. Never edit those copies by hand.
   `bash qcx/sync_ios.sh --check` reports any drift.
2. Writes `Native/qcx_champion.json` from `qcx/qcx_champion.json`, stamped with the
   source commit and the SHA-256 of the three C++ files. This file holds the champion's
   parameters (`worlds`, `util`, `view_len`, `players`, plus any future fields).
3. Builds `qcx/libqc.dylib` and regenerates `Native/qcx_parity_fixtures.json` with
   `qcx/export_ios_fixtures.py`. The fixtures cover 72 decisions from real Python rounds
   with 3, 4 and 5 players, across the discard, bid and play phases. Each fixture records
   the expected `qc_py.decide` action for a fixed seed.
4. With `--test`, runs the `QuantumCatIOSTests` unit tests on the "iPhone 17 Pro Max"
   simulator. Override the destination with `SIM_DEST=...`.
5. With `--install`, runs `QuantumCatIOS/Tools/build_device.sh` (a signed Debug build;
   the C++ is compiled at `-O3` in every configuration). It then runs
   `QuantumCatIOS/Tools/install_on_iphone.sh`, which uses `xcrun devicectl` to install
   and launch over Wi-Fi. A locked or unreachable phone gets 6 attempts, 30 s apart.
   `IPHONE_DEVICE_ID` overrides device selection. For example, the iPhone 17 Pro Max is
   `7F494DB0-B84F-5ECE-8B15-52C8F081AEC8`.

Fixtures need a Python with `pyspiel`. By default the script uses the main checkout's
`.venv/bin/python`; set `QCX_PYTHON` to use a different one. Use `--no-fixtures` only
when neither the engine nor the search changed. Any search change changes the
deterministic expected actions, so stale fixtures fail the parity test.

## Changing champion parameters

Edit `qcx/qcx_champion.json`, for example `"worlds": 128`, then run the command above.
Swift reads the bundled JSON when the app launches. Unknown keys are ignored, so a newer
JSON with extra fields still loads.

## When the C ABI or the view changes

- `Native/qcx_capi.h` is app-owned and declares `qc_decide` and `qc_legal_from_view`.
  It is force-included when `qc_capi.cpp` is compiled (`-include qcx_capi.h`), so a
  signature change is a compile error. The sync script also checks that the signatures
  are present.
- If `VIEW_LEN` in `qc_py.py` changes (for example, match totals are added to the view),
  the sync script stops. Update all four of these, then re-sync:
  - `QCX_VIEW_LEN` in `Native/qcx_capi.h`
  - `view_len` in `qcx/qcx_champion.json`
  - `QCXView.build` in `QuantumCatIOS/QuantumCatIOS/Models/QCXChampion.swift`
  - `QCXView.isConsistent`, if the new fields affect hand-size accounting

  `QCXView.build` is the only Swift code that builds the view, and it mirrors
  `qc_py.build_view` field for field. The parity fixtures verify that the Swift-built
  view equals Python's.
- A new model file, such as a bid model, must be added to the Xcode target's resources
  and copied by the sync script, the same way as `qcx_champion.json`.

## Information safety

`QCXView.build` reads only the acting seat's own hand and own discard, plus public
state: bids, board, colour tokens, tricks, the current trick, the start player, leader,
led colour and trick number, and hand sizes derived from public counts. Before any C++
call, `QCXView.isConsistent` checks the hidden-card accounting.

The app lets the round's starting player discard first, while OpenSpiel discards in
seat order. Discards are face down, so the order reveals nothing. The view uses the
reference convention (`has_discarded = phase != discard || seat < me`), which is what
the C++ world sampler assumes. `QCXChampionTests.testDecisionIgnoresOpponentsHiddenCards`
re-deals opponents' hidden cards and checks that the decision does not change.

Each decision is seeded with a hash of the view, so it is deterministic per
information set. Searches run on `ChampionInference.queue`, never on the main thread.

## Verifying on the phone

```sh
xcrun devicectl -t 600 device process launch --device 7F494DB0-B84F-5ECE-8B15-52C8F081AEC8 \
  --terminate-existing --console com.canal.quantumcat -- -qcxBenchmark -qcxBenchmarkGames 2
```

Check the output for these lines:

- `QCX_PARITY ... passed=true`: the Python fixtures are reproduced on the device CPU.
- `QCX_BENCH players=N worlds=W ... mean= p90= max=`: per-decision latency for full
  bot-only rounds. Add `-qcxBenchmarkWorlds 64,128,256` to compare world budgets.
- `QCX_BENCHMARK ... failures=0 illegal=0`

Then launch the app normally, without arguments. To publish to the App Store, use
`QuantumCatIOS/Tools/archive_appstore.sh` separately. The sync script never uploads.
