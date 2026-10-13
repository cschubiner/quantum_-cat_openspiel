# September 8, 2026 deployment receipt

Status: **installed, verified on the physical Wi-Fi iPhone, full Run1052 search
enabled, and app launched normally**. Verification completed after unlock.

- Repository branch: `codex/quantum-cat-ios-tap-bidding`.
- Installed app source commit: `b6ee89f6` (native search implementation was
  introduced in `31dda6ed`; frozen experiments in `c52b7833`).
- Device: paired Wi-Fi iPhone 17 Pro Max, CoreDevice identifier
  `7F494DB0-B84F-5ECE-8B15-52C8F081AEC8`.
- Bundle identifier: `com.canal.quantumcat`.
- Latest installation completed at 14:51 EDT. See `device/install.txt`.
- Signed executable SHA-256:
  `776860bf590bcc22bcfa84721093154e2d67e30e226d8ed805e2312a08b51992`.
- Search model float32 weight blob SHA-256:
  `4a8035fb0d39d85fc1e879288029d51f240bcde2879515916793c3cb49c21bbf`.
- Search fixture SHA-256:
  `cbb81b7b4c006cb441d7ad084c45f3679dce00f339ed03fb1ea5786a9fc19c60`.
- `codesign --verify --deep --strict` passed on the installed build artifact.
- Initial launches were blocked by the lock screen; the user unlocked the phone
  and all subsequent verification completed successfully at 15:00–15:01 EDT.

## Physical-device results

- Neural parity: 40 fixtures, zero move or native-input mismatches; all four
  model outputs passed. Maximum output difference was 0.00009155.
- Full-search parity: 441 continuations, 25 sampled worlds, ten root cases
  covering all five rounds; zero failures, including exact rollout/audit scores.
- Search activation persisted: `passed=true enabled=true`, followed by a new
  process exercising the actual gameplay route with `search_enabled=true`.
- Three five-player gameplay smoke rounds exercised 101 play decisions and
  12 expanded searches, with zero Core ML or search failures. Learned bidding
  remained enabled, and shared discard/bid checks passed.
- Maximum observed search latency was about 1.68 seconds in this small gameplay
  smoke; parity-case searches took about 0.06–1.40 seconds. These are observed
  test timings, not a worst-case guarantee or a new statistical strength result.
- The gameplay JSON was copied back from the phone's app container, not inferred
  from local logs. See `device/gameplay_smoke.json` and `device/gameplay_smoke.txt`.
- Normal foreground launch succeeded at 15:01 EDT, PID 3814, with no test
  arguments. See `device/normal_launch.json`.

The final iOS Simulator regression run passed all 27 tests, including the full
Run1052 search fixture suite. See `ios_tests.json`; separate physical-device
receipts are in `device/`.

## Repeat verification

Leave the iPhone awake, then run:

```sh
CHAMPION_SEARCH=1 INSTALL=0 \
  IPHONE_DEVICE_ID=7F494DB0-B84F-5ECE-8B15-52C8F081AEC8 \
  bash .codex/skills/test-model/scripts/run_ios_bridge_check.sh
```

Require `DEVICE_MODEL_MOVE_PARITY ... passed=true` and
`DEVICE_CHAMPION_1052_SEARCH_PARITY passed=true enabled=true`. Then exercise the
actual gameplay route (not just direct fixture inference):

```sh
xcrun devicectl -t 300 device process launch \
  --device 7F494DB0-B84F-5ECE-8B15-52C8F081AEC8 --terminate-existing --console \
  com.canal.quantumcat -- -deviceParadoxBenchmark \
  -deviceParadoxBenchmarkGames 3 -deviceParadoxBenchmarkSeats 5
```

Require zero Core ML/search failures, `search_enabled=true`, and at least one
expanded search decision. Finally launch the app normally, save the device
receipts, update this status, and commit/push those receipts. Preserve existing
user game data and the frozen Run1052 experiment evidence.
