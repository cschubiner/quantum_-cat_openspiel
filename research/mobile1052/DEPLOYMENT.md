# September 8, 2026 deployment receipt

Status: **signed app installed; physical-device verification and search activation
blocked by the iPhone lock screen**. Do not describe the full champion as active yet.

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
- CoreDevice confirmed installation, then rejected the launch with
  `FBSOpenApplicationErrorDomain code 7: Locked`; see `device/neural_parity.txt`.

The app's previous neural/tactical policy remains the fallback until the new
search parity receipt is successfully written on this physical device. No
on-device search latency or gameplay claim has been made.

The final iOS Simulator regression run passed all 27 tests, including the full
Run1052 search fixture suite. See `ios_tests.json`; this does not substitute for
the blocked physical-device verification.

## Resume after unlock

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
