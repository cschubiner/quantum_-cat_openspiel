# Run1052 native iPhone search port

Deployment complete: physical-iPhone neural and full-search parity passed, actual
gameplay exercised 12 expanded searches without failures, and the app is open
with full search enabled. See `DEPLOYMENT.md` and `device/` for receipts.

The full checkpoint is `../run1052/champion.pt` (SHA-256
`b9c1d5fd3ce60ea528cd7955a4e07868cd75eb203593c4cee8e0bb6a1281be4d`).
The original frozen Python source, selector, and confirmation receipts are unchanged.

The native implementation uses the same full float32 network, learned bids,
Q/risk/feasibility selection, public tactical protection, actor-information seeded
MT19937, uniform information-state sampling, full-match continuation, eight-world
selection, and fresh eight-world audit. It searches tricks 4–6 (zero-based) of
all five rounds. No actual opponent hand or face-down discard enters its API.

Two semantics-preserving execution changes are explicit:

- Unused model input padding is reduced from 1,000 to 36 action rows. All neural
  weights, architecture, and precision are retained; this is not distillation.
- A single surviving baseline candidate returns immediately. Search RNG is local
  and re-seeded at every decision; no downstream choice consumes skipped draws.

## Mac validation, September 8, 2026

- Existing Python game/neural suites: 299 tests passed.
- Native engine smoke: five supported human/bot configurations passed.
- Fresh Python fixtures use two complete five-player matches, independent of the
  protected Run1052 confirmation deals.
- 441 native continuation decisions and observations matched Python exactly.
- 1,300 MT19937 outputs, five future deals, and 25 hypothetical-world samples
  matched. All sampled-world safety filters and actor-only search seeds matched.
- Ten root-search cases cover all five rounds, including both accepted search
  overrides and single-candidate shortcuts. All selected actions and all
  multi-candidate selection/audit mean scores matched exactly.
- `mac_search_parity.log` records zero failures and per-case latency. These are
  Mac measurements, not iPhone measurements or new strength benchmarks.
- iOS Simulator regression suite: **27/27 passed**, including the full native
  search parity runner (`ios_tests.json`). The initial run had one outdated
  description-string assertion; it was updated for the new Run1052 description.

## Reproduce and install

Run `QuantumCatIOS/Tools/export_search_parity.py` with the repo virtualenv to
regenerate `search_parity.json`. Compile the Swift model sources plus
`Tools/search_parity.swift` with `swiftc -O`; pass the fixture and exported model
paths. The export command is:

```sh
PYTHONPATH=. .venv/bin/python QuantumCatIOS/Tools/export_coreml.py \
  --checkpoint research/run1052/champion.pt --players 5 --max-actions 36 \
  --named-outputs \
  --out QuantumCatIOS/QuantumCatIOS/Resources/Models/champion_1052_search.mlpackage
```

Build with `bash QuantumCatIOS/Tools/build_device.sh`, then:

```sh
CHAMPION_SEARCH=1 IPHONE_DEVICE_ID=<paired-device-id> \
  bash .codex/skills/test-model/scripts/run_ios_bridge_check.sh
```

The bridge checks checkpoint identity, verifies signing, installs the app, runs
neural-output/native-input parity and full-search parity on-device, then launches
normally. The native search is enabled only after the device parity run passes.
Until then the app retains its previous neural/tactical policy. Device receipts
and final deployment status belong in `device/` and `DEPLOYMENT.md`, respectively.

Bot thinking and hint generation run on a serial background queue. A failed
search logs `CHAMPION_1052_FAILURE` and uses the prior neural/tactical policy.
Preference scores remain the existing normalized neural/tactical preferences,
with the actual champion choice ranked first; they are not win probabilities or
estimates from an exhaustive search of every move.
