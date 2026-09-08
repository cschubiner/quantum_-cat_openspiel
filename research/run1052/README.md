# Run1052 confirmed five-player champion

The frozen experiment demonstrated 57.11% pairwise placement against the prior
champion (50% baseline), on 64 independent held-out deals with five seat rotations.
See `RESULTS.md` and `CONFIRMED.json` for the protocol, uncertainty, and limitations.

`champion.pt` is the full, undistilled neural checkpoint, SHA-256
`b9c1d5fd3ce60ea528cd7955a4e07868cd75eb203593c4cee8e0bb6a1281be4d`.
The champion also requires the exact root-rollout selector in `CONFIRMED.json`.
Installing the neural checkpoint alone does not install this champion.

`evidence.tar.gz` contains the complete original Run1052 directory, excluding
Python bytecode. SHA-256:
`9b8436d9b3201c7f94b0e1470549278177574e1aa2288c63d53b435828c0ba0a`.
Extract with `tar -xzf evidence.tar.gz` into a scratch directory. Original receipts
retain their absolute paths and hashes deliberately; these are historical evidence,
not portable launch scripts. The original workspace was
`/Users/canal/Documents/Codex/2026-08-19/find-my-quantum-cat-ios-ml`.
The checkpoint came from `run1049/real_bid_raw_grounded/checkpoint_0000_loaded_replay_best.pt`.

Confirmation did not certify mobile performance or deployment. Subsequent mobile
validation must be recorded separately, without changing the frozen evidence.
