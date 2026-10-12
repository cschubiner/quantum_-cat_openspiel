"""Run the locked curated suites (tactical, bid-regime, bidding) with a qcx bot.
usage: tactical_check.py <qcx_dir> <worlds> <util> [seeds]
Passing requires the accepted action for every seed (decisions are stochastic)."""
import importlib.util, json, sys
REPO = "/Users/canal/quantum_-cat_openspiel"
sys.path.insert(0, REPO)
qdir, worlds, util = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
seeds = int(sys.argv[4]) if len(sys.argv) > 4 else 3
learned = len(sys.argv) > 5 and sys.argv[5] == "learned"
spec = importlib.util.spec_from_file_location("qc_py", f"{qdir}/qc_py.py")
qc_py = importlib.util.module_from_spec(spec); spec.loader.exec_module(qc_py)
if learned:
  qc_py.load_policy(f"{qdir}/weights/g1_mlp.txt")
import quantum_cat_curated_validation as tv
import quantum_cat_curated_bidding_validation as bv

total = passed = 0
for mod, path in [(tv, f"{REPO}/experiments/quantum_cat_curated_tactical_validation.json"),
                  (tv, f"{REPO}/experiments/quantum_cat_curated_bid_regime_validation.json"),
                  (bv, f"{REPO}/experiments/quantum_cat_curated_bidding_validation.json")]:
  validated = mod.validate_suite(mod.load_suite(path))
  for seed in range(seeds):
    fn = qc_py.decide_learned if learned else qc_py.decide
    sel = lambda state, actor: fn(state, actor, worlds, util, 1000 + seed)
    rep = mod.evaluate_validated(validated, sel, "qcx")
    for r in rep["results"]:
      total += 1; passed += bool(r["passed"])
      if not r["passed"]:
        print("FAIL", path.split("/")[-1], r["id"], "rot", r["seatRotation"], "seed", seed,
              "got", r.get("actualMove", r.get("actualBid")),
              "accepted", r.get("acceptedMoves", r.get("acceptedBids")))
print(f"passed {passed}/{total}")
