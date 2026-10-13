import sys, ctypes
sys.path.insert(0, "/Users/canal/quantum_-cat_openspiel"); sys.path.insert(0, ".")
import qc_py
import quantum_cat_curated_bidding_validation as bv
v = bv.validate_suite(bv.load_suite())
for item in v:
  sc = item["scenario"]
  if sc["id"] not in ("5p-four-nines-requires-more-than-one", "5p-three-nines-requires-more-than-one"): continue
  st = item["state"]; me = int(sc["state"]["actingPlayer"])
  for util in (0, 1):
    view = qc_py.build_view(st, me)
    vals = (ctypes.c_double * 41)()
    a = qc_py._LIB.qc_decide(view.ctypes.data_as(ctypes.POINTER(ctypes.c_int)), 2048, util, 5, vals)
    print(sc["id"], "util", util, "chosen", a - 100, "values", [round(vals[i], 3) for i in range(4)])
