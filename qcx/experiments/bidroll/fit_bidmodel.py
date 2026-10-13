"""Fit multinomial-logistic bid model; features must mirror bid_features() in qc_agents.h."""
import sys, numpy as np, torch
P = int(sys.argv[1]); R = {3: 6, 4: 8, 5: 9}[P]
d = np.loadtxt(sys.argv[2], delimiter=",", dtype=np.int64)
d = d[d[:, 0] == P]
pos = d[:, 1]; hands = d[:, 2:2 + R]; prev = d[:, 11:11 + 4]; y = d[:, -1] - 1
def feats(hand, pos, prev):
  n = len(hand)
  nprev = pos.astype(float); sumprev = prev.sum(1).astype(float)
  strength = (hand * ((np.arange(R) + 1.0) / R)).sum(1)
  cols = [np.ones(n)] + [hand[:, r] for r in range(R)] + [(hand[:, r] >= 2).astype(float) for r in range(R)]
  t1 = hand[:, R-1]; t2 = t1 + hand[:, R-2]; t3 = t2 + hand[:, R-3]
  cols += [t1, t2, t3, strength, strength ** 2 / 10.0]
  cols += [(pos == k).astype(float) for k in range(5)]
  cols += [np.where(nprev > 0, sumprev / np.maximum(nprev, 1), 0.0), sumprev]
  return np.stack(cols, 1)
X = torch.tensor(feats(hands, pos, prev), dtype=torch.float64); Y = torch.tensor(y)
rng = np.random.default_rng(0); idx = rng.permutation(len(Y)); nv = len(Y) // 10
va, tr = idx[:nv], idx[nv:]
W = torch.zeros(4, X.shape[1], dtype=torch.float64, requires_grad=True)
opt = torch.optim.LBFGS([W], max_iter=500, line_search_fn="strong_wolfe")
def loss_fn(ix):
  return torch.nn.functional.cross_entropy(X[ix] @ W.T, Y[ix]) + 1e-4 * (W[:, 1:] ** 2).sum()
def closure():
  opt.zero_grad(); l = loss_fn(tr); l.backward(); return l
opt.step(closure)
with torch.no_grad():
  lv = torch.nn.functional.cross_entropy(X[va] @ W.T, Y[va]).item()
  acc = ((X[va] @ W.T).argmax(1) == Y[va]).float().mean().item()
  base = torch.bincount(Y[tr], minlength=4).double() / len(tr)
  lb = -(torch.log(base[Y[va]])).mean().item()
print(f"rows={len(Y)} val_nll={lv:.4f} (marginal {lb:.4f}) val_acc={acc:.3f} bid_dist={base.numpy().round(3)}")
with open(sys.argv[3], "w") as f:
  f.write(f"{P} {X.shape[1]}\n")
  for row in W.detach().numpy():
    f.write(" ".join(f"{v:.8g}" for v in row) + "\n")
