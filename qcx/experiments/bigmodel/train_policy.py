"""Train the learned PLAY-phase rollout policy from qc_gen data.

usage: python train_policy.py --data a.bin [b.bin ...] --out w.txt [--model linear|mlp]
       [--tau 0.2 | --tau 0 (argmax target)] [--sweep]

pi(a|s) = softmax(score(phi(s,a))); loss = CE(softmax(Q/tau) || pi) per decision.
10% of games (by game id) are held out; we report validation top-1 agreement with
PIMC's argmax (ties count as agreement), mean regret (Q_max - Q_chosen) and the
same numbers for the hand-written heuristic as a baseline.
"""
import argparse
import json
import sys

import numpy as np
import torch

torch.set_num_threads(4)
MAGIC = 0x51435847


def load(paths):
    Xs, qs, hs, keys = [], [], [], []
    for fi, p in enumerate(paths):
        raw = np.fromfile(p, dtype=np.int32, count=3)
        assert raw[0] == MAGIC, p
        F, ncols = int(raw[1]), int(raw[2])
        a = np.fromfile(p, dtype=np.float32, offset=12).reshape(-1, ncols)
        gid = a[:, 0].astype(np.int64) + fi * 10_000_000
        keys.append(gid * 100 + a[:, 1].astype(np.int64))
        qs.append(a[:, 2]); hs.append(a[:, 3]); Xs.append(a[:, 4:])
    X = np.concatenate(Xs); q = np.concatenate(qs); h = np.concatenate(hs); key = np.concatenate(keys)
    # rows of a decision are contiguous; segment ids from key changes
    seg = np.concatenate([[0], np.cumsum(key[1:] != key[:-1])])
    game = key // 100
    return X, q, h, seg, game


class Data:
    def __init__(self, X, q, h, seg):
        _, seg = np.unique(seg, return_inverse=True)
        self.X = torch.from_numpy(np.ascontiguousarray(X))
        self.q = torch.from_numpy(q.astype(np.float32))
        self.h = torch.from_numpy(h.astype(np.float32))
        self.seg = torch.from_numpy(seg.astype(np.int64))
        self.D = int(seg.max()) + 1


def seg_max(v, seg, D):
    return torch.full((D,), -1e30, dtype=v.dtype).scatter_reduce(0, seg, v, reduce="amax", include_self=True)


def seg_logsoftmax(z, seg, D):
    m = seg_max(z, seg, D)
    e = torch.exp(z - m[seg])
    s = torch.zeros(D, dtype=z.dtype).index_add_(0, seg, e)
    return z - m[seg] - torch.log(s[seg])


def targets(d, tau):
    if tau > 0:
        return torch.exp(seg_logsoftmax(d.q / tau, d.seg, d.D))
    qm = seg_max(d.q, d.seg, d.D)
    best = (d.q >= qm[d.seg] - 1e-6).float()
    cnt = torch.zeros(d.D).index_add_(0, d.seg, best)
    return best / cnt[d.seg]


def evaluate(scores, d):
    """top-1 agreement with PIMC argmax (ties ok) and mean regret in points."""
    g = torch.Generator().manual_seed(0)
    s = scores + 1e-7 * torch.rand(scores.shape, generator=g)
    sm = seg_max(s, d.seg, d.D)
    chosen = s >= sm[d.seg]
    qm = seg_max(d.q, d.seg, d.D)
    qc = torch.zeros(d.D).index_add_(0, d.seg, torch.where(chosen, d.q, torch.zeros_like(d.q)))
    agree = (qc >= qm - 1e-6).float().mean().item()
    regret = (qm - qc).mean().item()
    return agree, regret


class Model(torch.nn.Module):
    def __init__(self, F, kind, H):
        super().__init__()
        self.kind = kind
        if kind == "linear":
            self.lin = torch.nn.Linear(F, 1, bias=False)
            torch.nn.init.zeros_(self.lin.weight)
        else:
            self.l1 = torch.nn.Linear(F, H)
            self.l2 = torch.nn.Linear(H, 1, bias=False)

    def forward(self, x):
        if self.kind == "linear":
            return self.lin(x).squeeze(-1)
        return self.l2(torch.relu(self.l1(x))).squeeze(-1)


def train(tr, va, F, kind, H, tau, steps, lr, wd, mu, sd, verbose=False):
    torch.manual_seed(0)
    model = Model(F, kind, H)
    Xtr = (tr.X - mu) / sd
    Xva = (va.X - mu) / sd
    t = targets(tr, tau)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    best = (-1, None, None)
    for it in range(steps):
        opt.zero_grad()
        lp = seg_logsoftmax(model(Xtr), tr.seg, tr.D)
        loss = -(t * lp).sum() / tr.D
        loss.backward()
        opt.step()
        if (it + 1) % 50 == 0 or it == steps - 1:
            with torch.no_grad():
                ag, rg = evaluate(model(Xva), va)
            if verbose:
                print(f"  step {it+1} loss {loss.item():.4f} val agree {ag:.4f} regret {rg:.4f}", flush=True)
            if best[1] is None or rg < best[1]:
                best = (ag, rg, {k: v.clone() for k, v in model.state_dict().items()}, it + 1)
    model.load_state_dict(best[2])
    return model, best[0], best[1], best[3]


def export(model, kind, mu, sd, path):
    """Fold the standardisation into the weights so C++ consumes raw phi."""
    F = mu.numel()
    with torch.no_grad():
        if kind == "linear":
            w = (model.lin.weight[0] / sd).numpy()
            with open(path, "w") as f:
                f.write(f"linear {F}\n" + " ".join(f"{x:.8g}" for x in w) + "\n")
        else:
            W1 = (model.l1.weight / sd).numpy()
            b1 = (model.l1.bias - (model.l1.weight * (mu / sd)).sum(1)).numpy()
            w2 = model.l2.weight[0].numpy()
            H = W1.shape[0]
            with open(path, "w") as f:
                f.write(f"mlp {F} {H}\n")
                for row in W1:
                    f.write(" ".join(f"{x:.8g}" for x in row) + "\n")
                f.write(" ".join(f"{x:.8g}" for x in b1) + "\n")
                f.write(" ".join(f"{x:.8g}" for x in w2) + "\n")


def score_exported(path, X):
    """Score raw features exactly as the C++ LearnedPolicy does (float32)."""
    with open(path) as f:
        tok = f.read().split()
    kind, F = tok[0], int(tok[1])
    if kind == "linear":
        w = np.array(tok[2:2 + F], dtype=np.float32)
        return torch.from_numpy(X.numpy() @ w)
    H = int(tok[2])
    v = np.array(tok[3:], dtype=np.float32)
    W1 = v[:H * F].reshape(H, F); b1 = v[H * F:H * F + H]; w2 = v[H * F + H:H * F + 2 * H]
    return torch.from_numpy(np.maximum(X.numpy() @ W1.T + b1, 0) @ w2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="linear")
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--tau", type=float, default=0.2)
    ap.add_argument("--sweep", action="store_true", help="sweep tau and pick lowest val regret")
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--lr", type=float, default=0.03)
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--max_rows", type=int, default=0)
    args = ap.parse_args()

    X, q, h, seg, game = load(args.data)
    F = X.shape[1]
    ug = np.unique(game)
    rng = np.random.default_rng(0)
    val_games = set(rng.choice(ug, size=max(1, len(ug) // 10), replace=False).tolist())
    is_val = np.isin(game, list(val_games))
    tr = Data(X[~is_val], q[~is_val], h[~is_val], seg[~is_val])
    va = Data(X[is_val], q[is_val], h[is_val], seg[is_val])
    nact = np.bincount(np.unique(seg, return_inverse=True)[1])
    print(f"rows {len(q)} decisions {tr.D + va.D} games {len(ug)} (val games {len(val_games)}, val decisions {va.D}); "
          f"F={F}; mean legal {nact.mean():.2f}", flush=True)
    # baselines on validation
    ha, hr = evaluate(va.h, va)
    ra, rr = evaluate(torch.zeros_like(va.q), va)
    print(f"baseline heuristic: agree {ha:.4f} regret {hr:.4f};  random: agree {ra:.4f} regret {rr:.4f}", flush=True)
    mu = tr.X.mean(0)
    sd = tr.X.std(0).clamp_min(1e-3)
    lr = args.lr if args.model == "linear" else min(args.lr, 0.01)
    taus = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5] if args.sweep else [args.tau]
    results = []
    best = None
    for tau in taus:
        model, ag, rg, st = train(tr, va, F, args.model, args.hidden, tau, args.steps, lr, args.wd, mu, sd,
                                  verbose=not args.sweep)
        with torch.no_grad():
            tag, trg = evaluate(model((tr.X - mu) / sd), tr)
        print(f"tau {tau:.2f}: val agree {ag:.4f} regret {rg:.4f} (best step {st}); train agree {tag:.4f} regret {trg:.4f}",
              flush=True)
        results.append({"tau": tau, "val_agree": ag, "val_regret": rg, "train_agree": tag, "train_regret": trg})
        if best is None or rg < best[2]:
            best = (tau, model, rg, ag)
    export(best[1], args.model, mu, sd, args.out)
    torch.manual_seed(1)
    xa, xr = evaluate(score_exported(args.out, va.X), va)
    print(f"exported-weights check on raw phi: val agree {xa:.4f} regret {xr:.4f}", flush=True)
    summary = {"model": args.model, "tau": best[0], "val_agree": best[3], "val_regret": best[2],
               "heur_val_agree": ha, "heur_val_regret": hr, "rows": int(len(q)), "decisions": int(tr.D + va.D),
               "games": int(len(ug)), "sweep": results, "out": args.out}
    print(json.dumps(summary))
    with open(args.out + ".json", "w") as f:
        json.dump(summary, f, indent=1)


if __name__ == "__main__":
    main()
