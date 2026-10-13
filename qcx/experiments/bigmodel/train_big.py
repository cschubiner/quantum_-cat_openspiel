"""Train the "big" PLAY policy (state encoder + per-action scorer) from qc_gen data
written with the optional state sidecar (qc_gen ... out.bin 0 out.state.bin).

usage: python train_big.py --data results/bm_p5 results/bm_p4 ... --out weights/big_x.txt
         [--h1 256 --h1b 128 --h2 64 --h3 32] [--tau 0.2] [--epochs 30] [--vw 0.0]
         [--small]   (train the 64-feature MLP-32 baseline on the same split instead)

Each --data prefix P means P.bin (rows: game, dec, q, heur, phi[64]) and
P.state.bin (per decision: game, dec, P, outcome, n, actions[37], state[431]).
Validation = 10% of games (seeded, per file), identical for every model, so the
reported numbers are directly comparable; g3 (weights/g3_mlp32.txt) and the
hand-written heuristic are scored on exactly the same held-out decisions.
"""
import argparse
import json
import time

import numpy as np
import torch

torch.set_num_threads(6)
NSTATE, NFEAT, NEXTRA = 431, 64, 13
NACT = NFEAT + NEXTRA
RANKS = {3: 6, 4: 8, 5: 9}


def load(prefixes):
    S, OUT, PL, X, Q, H, NA = [], [], [], [], [], [], []
    GAME = []
    for fi, pre in enumerate(prefixes):
        hdr = np.fromfile(pre + ".bin", dtype=np.int32, count=3)
        rows = np.fromfile(pre + ".bin", dtype=np.float32, offset=12).reshape(-1, int(hdr[2]))
        sh = np.fromfile(pre + ".state.bin", dtype=np.int32, count=3)
        assert sh[0] == 0x51435853 and sh[1] == NSTATE, pre
        st = np.fromfile(pre + ".state.bin", dtype=np.float32, offset=12).reshape(-1, int(sh[2]))
        # rows of one decision are contiguous; order decisions by (game, dec) in both files
        rkey = rows[:, 0].astype(np.int64) * 1000 + rows[:, 1].astype(np.int64)
        o = np.argsort(rkey, kind="stable")
        rows, rkey = rows[o], rkey[o]
        skey = st[:, 0].astype(np.int64) * 1000 + st[:, 1].astype(np.int64)
        o = np.argsort(skey, kind="stable")
        st, skey = st[o], skey[o]
        n = st[:, 4].astype(np.int64)
        assert n.sum() == len(rows), (pre, n.sum(), len(rows))
        assert np.array_equal(np.repeat(skey, n), rkey), pre
        P = st[:, 2].astype(np.int64)
        R = np.vectorize(RANKS.get)(P)
        acts = st[:, 5:5 + 37]
        # actions in row order
        pos = np.arange(len(rows)) - np.repeat(np.cumsum(n) - n, n)
        a = acts[np.repeat(np.arange(len(st)), n), pos].astype(np.int64)
        Rr = np.repeat(R, n)
        ext = np.zeros((len(rows), NEXTRA), np.float32)
        ext[np.arange(len(rows)), a // Rr] = 1
        ext[np.arange(len(rows)), 4 + a % Rr] = 1
        S.append(st[:, 42:]); OUT.append(st[:, 3]); PL.append(P); NA.append(n)
        X.append(np.concatenate([rows[:, 4:], ext], 1)); Q.append(rows[:, 2]); H.append(rows[:, 3])
        GAME.append(st[:, 0].astype(np.int64) + fi * 10_000_000)
    return (np.concatenate(S), np.concatenate(OUT), np.concatenate(PL), np.concatenate(NA),
            np.concatenate(X), np.concatenate(Q), np.concatenate(H), np.concatenate(GAME))


class Split:
    def __init__(self, S, out, P, n, X, q, h, dmask):
        start = np.cumsum(n) - n
        rmask = np.repeat(dmask, n)
        self.S = torch.from_numpy(np.ascontiguousarray(S[dmask]))
        self.out = torch.from_numpy(out[dmask].astype(np.float32))
        self.P = P[dmask]
        self.n = n[dmask]
        self.X = torch.from_numpy(np.ascontiguousarray(X[rmask]))
        self.q = torch.from_numpy(q[rmask].astype(np.float32))
        self.h = torch.from_numpy(h[rmask].astype(np.float32))
        self.D = int(dmask.sum())
        self.start = np.cumsum(self.n) - self.n
        self.seg = torch.from_numpy(np.repeat(np.arange(self.D), self.n))

    def batch(self, idx):
        n = self.n[idx]
        rows = np.repeat(self.start[idx], n) + (np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n))
        seg = np.repeat(np.arange(len(idx)), n)
        return (self.S[idx], self.X[rows], self.q[rows], torch.from_numpy(seg), len(idx), self.out[idx])


def seg_max(v, seg, D):
    return torch.full((D,), -1e30, dtype=v.dtype).scatter_reduce(0, seg, v, reduce="amax", include_self=True)


def seg_logsoftmax(z, seg, D):
    m = seg_max(z, seg, D)
    e = torch.exp(z - m[seg])
    s = torch.zeros(D, dtype=z.dtype).index_add_(0, seg, e)
    return z - m[seg] - torch.log(s[seg])


def metrics(scores, q, seg, D):
    g = torch.Generator().manual_seed(0)
    s = scores + 1e-7 * torch.rand(scores.shape, generator=g)
    chosen = s >= seg_max(s, seg, D)[seg]
    qm = seg_max(q, seg, D)
    qc = torch.zeros(D).index_add_(0, seg, torch.where(chosen, q, torch.zeros_like(q)))
    agree = (qc >= qm - 1e-6).float()
    return agree, qm - qc


class Big(torch.nn.Module):
    def __init__(self, h1, h1b, h2, h3, small=False, hs=32, drop=0.0):
        super().__init__()
        self.small = small
        self.drop = torch.nn.Dropout(drop)
        if small:
            self.l1 = torch.nn.Linear(NFEAT, hs)
            self.l2 = torch.nn.Linear(hs, 1, bias=False)
            return
        self.s1 = torch.nn.Linear(NSTATE, h1)
        self.s2 = torch.nn.Linear(h1, h1b) if h1b > 0 else None
        hs_ = h1b if h1b > 0 else h1
        self.sp = torch.nn.Linear(hs_, h2)
        self.a1 = torch.nn.Linear(NACT, h2)
        self.a3 = torch.nn.Linear(h2, h3) if h3 > 0 else None
        self.out = torch.nn.Linear(h3 if h3 > 0 else h2, 1, bias=False)
        self.val = torch.nn.Linear(hs_, 1)

    def forward(self, S, X, seg):
        if self.small:
            return self.l2(torch.relu(self.l1(X[:, :NFEAT]))).squeeze(-1), None
        h = self.drop(torch.relu(self.s1(S)))
        if self.s2 is not None:
            h = self.drop(torch.relu(self.s2(h)))
        u = self.sp(h)
        z = torch.relu(u[seg] + self.a1(X))
        if self.a3 is not None:
            z = torch.relu(self.a3(self.drop(z)))
        return self.out(z).squeeze(-1), self.val(h).squeeze(-1)


def run_eval(model, sp, musd, bs=4096):
    mu_s, sd_s, mu_a, sd_a = musd
    scores = []
    with torch.no_grad():
        for b0 in range(0, sp.D, bs):
            idx = np.arange(b0, min(sp.D, b0 + bs))
            S, X, q, seg, D, out = sp.batch(idx)
            sc, _ = model((S - mu_s) / sd_s, (X - mu_a) / sd_a, seg)
            scores.append(sc)
    return torch.cat(scores)


def export(model, musd, a, path):
    mu_s, sd_s, mu_a, sd_a = musd
    with torch.no_grad():
        W1 = model.s1.weight / sd_s                       # H1 x S
        b1 = model.s1.bias - (model.s1.weight * (mu_s / sd_s)).sum(1)
        Wa = model.a1.weight / sd_a                       # H2 x A
        bp = model.sp.bias + model.a1.bias - (model.a1.weight * (mu_a / sd_a)).sum(1)
        f = open(path, "w")
        wr = lambda t: f.write(" ".join(f"{x:.8g}" for x in t.reshape(-1).tolist()) + "\n")
        f.write(f"big {NSTATE} {NACT} {a.h1} {a.h1b} {a.h2} {a.h3}\n")
        wr(W1.t().contiguous()); wr(b1)
        if a.h1b > 0:
            wr(model.s2.weight); wr(model.s2.bias)
        wr(model.sp.weight); wr(bp)
        wr(Wa.t().contiguous())
        if a.h3 > 0:
            wr(model.a3.weight); wr(model.a3.bias)
        wr(model.out.weight[0])
        f.close()


def score_exported_big(path, S, X, seg):
    tok = open(path).read().split()
    _, Sd, A, h1, h1b, h2, h3 = tok[:7]
    Sd, A, h1, h1b, h2, h3 = map(int, (Sd, A, h1, h1b, h2, h3))
    v = np.array(tok[7:], dtype=np.float32)
    o = 0
    def take(k, shape):
        nonlocal o
        r = v[o:o + k].reshape(shape); o += k
        return r
    W1T = take(Sd * h1, (Sd, h1)); b1 = take(h1, (h1,))
    h = np.maximum(S @ W1T + b1, 0)
    if h1b:
        W2 = take(h1b * h1, (h1b, h1)); b2 = take(h1b, (h1b,)); h = np.maximum(h @ W2.T + b2, 0)
    hs = h.shape[1]
    Wp = take(h2 * hs, (h2, hs)); bp = take(h2, (h2,))
    u = h @ Wp.T + bp
    WaT = take(A * h2, (A, h2))
    z = np.maximum(u[seg] + X @ WaT, 0)
    if h3:
        W3 = take(h3 * h2, (h3, h2)); b3 = take(h3, (h3,)); z = np.maximum(z @ W3.T + b3, 0)
    w4 = take(z.shape[1], (z.shape[1],))
    assert o == len(v)
    return z @ w4


def score_g3(path, X):
    tok = open(path).read().split()
    F, H = int(tok[1]), int(tok[2])
    v = np.array(tok[3:], dtype=np.float32)
    W1 = v[:H * F].reshape(H, F); b1 = v[H * F:H * F + H]; w2 = v[H * F + H:H * F + 2 * H]
    return np.maximum(X[:, :F] @ W1.T + b1, 0) @ w2


def report(name, scores, va):
    ag, rg = metrics(scores, va.q, va.seg, va.D)
    res = {"agree": ag.mean().item(), "regret": rg.mean().item()}
    for P in (3, 4, 5):
        m = torch.from_numpy(va.P == P)
        if m.any():
            res[f"agree_p{P}"] = ag[m].mean().item(); res[f"regret_p{P}"] = rg[m].mean().item()
    print(f"{name:>12}: " + " ".join(f"{k} {v:.4f}" for k, v in res.items()), flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--h1", type=int, default=256)
    ap.add_argument("--h1b", type=int, default=128)
    ap.add_argument("--h2", type=int, default=64)
    ap.add_argument("--h3", type=int, default=32)
    ap.add_argument("--small", action="store_true")
    ap.add_argument("--tau", type=float, default=0.2)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--bs", type=int, default=512)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--drop", type=float, default=0.0)
    ap.add_argument("--vw", type=float, default=0.0, help="value-head loss weight (aux, not exported)")
    ap.add_argument("--g3", default="weights/g3_mlp32.txt")
    args = ap.parse_args()
    t0 = time.time()
    S, out, P, n, X, q, h, game = load(args.data)
    ug = np.unique(game)
    rng = np.random.default_rng(0)
    val_games = rng.choice(ug, size=max(1, len(ug) // 10), replace=False)
    dval = np.isin(game, val_games)
    tr = Split(S, out, P, n, X, q, h, ~dval)
    va = Split(S, out, P, n, X, q, h, dval)
    print(f"decisions {len(n)} rows {len(q)} games {len(ug)}; train {tr.D} val {va.D}; "
          f"by P {dict(zip(*np.unique(P, return_counts=True)))}; load {time.time()-t0:.0f}s", flush=True)
    base = {"heuristic": report("heuristic", va.h, va),
            "g3_mlp32": report("g3_mlp32", torch.from_numpy(score_g3(args.g3, va.X.numpy())), va)}
    mu_s = tr.S.mean(0); sd_s = tr.S.std(0); sd_s = torch.where(sd_s < 1e-2, torch.ones_like(sd_s), sd_s)
    mu_a = tr.X.mean(0); sd_a = tr.X.std(0); sd_a = torch.where(sd_a < 1e-2, torch.ones_like(sd_a), sd_a)
    musd = (mu_s, sd_s, mu_a, sd_a)
    torch.manual_seed(0)
    model = Big(args.h1, args.h1b, args.h2, args.h3, small=args.small, drop=args.drop)
    nparam = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    steps_per = (tr.D + args.bs - 1) // args.bs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * steps_per,
                                                pct_start=0.1)
    best = None
    hist = []
    for ep in range(args.epochs):
        model.train()
        perm = np.random.default_rng(ep).permutation(tr.D)
        tl = tv = 0.0
        for b0 in range(0, tr.D, args.bs):
            idx = perm[b0:b0 + args.bs]
            Sb, Xb, qb, seg, D, ob = tr.batch(idx)
            sc, v = model((Sb - mu_s) / sd_s, (Xb - mu_a) / sd_a, seg)
            t = torch.exp(seg_logsoftmax(qb / args.tau, seg, D))
            loss = -(t * seg_logsoftmax(sc, seg, D)).sum() / D
            tl += loss.item() * D
            if args.vw > 0 and v is not None:
                lv = ((v - ob) ** 2).mean()
                tv += lv.item() * D
                loss = loss + args.vw * lv
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        model.eval()
        ag, rg = metrics(run_eval(model, va, musd), va.q, va.seg, va.D)
        hist.append({"epoch": ep + 1, "loss": tl / tr.D, "vloss": tv / tr.D, "val_agree": ag.mean().item(),
                     "val_regret": rg.mean().item()})
        print(f"ep {ep+1} loss {tl/tr.D:.4f} vloss {tv/tr.D:.3f} val agree {ag.mean():.4f} regret {rg.mean():.4f} "
              f"({time.time()-t0:.0f}s)", flush=True)
        if best is None or rg.mean().item() < best[0]:
            best = (rg.mean().item(), {k: v.clone() for k, v in model.state_dict().items()}, ep + 1)
    model.load_state_dict(best[1])
    model.eval()
    # train-set metrics (subsample) for over/underfitting
    sub = np.random.default_rng(1).choice(tr.D, size=min(tr.D, 20000), replace=False)
    sub.sort()
    with torch.no_grad():
        Sb, Xb, qb, seg, D, ob = tr.batch(sub)
        sc, _ = model((Sb - mu_s) / sd_s, (Xb - mu_a) / sd_a, seg)
        tag, trg = metrics(sc, qb, seg, D)
    res = report("model", run_eval(model, va, musd), va)
    res.update({"train_agree": tag.mean().item(), "train_regret": trg.mean().item(), "best_epoch": best[2]})
    if not args.small:
        export(model, musd, args, args.out)
        xs = score_exported_big(args.out, va.S.numpy(), va.X.numpy(), va.seg.numpy())
        chk = report("exported", torch.from_numpy(xs), va)
        res["exported_agree"] = chk["agree"]
    else:
        with torch.no_grad():
            W1 = model.l1.weight / sd_a[:NFEAT]
            b1 = model.l1.bias - (model.l1.weight * (mu_a[:NFEAT] / sd_a[:NFEAT])).sum(1)
            with open(args.out, "w") as f:
                f.write(f"mlp {NFEAT} {W1.shape[0]}\n")
                for row in W1.numpy():
                    f.write(" ".join(f"{x:.8g}" for x in row) + "\n")
                f.write(" ".join(f"{x:.8g}" for x in b1.numpy()) + "\n")
                f.write(" ".join(f"{x:.8g}" for x in model.l2.weight[0].numpy()) + "\n")
        chk = report("exported", torch.from_numpy(score_g3(args.out, va.X.numpy())), va)
        res["exported_agree"] = chk["agree"]
    summary = {"args": vars(args), "params": nparam, "decisions": int(len(n)), "rows": int(len(q)),
               "val_decisions": va.D, "baselines": base, "model": res, "history": hist}
    print(json.dumps({k: summary[k] for k in ("params", "decisions", "val_decisions", "baselines", "model")}))
    with open(args.out + ".json", "w") as f:
        json.dump(summary, f, indent=1)


if __name__ == "__main__":
    main()
