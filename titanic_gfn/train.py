#!/usr/bin/env python3
"""GFlowNet (trajectory balance, U-Net policy from gfn/) over Titanic feature hypotheses.

Rewards are CV accuracies evaluated once per unique hypothesis (exact cache); evaluations of
a batch's new hypotheses run in a small process pool. The run is seeded end to end.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch
import torch.nn.functional as F

from titanic.determinism import reexec_with_fixed_hashseed, seed_everything, sha256_file

if __name__ == "__main__":  # re-exec only as the program, never when imported (compare.py imports this)
    reexec_with_fixed_hashseed()

from gfn.unet import UNetPolicy  # noqa: E402
from titanic import data  # noqa: E402
from .env import CanvasEnv  # noqa: E402
from .reward import DEFAULTS, RewardEvaluator  # noqa: E402

_EVAL = None


def _init_worker(seed):
    global _EVAL
    X, y, _ = data.load()
    _EVAL = RewardEvaluator(X, y, seed=seed)


def _eval_worker(args):
    cells, hp = args
    return _EVAL.cv_accuracy(list(cells), dict(hp))


class BatchedReward:
    """Main-process cache + process pool for the misses. Deterministic: each evaluation is an
    independent seeded CV; the cache is keyed by hypothesis, so completion order is irrelevant."""

    def __init__(self, seed: int, workers: int, beta: float, base: float):
        self.seed, self.beta, self.base = seed, beta, base
        X, y, _ = data.load()
        self.local = RewardEvaluator(X, y, seed=seed, beta=beta, base=base)
        self.pool = ProcessPoolExecutor(max_workers=workers, initializer=_init_worker, initargs=(seed,)) if workers > 1 else None
        self.evals = 0
        self.eval_seconds = 0.0

    def accuracies(self, hyps: list[tuple]) -> list[float]:
        keys = [RewardEvaluator.key(c, {**DEFAULTS, **h}) for c, h in hyps]
        todo = sorted({k for k in keys if k not in self.local.cache})
        if todo:
            t = time.time()
            if self.pool is not None:
                accs = list(self.pool.map(_eval_worker, [(k[0], k[1]) for k in todo], chunksize=1))
            else:
                accs = [self.local.cv_accuracy(list(k[0]), dict(k[1])) for k in todo]
            for k, a in zip(todo, accs):
                self.local.cache[k] = a
            self.evals += len(todo)
            self.eval_seconds += time.time() - t
        return [self.local.cache[k] for k in keys]

    def log_rewards(self, hyps) -> torch.Tensor:
        return torch.tensor([self.beta * (a - self.base) for a in self.accuracies(hyps)])

    def close(self):
        if self.pool is not None:
            self.pool.shutdown()


def masked_log_softmax(logits, mask):
    return F.log_softmax(logits.masked_fill(~mask, float("-inf")), dim=-1)


def rollout(model, env: CanvasEnv, batch: int, eps: float, gen: torch.Generator):
    canvas = env.initial_state(batch)
    done = torch.zeros(batch, dtype=torch.bool)
    log_pf = torch.zeros(batch)
    for _ in range(env.max_steps + 1):
        mask = env.action_mask(canvas)
        logp = masked_log_softmax(model(canvas), mask)
        with torch.no_grad():
            probs = logp.exp()
            if eps > 0:
                probs = (1 - eps) * probs + eps * mask.float() / mask.float().sum(-1, keepdim=True)
            a = torch.multinomial(probs, 1, generator=gen).squeeze(1)
        log_pf = log_pf + torch.where(done, torch.zeros(batch), logp.gather(1, a[:, None]).squeeze(1))
        a = torch.where(done, torch.full_like(a, env.STOP), a)
        canvas = env.step(canvas, a)
        done = done | (a == env.STOP)
        if bool(done.all()):
            break
    return canvas, log_pf


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--iters", type=int, default=300)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--eps", type=float, default=0.1)
    p.add_argument("--beta", type=float, default=60.0)
    p.add_argument("--base", type=float, default=0.78)
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--samples", type=int, default=512, help="on-policy samples drawn after training")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="out/titanic_gfn")
    args = p.parse_args()

    seed_everything(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    gen = torch.Generator().manual_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)

    env = CanvasEnv()
    model = UNetPolicy(base=32)
    log_z = torch.nn.Parameter(torch.tensor(25.0))
    opt = torch.optim.Adam([{"params": model.parameters(), "lr": args.lr}, {"params": [log_z], "lr": 0.1}])
    R = BatchedReward(args.seed, args.workers, args.beta, args.base)

    hist, seen = [], {}
    t0 = time.time()
    for it in range(1, args.iters + 1):
        canvas, log_pf = rollout(model, env, args.batch, args.eps, gen)
        hyps = [env.decode(c) for c in canvas]
        log_r = R.log_rewards(hyps)
        n = canvas.reshape(args.batch, -1).sum(1)
        loss = (log_z + log_pf - log_r - env.log_pb_sum(n)).pow(2).mean()
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        accs = (log_r / args.beta + args.base)
        for c, a in zip(canvas, accs.tolist()):
            seen[env.canvas_key(c)] = a
        hist.append({"it": it, "loss": round(loss.item(), 4), "log_z": round(log_z.item(), 3),
                     "acc_mean": round(accs.mean().item(), 4), "acc_max": round(accs.max().item(), 4),
                     "n_mean": round(n.float().mean().item(), 2), "evals": R.evals, "t": round(time.time() - t0, 1)})
        if it % 10 == 0 or it == 1:
            h = hist[-1]
            print(f"it {it:4d} | loss {h['loss']:8.3f} | logZ {h['log_z']:7.3f} | acc mean {h['acc_mean']:.4f} max {h['acc_max']:.4f} "
                  f"| painted {h['n_mean']:5.2f} | evals {h['evals']:5d} ({R.eval_seconds:5.0f}s) | {h['t']:6.0f}s", flush=True)

    # on-policy samples from the trained generator
    model.eval()
    with torch.no_grad():
        canvases = []
        for _ in range(math.ceil(args.samples / args.batch)):
            c, _ = rollout(model, env, args.batch, 0.0, gen)
            canvases.append(c)
        canvas = torch.cat(canvases)[:args.samples]
    hyps = [env.decode(c) for c in canvas]
    accs = R.accuracies(hyps)
    keys = [env.canvas_key(c) for c in canvas]
    from collections import Counter
    freq = Counter(keys)
    uniq = {k: (h, a) for k, h, a in zip(keys, hyps, accs)}
    top = sorted(uniq.items(), key=lambda kv: -kv[1][1])[:10]
    sample_rows = [{"key": k, "freq": freq[k], "acc": a, "cells": [f"{c}:{o}" for c, o in h[0]], "hp": h[1]}
                   for k, (h, a) in sorted(uniq.items(), key=lambda kv: -kv[1][1])]
    R.close()

    total = time.time() - t0
    best_key, (best_hyp, best_acc) = top[0]
    results = {
        "seed": args.seed, "iters": args.iters, "batch": args.batch, "beta": args.beta, "base": args.base,
        "unique_evaluations": R.evals, "eval_seconds": round(R.eval_seconds, 1), "total_seconds": round(total, 1),
        "log_z": round(log_z.item(), 3),
        "train_best_acc": max(seen.values()), "train_unique_hypotheses": len(seen),
        "train_hyps_above_083": sum(a >= 0.83 for a in seen.values()),
        "samples": args.samples, "sample_unique": len(uniq), "sample_mean_acc": round(float(np.mean(accs)), 4),
        "sample_hyps_above_083": sum(a >= 0.83 for _, a in uniq.values()),
        "best_sampled": {"acc": best_acc, "cells": [f"{c}:{o}" for c, o in best_hyp[0]], "hp": best_hyp[1], "freq": freq[best_key]},
        "top10_sampled": [{"acc": a, "freq": freq[k], "cells": [f"{c}:{o}" for c, o in h[0]], "hp": h[1]} for k, (h, a) in top],
    }
    json.dump(results, open(os.path.join(args.out, "results.json"), "w"), indent=2)
    json.dump(hist, open(os.path.join(args.out, "history.json"), "w"))
    json.dump(sample_rows, open(os.path.join(args.out, "samples.json"), "w"))
    torch.save({"model": model.state_dict(), "log_z": log_z.detach()}, os.path.join(args.out, "model.pt"))
    cache_rows = [{"cells": list(k[0]), "hp": dict(k[1]), "acc": a} for k, a in sorted(R.local.cache.items(), key=lambda kv: -kv[1])]
    json.dump(cache_rows, open(os.path.join(args.out, "evaluations.json"), "w"))
    print(f"\nunique evals {R.evals} ({R.eval_seconds:.0f}s eval, {total:.0f}s total) | train best acc {results['train_best_acc']:.4f} "
          f"| sampled {args.samples}: unique {len(uniq)}, mean acc {results['sample_mean_acc']:.4f}, best {best_acc:.4f} (freq {freq[best_key]})")
    for r in results["top10_sampled"][:5]:
        print(f"  {r['acc']:.4f} x{r['freq']:3d} {r['hp']} {r['cells']}")


if __name__ == "__main__":
    main()
