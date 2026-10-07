#!/usr/bin/env python3
"""Train with checkpoints and dump everything needed for the interactive report.

Writes out/run.json and, if viz/report_template.html exists, out/report.html
(the template with the JSON inlined at the /*__RUN_JSON__*/ marker).
"""
import argparse
import json
import math
import os
import time

import torch

from gfn.env import GridEnv
from gfn.evaluate import exact_terminal_distribution, sample_metrics
from gfn.trainer import masked_log_softmax, rollout, train
from gfn.unet import UNetPolicy

CKPTS = [0, 25, 50, 100, 150, 200, 300, 400, 500, 700, 1000, 1500]
TRAJ_CKPTS = {0: 4, 100: 4, 500: 4, 1500: 8}


def r4(x):
    if isinstance(x, torch.Tensor):
        x = x.tolist()
    if isinstance(x, list):
        return [r4(v) for v in x]
    return round(float(x), 4)


def sig(x, digits: int = 4):
    """Round to `digits` significant figures (probabilities span many decades)."""
    if isinstance(x, torch.Tensor):
        x = x.tolist()
    if isinstance(x, list):
        return [sig(v, digits) for v in x]
    return float(f"{float(x):.{digits}g}")


def canvas_id(canvas: torch.Tensor) -> int:
    flat = canvas.reshape(-1).long()
    return int((flat << torch.arange(flat.numel())).sum())


def chan_maps(t: torch.Tensor, k: int):
    """t: (1, C, h, w) -> mean|act| map (h*w) and first k channel maps (k, h*w)."""
    t = t[0]
    return r4(t.abs().mean(0).reshape(-1)), (r4(t[:k].reshape(k, -1)) if k else [])


@torch.no_grad()
def record_trajectory(model, env, log_z):
    canvas = env.initial_state(1)
    steps, sum_logpf = [], 0.0
    for _ in range(env.H * env.W + 1):
        mask = env.action_mask(canvas)
        logits, acts = model.forward_with_activations(canvas)
        logp = masked_log_softmax(logits, mask)
        a = int(torch.multinomial(logp.exp(), 1))
        lp = float(logp[0, a])
        sum_logpf += lp
        e1m, e1c = chan_maps(acts["e1"], 4)
        e2m, _ = chan_maps(acts["e2"], 0)
        mm, _ = chan_maps(acts["mid"], 0)
        d1m, d1c = chan_maps(acts["d1"], 4)
        steps.append({
            "canvas": canvas_id(canvas),
            "probs": sig(logp[0].exp()),
            "action": a,
            "logp": round(lp, 4),
            "cum_logpf": round(sum_logpf, 4),
            "acts": {"e1": e1m, "e1c": e1c, "e2": e2m, "mid": mm, "d1": d1m, "d1c": d1c,
                     "logits": r4(acts["pix"][0]), "stop_logit": round(float(acts["stop"][0, 0]), 4)},
        })
        canvas = env.step(canvas, torch.tensor([a]))
        if a == env.STOP:
            break
    n = int(canvas.sum())
    log_r = float(env.log_reward(canvas)[0])
    log_pb = -math.lgamma(n + 1)
    return {
        "steps": steps, "final": canvas_id(canvas), "n": n,
        "log_r": round(log_r, 4), "log_pf": round(sum_logpf, 4), "log_pb": round(log_pb, 4),
        "log_z": round(float(log_z), 4),
        "residual": round(float(log_z) + sum_logpf - log_r - log_pb, 4),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--iters", type=int, default=1500)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="out")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    torch.set_num_threads(max(1, os.cpu_count() // 2))
    env = GridEnv(size=4, seed=args.seed)
    model = UNetPolicy(base=32)
    n_params = sum(q.numel() for q in model.parameters())

    states = env.all_states()
    ids = torch.arange(states.shape[0])
    log_r_all = env.log_reward(states).double()
    true_log_z = torch.logsumexp(log_r_all, 0)
    p_star = (log_r_all - true_log_z).exp()
    protos_flat = env.protos.reshape(env.protos.shape[0], -1)
    hamming = (states.reshape(states.shape[0], -1)[:, None] - protos_flat[None]).abs().sum(-1)
    hmin = hamming.min(1).values
    proto_ids = [canvas_id(pr) for pr in env.protos]
    top40 = torch.argsort(p_star, descending=True)[:40]

    ckpt_records = []
    trajectories = {}

    eval_seconds = [0.0]

    def on_checkpoint(it, model, log_z):
        t = time.time()
        model.eval()
        p_model = exact_terminal_distribution(model, env)
        rec = {
            "it": it,
            "tv": round(0.5 * (p_model - p_star).abs().sum().item(), 4),
            "corr": round(torch.corrcoef(torch.stack([p_model, p_star]))[0, 1].item(), 4),
            "log_z": round(float(log_z), 4),
            "top40_p": sig(p_model[top40]),
            "mass1px": round(p_model[hmin <= 1].sum().item(), 4),
            "proto_hits": sig(p_model[proto_ids]),
            "samples": [canvas_id(c) for c in rollout(model, env, 16).canvas],
        }
        if it in TRAJ_CKPTS:
            trajectories[str(it)] = [record_trajectory(model, env, log_z) for _ in range(TRAJ_CKPTS[it])]
        ckpt_records.append(rec)
        model.train()
        eval_seconds[0] += time.time() - t
        print(f"  ckpt {it:5d}: TV {rec['tv']:.4f} corr {rec['corr']:.4f} logZ {rec['log_z']:.3f} ({time.time() - t:.1f}s)")

    t0 = time.time()
    log_z, history = train(env, model, iters=args.iters, batch=args.batch,
                           checkpoints=CKPTS, on_checkpoint=on_checkpoint)
    train_seconds = time.time() - t0 - eval_seconds[0]  # optimisation only, checkpoint evaluation excluded

    model.eval()
    sm = sample_metrics(model, env)
    p_model = exact_terminal_distribution(model, env)
    g = torch.Generator().manual_seed(1)
    sub = torch.randperm(states.shape[0], generator=g)[:3000]
    sub = torch.unique(torch.cat([sub, torch.argsort(p_star, descending=True)[:150]]))
    sample_canvas = sm["canvas"][:64]
    sample_h = (sample_canvas.reshape(64, -1)[:, None] - protos_flat[None]).abs().sum(-1).min(1).values

    run = {
        "meta": {"size": 4, "n_protos": env.protos.shape[0], "tau": env.tau, "floor": env.reward_floor,
                 "iters": args.iters, "batch": args.batch, "lr": 2e-3, "eps": 0.1, "base": 32,
                 "params": n_params, "train_seconds": round(train_seconds, 1), "seed": args.seed,
                 "threads": torch.get_num_threads(), "n_states": states.shape[0]},
        "protos": [[int(v) for v in pr.reshape(-1)] for pr in env.protos],
        "proto_ids": proto_ids,
        "true_log_z": round(true_log_z.item(), 4),
        "target": {"mass1px": round(p_star[hmin <= 1].sum().item(), 4),
                   "mass2px": round(p_star[hmin <= 2].sum().item(), 4),
                   "proto_hits": sig(p_star[proto_ids])},
        "history": {"it": [h[0] for h in history], "loss": r4([h[1] for h in history]),
                    "log_z": r4([h[2] for h in history]), "log_r": r4([h[3] for h in history])},
        "top40": {"ids": top40.tolist(), "p_star": sig(p_star[top40]),
                  "hmin": hmin[top40].tolist(), "n": states[top40].reshape(40, -1).sum(1).tolist()},
        "checkpoints": ckpt_records,
        "trajectories": trajectories,
        "final": {
            "tv": round(0.5 * (p_model - p_star).abs().sum().item(), 4),
            "corr": round(torch.corrcoef(torch.stack([p_model, p_star]))[0, 1].item(), 4),
            "log_z": round(float(log_z), 4),
            "mean_reward": round(sm["mean_reward"], 4),
            "unique_states": sm["unique_states"],
            "frac1px": round(sm["frac_within_1_of_proto"], 4),
            "proto_hits_sample": r4(sm["proto_hits"]),
            "mass1px_exact": round(p_model[hmin <= 1].sum().item(), 4),
            "scatter": {"p_star": sig(p_star[sub], 3), "p_model": sig(p_model[sub], 3), "hmin": hmin[sub].tolist()},
            "samples": [canvas_id(c) for c in sample_canvas],
            "samples_hmin": sample_h.tolist(),
        },
    }
    os.makedirs(args.out, exist_ok=True)
    with open(f"{args.out}/run.json", "w") as f:
        json.dump(run, f, separators=(",", ":"))
    print(f"wrote {args.out}/run.json ({os.path.getsize(f'{args.out}/run.json') // 1024} KB)")
    tpl = "viz/report_template.html"
    if os.path.exists(tpl):
        body = open(tpl, encoding="utf-8").read().replace("/*__RUN_JSON__*/", json.dumps(run, separators=(",", ":")))
        html = ('<!doctype html><html lang="ru"><head><meta charset="utf-8">'
                '<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover"></head><body>\n'
                + body + "\n</body></html>\n")
        with open(f"{args.out}/report.html", "w", encoding="utf-8") as f:
            f.write(html)
        print(f"wrote {args.out}/report.html")


if __name__ == "__main__":
    main()
