#!/usr/bin/env python3
"""Train a GFlowNet (U-Net policy, trajectory balance) on a synthetic binary-grid task. CPU only."""
import argparse
import os
import time

import torch

from gfn.env import GridEnv
from gfn.evaluate import exact_metrics, sample_metrics
from gfn.trainer import train
from gfn.unet import UNetPolicy


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--size", type=int, default=4, help="grid side; exact eval only when size*size <= 16")
    p.add_argument("--protos", type=int, default=3)
    p.add_argument("--floor", type=float, default=1e-4, help="additive reward floor")
    p.add_argument("--tau", type=float, default=0.4)
    p.add_argument("--iters", type=int, default=1500)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--eps", type=float, default=0.1, help="exploration: prob of uniform random action")
    p.add_argument("--base", type=int, default=32, help="U-Net base channels")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="out")
    p.add_argument("--no-plot", action="store_true")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    torch.set_num_threads(max(1, os.cpu_count() // 2))

    env = GridEnv(size=args.size, n_protos=args.protos, tau=args.tau, reward_floor=args.floor, seed=args.seed)
    model = UNetPolicy(base=args.base)
    print(env, "| params:", sum(p.numel() for p in model.parameters()))

    exact = env.H * env.W <= 16
    if exact:
        print(f"true log Z = {env.exact_log_z():.4f}")

    t0 = time.time()
    log_z, history = train(env, model, iters=args.iters, batch=args.batch, lr=args.lr, eps=args.eps)
    print(f"training done in {time.time() - t0:.1f}s")

    os.makedirs(args.out, exist_ok=True)
    torch.save({"model": model.state_dict(), "log_z": log_z, "args": vars(args)}, f"{args.out}/model.pt")

    sm = sample_metrics(model, env)
    print("\n== on-policy samples (20k) ==")
    print(f"mean reward          : {sm['mean_reward']:.4f}")
    print(f"unique terminal states: {sm['unique_states']}")
    print(f"frac within 1px of a prototype: {sm['frac_within_1_of_proto']:.3f}")
    print(f"exact prototype hit rates: {[round(h, 3) for h in sm['proto_hits']]}")

    em = None
    if exact:
        em = exact_metrics(model, env, log_z)
        print("\n== exact (DP over all 2^16 states) ==")
        print(f"sum P_model(x)        : {em['mass_check']:.4f}")
        print(f"L1(P_model, R/Z)      : {em['l1']:.4f}   (TV = {em['tv']:.4f})")
        print(f"corr(P_model, R/Z)    : {em['corr']:.4f}")
        print(f"log Z  learned / true : {em['learned_log_z']:.4f} / {em['true_log_z']:.4f}")

    if not args.no_plot:
        from plot import make_plots
        make_plots(env, history, sm, em, args.out)
        print(f"\nplots written to {args.out}/")


if __name__ == "__main__":
    main()
