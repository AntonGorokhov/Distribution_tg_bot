"""Diagnostic figures: training curves, prototypes vs samples, model vs target probabilities."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch


def make_plots(env, history, sm, em, out):
    it, loss, logz, logr = zip(*history)

    fig, ax = plt.subplots(1, 3, figsize=(14, 3.6))
    ax[0].plot(it, loss); ax[0].set_yscale("log"); ax[0].set_title("TB loss")
    ax[1].plot(it, logz, label="learned")
    if em: ax[1].axhline(em["true_log_z"], color="k", ls="--", label="true")
    ax[1].legend(); ax[1].set_title("log Z")
    ax[2].plot(it, logr); ax[2].set_title("mean log R (batch)")
    for a in ax: a.set_xlabel("iter")
    fig.tight_layout(); fig.savefig(f"{out}/training.png", dpi=120); plt.close(fig)

    K = env.protos.shape[0]
    n_show = 10
    fig, ax = plt.subplots(2, max(K, n_show), figsize=(1.3 * max(K, n_show), 3))
    for a in ax.ravel(): a.axis("off")
    for k in range(K):
        ax[0, k].imshow(env.protos[k], cmap="gray_r", vmin=0, vmax=1); ax[0, k].set_title(f"proto {k}", fontsize=8)
    idx = torch.randperm(sm["canvas"].shape[0])[:n_show]
    for j, i in enumerate(idx):
        ax[1, j].imshow(sm["canvas"][i, 0], cmap="gray_r", vmin=0, vmax=1)
    ax[1, 0].set_title("samples", fontsize=8, loc="left")
    fig.tight_layout(); fig.savefig(f"{out}/samples.png", dpi=120); plt.close(fig)

    if em:
        top = torch.argsort(em["p_star"], descending=True)[:40]
        fig, ax = plt.subplots(figsize=(10, 3.2))
        x = torch.arange(len(top))
        ax.bar(x - 0.2, em["p_star"][top], width=0.4, label="R(x)/Z")
        ax.bar(x + 0.2, em["p_model"][top], width=0.4, label="P_model(x)")
        ax.set_title("top-40 states by target probability"); ax.set_xlabel("state rank"); ax.legend()
        fig.tight_layout(); fig.savefig(f"{out}/target_vs_model.png", dpi=120); plt.close(fig)

        fig, ax = plt.subplots(figsize=(4, 4))
        ax.loglog(em["p_star"], em["p_model"], ".", ms=2, alpha=0.4)
        lim = [em["p_star"].min().item() * 0.5, 1]
        ax.plot(lim, lim, "k--", lw=1); ax.set_xlabel("R(x)/Z"); ax.set_ylabel("P_model(x)")
        ax.set_title("all 65536 states")
        fig.tight_layout(); fig.savefig(f"{out}/scatter.png", dpi=120); plt.close(fig)
