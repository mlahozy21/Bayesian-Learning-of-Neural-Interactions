"""
Figure 3 -- Posterior recovery of the interaction matrix W for n=5.

Left panel : ground truth W_true (binary).
Middle     : posterior mean at T/4.
Right      : posterior mean at final time T.

Colorbar shows E[W_{j,i} | G_t] in [0,1].
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from simulators import SimulatorN, posterior_W_mean

N = 5
T = 1500.0
SEED = 7            # base seed; replicates use SEED, SEED+1, ...
N_REPLICATES = 10   # default #seeds for the headline mean±std (CPU-cheap)
FIGDIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures"
)
os.makedirs(FIGDIR, exist_ok=True)


def normalized_frobenius(W_mean_final, W_true, n):
    """Frobenius recovery error normalised by sqrt(n(n-1)).

    We report the *normalised* error consistently with fig4
    (``error_vs_T``), which divides by ``sqrt(n(n-1))`` (the number of
    off-diagonal entries). This makes the two figures' error scales
    comparable. The raw (un-normalised) Frobenius norm is also returned.
    """
    raw = float(np.linalg.norm(W_mean_final - W_true, ord="fro"))
    norm = np.sqrt(n * (n - 1))
    return raw, raw / norm


if __name__ == "__main__":
    # By default we average the final recovery error over N_REPLICATES seeds
    # and report mean±std (so the headline number is not a single-seed claim).
    # Override with --n-replicates K (e.g. K=1 for a quick single-seed run).
    n_rep = N_REPLICATES
    args = sys.argv[1:]
    for i, a in enumerate(args):
        if a in ("--n-replicates", "-r") and i + 1 < len(args):
            n_rep = int(args[i + 1])
        elif a.startswith("--n-replicates="):
            n_rep = int(a.split("=", 1)[1])

    raw_errs, norm_errs = [], []
    res = None
    for rep in range(n_rep):
        sim = SimulatorN(n=N, T=T, p_edge=0.4, log_stride=50, seed=SEED + rep)
        res_rep = sim.run()
        Wm = posterior_W_mean(res_rep["posterior"], res_rep["W_spaces"])
        Wt = res_rep["W_true"].astype(float)
        raw, normed = normalized_frobenius(Wm[-1], Wt, N)
        raw_errs.append(raw)
        norm_errs.append(normed)
        if rep == 0:
            res = res_rep  # keep the base-seed run for the figure

    if n_rep > 1:
        raw_errs = np.array(raw_errs); norm_errs = np.array(norm_errs)
        print(
            f"Final recovery error over {n_rep} seeds "
            f"(seed {SEED}..{SEED + n_rep - 1}):\n"
            f"  raw  ||W_mean - W_true||_F          = "
            f"{raw_errs.mean():.3f} ± {raw_errs.std():.3f}\n"
            f"  normalised /sqrt(n(n-1))            = "
            f"{norm_errs.mean():.3f} ± {norm_errs.std():.3f}"
        )

    # Figure uses the base-seed run.
    W_mean = posterior_W_mean(res["posterior"], res["W_spaces"])  # (T', n, n)
    W_true = res["W_true"].astype(float)
    times = res["times"]

    # pick three snapshots: t=0 (prior), t=T/4, t=T
    idx_start = 0
    idx_mid = np.searchsorted(times, T / 4)
    idx_end = len(times) - 1

    fig, axes = plt.subplots(1, 4, figsize=(13, 3.3), constrained_layout=True)
    titles = [
        f"(a) True W",
        f"(b) Posterior mean, t = 0 (prior)",
        f"(c) Posterior mean, t = {T/4:.0f}",
        f"(d) Posterior mean, t = {T:.0f}",
    ]
    mats = [W_true, W_mean[idx_start], W_mean[idx_mid], W_mean[idx_end]]

    for ax, m, ttl in zip(axes, mats, titles):
        im = ax.imshow(m, vmin=0, vmax=1, cmap="viridis")
        ax.set_title(ttl, fontsize=10)
        ax.set_xticks(range(N)); ax.set_yticks(range(N))
        ax.set_xlabel("target neuron i")
        ax.set_ylabel("source neuron j")
        for i in range(N):
            for j in range(N):
                ax.text(i, j, f"{m[j,i]:.2f}", ha="center", va="center",
                        color="w" if m[j, i] < 0.5 else "k", fontsize=7)
    fig.colorbar(im, ax=axes, shrink=0.75, label=r"$\mathbb{E}[W_{j,i}\mid\mathcal{G}_t]$")

    out = os.path.join(FIGDIR, "fig3_posterior_n5.pdf")
    fig.savefig(out, bbox_inches="tight")
    print(f"Saved {out}")
    # Also report final recovery error (base seed) — both raw and the
    # normalised version used by fig4, so the two figures are comparable.
    raw, normed = normalized_frobenius(W_mean[idx_end], W_true, N)
    print(f"Final Frobenius error (seed {SEED}):")
    print(f"  raw ||W_mean - W_true||_F           = {raw:.3f}")
    print(f"  normalised /sqrt(n(n-1))            = {normed:.3f}")
