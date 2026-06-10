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
FIGDIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures"
)
os.makedirs(FIGDIR, exist_ok=True)


if __name__ == "__main__":
    sim = SimulatorN(n=N, T=T, p_edge=0.4, log_stride=50, seed=7)
    res = sim.run()
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
    # Also report final recovery error
    err_F = np.linalg.norm(W_mean[idx_end] - W_true, ord="fro")
    print(f"Final Frobenius error ||W_mean - W_true||_F = {err_F:.3f}")
