"""
Figure 1 — Posterior trajectories for the two-neuron case.

Two panels, one per ground-truth value of J.  For each panel we show
several Monte Carlo trajectories of  p_t^1 = P(J = 1 | G_t)  and their
pointwise mean, to illustrate (i) convergence to the truth and
(ii) run-to-run variability.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from simulators import Simulator2

# ------------------------------------------------------------------
N_RUNS   = 20
T        = 150.0
P0_PRIOR = 0.5
THETA    = 1.0
FMAX     = 1.0
FIGDIR   = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "figures")
os.makedirs(FIGDIR, exist_ok=True)

# ------------------------------------------------------------------

def run_batch(true_J: int):
    runs = []
    for seed in range(N_RUNS):
        res = Simulator2(
            theta=THETA, p0=P0_PRIOR, true_J=true_J,
            f_max=FMAX, T=T, seed=seed
        ).run()
        runs.append(res)
    return runs


def plot_batch(ax, runs, true_J: int):
    # Interpolate every run on a common grid for averaging.
    grid = np.linspace(0.0, T, 1000)
    curves = []
    for r in runs:
        c = np.interp(grid, r["times"], r["posterior"])
        curves.append(c)
        ax.plot(r["times"], r["posterior"], color="C0", lw=0.5, alpha=0.3)
    curves = np.stack(curves, axis=0)
    mean_curve = curves.mean(axis=0)
    ax.plot(grid, mean_curve, color="C3", lw=2.0, label="MC mean")
    ax.axhline(float(true_J), color="k", ls="--", lw=1.0, alpha=0.7,
               label=f"true J = {true_J}")
    ax.set_xlim(0, T); ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$p_t^1 = \mathbb{P}(J=1 \mid \mathcal{G}_t)$")
    ax.set_title(fr"true $J = {true_J}$  ({N_RUNS} runs)")
    ax.legend(loc="center right", fontsize=9)


if __name__ == "__main__":
    runs_J1 = run_batch(1)
    runs_J0 = run_batch(0)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True,
                             constrained_layout=True)
    plot_batch(axes[0], runs_J1, true_J=1)
    plot_batch(axes[1], runs_J0, true_J=0)
    fig.suptitle("Posterior trajectories in the two-neuron model",
                 fontsize=11)

    out = os.path.join(FIGDIR, "fig1_posterior_n2.pdf")
    fig.savefig(out, bbox_inches="tight")
    print(f"Saved {out}")
