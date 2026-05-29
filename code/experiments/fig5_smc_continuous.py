"""
Figure 5 -- SMC proof-of-concept: Bayesian recovery of a continuous
interaction parameter J in the 2-neuron model.

We simulate data under several ground-truth values J* and run a
bootstrap particle filter (`SMCFilter2`) with a diffuse Gaussian prior.
We then plot:
    (a) posterior mean E[J | G_t] +- 1 posterior std vs time, for
        several true J*;
    (b) final posterior histogram of J versus a vertical line at J*.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from smc import simulate_data_2neuron, SMCFilter2

T = 500.0
THETA = 1.0
M = 2000
TRUE_JS = [0.0, 1.0, 2.0]
COLORS = ["C2", "C0", "C3"]

FIGDIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures"
)
os.makedirs(FIGDIR, exist_ok=True)


if __name__ == "__main__":
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), constrained_layout=True)

    final_particles = {}
    for col, J_star in zip(COLORS, TRUE_JS):
        data = simulate_data_2neuron(
            true_J=J_star, theta=THETA, T=T, seed=int(10 * J_star) + 7
        )
        smc = SMCFilter2(
            M=M, mu0=0.5, sigma0=1.0, ess_frac=0.5,
            sigma_jitter0=0.08, jitter_decay=0.3, seed=int(10 * J_star),
        )
        res = smc.run(data)
        t = res["times"]; mean = res["mean"]; std = res["std"]

        ax = axes[0]
        ax.plot(t, mean, color=col, lw=1.6, label=fr"$J^\star = {J_star:.1f}$")
        ax.fill_between(t, mean - std, mean + std, color=col, alpha=0.2)
        ax.axhline(J_star, color=col, ls="--", lw=0.8, alpha=0.7)

        final_particles[J_star] = (res["J_final"], res["w_final"])

        print(
            f"J*={J_star:.1f}: final E[J]={mean[-1]:.3f}, "
            f"std={std[-1]:.3f}, n_events={len(t)-1}"
        )

    ax = axes[0]
    ax.set_xlabel(r"time $t$")
    ax.set_ylabel(r"$\mathbb{E}[J \mid \mathcal{G}_t]\ \pm\ \mathrm{std}$")
    ax.set_title("(a) SMC posterior for continuous $J$")
    ax.legend(fontsize=9, loc="best")

    # Right panel: final histograms
    ax = axes[1]
    for col, J_star in zip(COLORS, TRUE_JS):
        Jf, wf = final_particles[J_star]
        ax.hist(
            Jf, bins=40, weights=wf, alpha=0.45, color=col, density=True,
            label=fr"$J^\star = {J_star:.1f}$",
        )
        ax.axvline(J_star, color=col, ls="--", lw=1.3)
    ax.set_xlabel(r"$J$")
    ax.set_ylabel("posterior density")
    ax.set_title("(b) Final posterior of $J$")
    ax.legend(fontsize=9, loc="upper right")

    out = os.path.join(FIGDIR, "fig5_smc_continuous.pdf")
    fig.savefig(out, bbox_inches="tight")
    print(f"Saved {out}")
