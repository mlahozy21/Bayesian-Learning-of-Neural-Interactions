"""
Figure 4 -- Mean recovery error of the interaction matrix as a function of
time, for several network sizes n.

Results are cached in `cache/fig4_n{n}_T{T}_seed{s}.npz`, so re-running the
script is cheap after the first (expensive) run.  Pass `n=5` to only run
a specific network size, or `--plot-only` to just plot from cache.
"""

import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from simulators import SimulatorN, posterior_W_mean

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGDIR = os.path.join(BASE, "figures")
CACHE = os.path.join(BASE, "cache")
os.makedirs(FIGDIR, exist_ok=True)
os.makedirs(CACHE, exist_ok=True)

N_REPLICATES = 3
CONFIGS = [
    dict(n=3, T=400.0,  p_edge=0.5, log_stride=40),
    dict(n=5, T=800.0,  p_edge=0.4, log_stride=80),
    dict(n=7, T=1200.0, p_edge=0.3, log_stride=200),
]


def cache_path(cfg, seed):
    return os.path.join(
        CACHE, f"fig4_n{cfg['n']}_T{int(cfg['T'])}_seed{seed}.npz"
    )


def run_one(cfg, seed):
    path = cache_path(cfg, seed)
    if os.path.exists(path):
        d = np.load(path)
        return d["t"], d["e"]

    t0 = time.time()
    sim = SimulatorN(**cfg, seed=seed)
    res = sim.run()
    W_mean_t = posterior_W_mean(res["posterior"], res["W_spaces"])
    W_true = res["W_true"].astype(float)
    n = cfg["n"]
    norm = np.sqrt(n * (n - 1))
    err = np.linalg.norm(W_mean_t - W_true[None, :, :], axis=(1, 2)) / norm
    t = res["times"]
    np.savez(path, t=t, e=err)
    print(f"  [n={n} seed={seed}] ran in {time.time()-t0:.1f}s, cached")
    return t, err


def plot_from_cache():
    fig, ax = plt.subplots(figsize=(6.0, 4.2), constrained_layout=True)
    colors = ["C0", "C3", "C2"]
    for col, cfg in zip(colors, CONFIGS):
        all_t, all_e = [], []
        max_T = cfg["T"]
        for seed in range(N_REPLICATES):
            p = cache_path(cfg, seed)
            if not os.path.exists(p):
                print(f"  [n={cfg['n']} seed={seed}] cache missing; skipping.")
                continue
            d = np.load(p)
            all_t.append(d["t"])
            all_e.append(d["e"])
        if not all_e:
            continue
        grid = np.linspace(0, max_T, 200)
        interp = np.stack([np.interp(grid, t, e) for t, e in zip(all_t, all_e)])
        mean = interp.mean(axis=0)
        std = interp.std(axis=0)
        ax.plot(grid, mean, color=col, lw=1.8, label=f"n = {cfg['n']}")
        ax.fill_between(grid, mean - std, mean + std, color=col, alpha=0.2)

    ax.set_xlabel(r"time $t$")
    ax.set_ylabel(r"$\|\hat W_t - W_{\star}\|_F / \sqrt{n(n-1)}$")
    ax.set_title(f"Posterior recovery error vs time ({N_REPLICATES} replicates)")
    ax.axhline(0.0, color="k", lw=0.5, alpha=0.5)
    ax.legend(fontsize=10)

    out = os.path.join(FIGDIR, "fig4_error_vs_T.pdf")
    fig.savefig(out, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    only_n = None
    only_seeds = None
    plot_only = False
    for arg in sys.argv[1:]:
        if arg.startswith("n="):
            only_n = int(arg.split("=")[1])
        elif arg.startswith("seeds="):
            only_seeds = [int(x) for x in arg.split("=")[1].split(",")]
        elif arg == "--plot-only":
            plot_only = True

    if not plot_only:
        for cfg in CONFIGS:
            if only_n is not None and cfg["n"] != only_n:
                continue
            seeds = only_seeds if only_seeds is not None else range(N_REPLICATES)
            for seed in seeds:
                try:
                    run_one(cfg, seed)
                except Exception as exc:
                    print(f"  [n={cfg['n']} seed={seed}] failed: {exc}")

    if only_n is not None and not plot_only:
        print("Subset simulation done; run with --plot-only to plot.")
        sys.exit(0)

    plot_from_cache()
