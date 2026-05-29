"""
Figure 2 -- Empirical convergence rate of the posterior.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from math import erf
from simulators import Simulator2, sigmoid


def norm_cdf(z):
    return 0.5 * (1.0 + np.vectorize(erf)(np.asarray(z) / np.sqrt(2.0)))


# ------------------------------------------------------------------
N_RUNS = 40
T = 200.0
P0_PRIOR = 0.5
THETA = 1.0
FMAX = 1.0
EPS = 0.1
FIGDIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures"
)
os.makedirs(FIGDIR, exist_ok=True)


def simulate_interspike_logratios(n_samples=5000, theta=THETA, f=sigmoid, rng=None):
    rng = rng if rng is not None else np.random.default_rng(0)
    f0 = float(f(0.0))
    logs = np.empty(n_samples)
    for k in range(n_samples):
        V = 0.0
        t = 0.0
        int_fV = 0.0
        while True:
            tau_tilde = rng.exponential(1.0 / theta)
            tau_N = rng.exponential(1.0 / max(f(V), 1e-12))
            if tau_N < tau_tilde:
                int_fV += f(V) * tau_N
                S = t + tau_N
                log_ratio = (
                    np.log(max(f(V), 1e-12))
                    - np.log(f0)
                    - (int_fV - S * f0)
                )
                logs[k] = log_ratio
                break
            else:
                int_fV += f(V) * tau_tilde
                V += 1.0
                t += tau_tilde
    return logs


def run_batch(true_J, n_runs):
    return [
        Simulator2(theta=THETA, p0=P0_PRIOR, true_J=true_J,
                   f_max=FMAX, T=T, seed=seed).run()
        for seed in range(n_runs)
    ]


def extract_logratio_series(run):
    events = np.sort(np.concatenate([run["spikes_N"], run["spikes_Nt"]]))
    t = run["times"]; p = run["posterior"]
    idx = np.clip(np.searchsorted(t, events, side="right") - 1, 0, len(t) - 1)
    p_at = np.clip(p[idx], 1e-12, 1 - 1e-12)
    log_r = np.log(p_at / (1 - p_at))
    return np.arange(1, len(events) + 1), log_r


def hitting_time_k_eps(run, eps):
    target = run["true_J"]
    events = np.sort(np.concatenate([run["spikes_N"], run["spikes_Nt"]]))
    t = run["times"]; p = run["posterior"]
    idx = np.clip(np.searchsorted(t, events, side="right") - 1, 0, len(t) - 1)
    p_at = p[idx]
    hit = np.where(np.abs(p_at - target) < eps)[0]
    return int(hit[0] + 1) if len(hit) > 0 else None


if __name__ == "__main__":
    rng = np.random.default_rng(12345)

    print("Estimating KL(K_1, K_0) empirically ...", flush=True)
    logs = simulate_interspike_logratios(n_samples=5000, rng=rng)
    KL_est = float(logs.mean())
    sigma2 = float(logs.var(ddof=1))
    print(f"  KL(K_1, K_0) estimate = {KL_est:.4f}")
    print(f"  Var(log K_1/K_0 | J=1) = {sigma2:.4f}")

    runs1 = run_batch(true_J=1, n_runs=N_RUNS)
    runs0 = run_batch(true_J=0, n_runs=N_RUNS)

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), constrained_layout=True)

    ax = axes[0]
    for r in runs1:
        k, lr = extract_logratio_series(r)
        ax.plot(k, lr, color="C0", alpha=0.25, lw=0.7)
    for r in runs0:
        k, lr = extract_logratio_series(r)
        ax.plot(k, lr, color="C3", alpha=0.25, lw=0.7)

    k_max = 60
    ks = np.arange(1, k_max + 1)
    ax.plot(ks, KL_est * ks, color="C0", lw=2.0, ls="--",
            label=f"theory J=1: slope ~KL = {KL_est:.2f}")
    ax.plot(ks, -KL_est * ks, color="C3", lw=2.0, ls="--",
            label="theory J=0: slope -KL")
    ax.set_xlim(0, k_max)
    ax.set_xlabel("number of observed events k")
    ax.set_ylabel(r"$\log r(\tau_k)$")
    ax.set_title("(a) log-likelihood ratio vs event count")
    ax.axhline(0.0, color="k", lw=0.6, alpha=0.6)
    ax.legend(fontsize=9, loc="lower left")

    ax = axes[1]
    k_eps_vals = [hitting_time_k_eps(r, EPS) for r in runs1]
    k_eps_vals = np.array([k for k in k_eps_vals if k is not None])
    if len(k_eps_vals) >= 2:
        k_sorted = np.sort(k_eps_vals)
        cdf = np.arange(1, len(k_sorted) + 1) / len(k_sorted)
        ax.step(k_sorted, cdf, where="post", color="C0",
                label=f"empirical (J=1, eps={EPS})")

    A_eps = np.log((1 - EPS) / EPS)
    if len(k_eps_vals):
        k_grid = np.arange(1, max(int(k_eps_vals.max()), 60) + 1)
    else:
        k_grid = np.arange(1, 60)
    z = (A_eps - k_grid * KL_est) / (np.sqrt(k_grid) * np.sqrt(sigma2))
    ccdf_theory = 1 - norm_cdf(z)
    ax.plot(k_grid, ccdf_theory, color="k", lw=1.4, ls=":",
            label="CLT approximation")
    ax.set_xlim(0, max(60, int(k_grid.max())))
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("number of events k")
    ax.set_ylabel(r"$\mathbb{P}(k_\varepsilon \leq k)$")
    ax.set_title(f"(b) hitting time under J=1, eps={EPS}")
    ax.legend(fontsize=9, loc="lower right")

    out = os.path.join(FIGDIR, "fig2_convergence_rate.pdf")
    fig.savefig(out, bbox_inches="tight")
    print(f"Saved {out}")
