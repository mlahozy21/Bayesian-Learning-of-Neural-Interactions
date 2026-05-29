"""
smc.py
======

Sequential Monte Carlo (bootstrap particle filter) for inference of a
*continuous* interaction parameter in the two-neuron point-process model.

Model
-----
* Ntilde is a Poisson process of rate ``theta`` (exogenous).
* V_t is the potential of the observed neuron with dynamics
      dV_t = J dNtilde_t - V_{t-} dN_t,   V_0 = 0,
  where J in R is unknown.
* N has stochastic intensity f(V_{t-}).  We observe both (Ntilde, N).

Prior: J ~ N(mu0, sigma0^2).

SMC algorithm (bootstrap filter for a static parameter)
-------------------------------------------------------
* Draw particles J^(m) ~ N(mu0, sigma0^2), m = 1, ..., M.
* Each particle maintains its own potential V^(m).  Initial V^(m) = 0.
* Sequentially process events in chronological order:
    - At a jump of Ntilde at time t: V^(m) += J^(m) (no weight update,
      since Ntilde has a known exogenous rate theta).
    - At a jump of N at time t: multiplicative weight update
        w^(m) <- w^(m) * f(V^(m)(t-));
      then reset V^(m) <- 0 for all particles.
    - Between events we accumulate the negative log likelihood
        -integral_{s in [t_prev, t]} f(V^(m)(s)) ds = -(t - t_prev) f(V^(m)),
      since V^(m) is constant between events.  This is applied
      multiplicatively to w^(m) as well.
* After each event, compute ESS.  If ESS < ess_frac * M, resample with
  replacement and apply a small Gaussian jitter
      J^(m) <- J^(m) + sigma_jitter * eps,   eps ~ N(0, 1),
  to preserve particle diversity for the static parameter J.  The jitter
  plays the role of the artificial-dynamics perturbation of Liu and West
  (2001) and is shrunk with time so the filter is asymptotically unbiased
  (see Chopin & Papaspiliopoulos 2020, Chapter 17).

Reference: Doucet, Godsill & Andrieu (2000); Chopin & Papaspiliopoulos
(2020); Liu & West (2001).
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable
import numpy as np


def sigmoid(v):
    return 1.0 / (1.0 + np.exp(-v))


# ---------------------------------------------------------------------
# Data generation for the 2-neuron model with continuous J
# ---------------------------------------------------------------------

def simulate_data_2neuron(
    true_J: float,
    theta: float = 1.0,
    f: Callable[[float], float] = sigmoid,
    f_max: float = 1.0,
    T: float = 200.0,
    seed: int | None = None,
):
    """Simulate (N, Ntilde) on [0, T] with known true_J."""
    rng = np.random.default_rng(seed)
    t, V = 0.0, 0.0
    spikes_N, spikes_Nt = [], []
    while t < T:
        tau_tilde = rng.exponential(1.0 / theta)
        tau_N = rng.exponential(1.0 / f_max)
        t_tilde, t_N = t + tau_tilde, t + tau_N
        t_next = min(t_tilde, t_N, T)
        if t_next >= T:
            break
        if t_next == t_tilde:
            V += true_J
            t = t_tilde
            spikes_Nt.append(t)
        else:
            t = t_N
            if rng.uniform() < f(V) / f_max:
                spikes_N.append(t)
                V = 0.0
    return {
        "spikes_N": np.asarray(spikes_N),
        "spikes_Nt": np.asarray(spikes_Nt),
        "true_J": true_J,
        "T": T,
    }


# ---------------------------------------------------------------------
# Bootstrap particle filter
# ---------------------------------------------------------------------

@dataclass
class SMCFilter2:
    """
    Bootstrap particle filter for the two-neuron model with continuous J.

    Parameters
    ----------
    M           : number of particles.
    mu0, sigma0 : prior mean and std for J (Gaussian).
    f, f_max    : intensity function and its upper bound.
    ess_frac    : resample when ESS / M < ess_frac.
    sigma_jitter0: initial jitter std applied to J after resampling.
    jitter_decay: jitter std at step k is sigma_jitter0 * (1+k)^(-jitter_decay/2),
                  decaying slowly to preserve asymptotic correctness.
    """

    M:             int   = 500
    mu0:           float = 0.5
    sigma0:        float = 1.0
    f:             Callable[[float], float] = field(default=sigmoid)
    f_max:         float = 1.0
    ess_frac:      float = 0.5
    sigma_jitter0: float = 0.05
    jitter_decay:  float = 0.3
    seed:          int | None = None

    # ---------------------------------------------------------------
    def run(self, data: dict) -> dict:
        rng = np.random.default_rng(self.seed)
        M = self.M

        # Prior sample
        J = rng.normal(self.mu0, self.sigma0, size=M)
        V = np.zeros(M)
        logw = np.zeros(M)  # log weights (unnormalised)

        # Merge and sort events with tags
        events = []
        for t in data["spikes_Nt"]:
            events.append((t, "tilde"))
        for t in data["spikes_N"]:
            events.append((t, "obs"))
        events.sort(key=lambda x: x[0])

        # Logs
        times_log = [0.0]
        mean_log = [float(np.mean(J))]
        std_log = [float(np.std(J))]
        ess_log = [float(M)]

        t_prev = 0.0
        resample_counter = 0

        for k, (t_ev, tag) in enumerate(events):
            dt_ev = t_ev - t_prev
            # No-spike-in-dt_ev contribution: exp(-integral f(V) ds)
            # V is constant between events, so this is exp(-dt_ev * f(V)).
            logw -= dt_ev * self.f(V)

            if tag == "tilde":
                # Exogenous Poisson jump: V <- V + J
                V = V + J
            else:  # 'obs' = spike of N
                # Likelihood update by intensity f(V_{t-})
                logw = logw + np.log(np.clip(self.f(V), 1e-300, None))
                # Reset potential
                V = np.zeros(M)

            # Normalise weights, compute ESS, possibly resample
            lw = logw - np.max(logw)
            w = np.exp(lw)
            w_sum = w.sum()
            if w_sum <= 0:
                w = np.ones(M) / M
            else:
                w = w / w_sum
            ess = 1.0 / np.sum(w ** 2)

            if ess < self.ess_frac * M:
                idx = rng.choice(M, size=M, replace=True, p=w)
                J = J[idx].copy()
                V = V[idx].copy()
                # Jitter J to preserve diversity; decay slowly.
                sigma_k = self.sigma_jitter0 * (1.0 + resample_counter) ** (-self.jitter_decay / 2.0)
                J = J + sigma_k * rng.standard_normal(M)
                logw = np.zeros(M)
                w = np.ones(M) / M
                resample_counter += 1

            # Log posterior moments
            times_log.append(t_ev)
            mean_log.append(float(np.sum(w * J)))
            std_log.append(float(np.sqrt(max(0.0, np.sum(w * (J - np.sum(w * J)) ** 2)))))
            ess_log.append(float(ess))

            t_prev = t_ev

        return {
            "times": np.asarray(times_log),
            "mean":  np.asarray(mean_log),
            "std":   np.asarray(std_log),
            "ess":   np.asarray(ess_log),
            "J_final": J,
            "w_final": w,
        }
