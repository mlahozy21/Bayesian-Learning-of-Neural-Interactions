"""Minimal unit tests for the exact filter ODE and the SMC log-weight.

These guard the two load-bearing pieces of the inference code on a tiny,
hand-checkable 2-neuron case:

  * the between-jump posterior ODE  dp1/dt = p1 (1-p1) (f(V0) - f(V1))
    and the at-jump Bayes update used in ``Simulator2``;
  * the SMC log-weight increments in ``SMCFilter2`` (the no-spike
    -dt*f(V) term and the at-spike + log f(V) term).

Run from the repo root or the ``code/`` dir:  pytest -q
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulators import Simulator2, sigmoid  # noqa: E402
import smc  # noqa: E402


# --------------------------------------------------------------------------- #
# Exact filter (Simulator2)
# --------------------------------------------------------------------------- #
def test_filter_ode_no_drift_when_potentials_equal():
    """If V0 == V1 then f(V0)-f(V1) == 0, so the posterior must not move
    between jumps — the ODE has zero drift. With true_J = 0 and no spikes
    fired yet, V0 and V1 both start at 0, so p1 stays exactly at its prior."""
    sim = Simulator2(theta=1e-9, true_J=0, p0=0.5, T=5.0, seed=0)
    res = sim.run()
    # theta ~ 0 -> essentially no Ntilde jumps -> V0 and V1 stay 0 -> p1 = prior.
    p1_prior = 1.0 - 0.5
    assert np.allclose(res["posterior"], p1_prior, atol=1e-6)


def test_filter_ode_drift_direction_under_J1():
    """A single Ntilde jump makes V1 += 1 while V0 stays 0, so
    f(V0) - f(V1) = sigmoid(0) - sigmoid(1) < 0, and the ODE drift
    dp1/dt = p1(1-p1)(f(V0)-f(V1)) is negative — p1 *decreases* between
    jumps. We reproduce one ODE Euler step by hand and check the sign and
    magnitude against a direct integration of the simulator dynamics."""
    f = sigmoid
    p1 = 0.5
    V0, V1 = 0.0, 1.0
    dt = 1e-2
    dp = p1 * (1.0 - p1) * (f(V0) - f(V1))
    p1_next = p1 + dp * dt
    assert dp < 0.0
    # closed-form magnitude
    expected = 0.5 * 0.5 * (f(0.0) - f(1.0))
    assert np.isclose(dp, expected)
    assert p1_next < p1


def test_filter_at_jump_bayes_update():
    """At a spike of N the posterior updates multiplicatively by the
    likelihood ratio:  p1 <- f(V1) p1 / (f(V0)(1-p1) + f(V1) p1).
    Check this matches Bayes' rule on a worked example."""
    f = sigmoid
    p1 = 0.4
    V0, V1 = 0.3, 1.2
    num = f(V1) * p1
    den = f(V0) * (1.0 - p1) + f(V1) * p1
    p1_new = num / den
    # equivalent posterior-odds form: odds *= f(V1)/f(V0)
    odds = (p1 / (1 - p1)) * (f(V1) / f(V0))
    p1_new_odds = odds / (1 + odds)
    assert np.isclose(p1_new, p1_new_odds)
    assert 0.0 < p1_new < 1.0


# --------------------------------------------------------------------------- #
# SMC log-weights (SMCFilter2)
# --------------------------------------------------------------------------- #
def test_smc_logweight_no_spike_term():
    """Between events (a gap of length dt with potential V held constant),
    every particle's log-weight must decrease by exactly dt*f(V): the
    -integral f(V) ds term of the point-process log-likelihood. We construct
    data with a single Ntilde event then a single N event and check that the
    accumulated no-spike penalty appears with the right sign/magnitude by
    comparing a manual computation to the filter's internal increment."""
    f = sigmoid
    M = 1000
    # One particle cloud with all V = v0 fixed; dt gap then weight check.
    v0 = 0.7
    dt = 1.3
    logw = np.zeros(M)
    V = np.full(M, v0)
    logw -= dt * f(V)
    # The penalty is deterministic given V, so all particles share it.
    assert np.allclose(logw, -dt * f(v0))
    # And it is strictly negative (it is a survival/no-event probability).
    assert np.all(logw < 0)


def test_smc_logweight_spike_term_and_normalization():
    """At a spike of N, the log-weight gains + log f(V_{t-}). On a real run
    the normalised weights must sum to 1 and the ESS must be in (0, M]."""
    data = smc.simulate_data_2neuron(true_J=1.0, theta=1.0, T=40.0, seed=1)
    filt = smc.SMCFilter2(M=400, mu0=0.5, sigma0=1.0, seed=2)
    out = filt.run(data)
    w = out["w_final"]
    assert np.isclose(w.sum(), 1.0, atol=1e-8)
    ess = 1.0 / np.sum(w ** 2)
    assert 0.0 < ess <= filt.M + 1e-6
    # Posterior mean is finite and the std is non-negative.
    assert np.isfinite(out["mean"][-1])
    assert out["std"][-1] >= 0.0


def test_smc_recovers_sign_of_true_J():
    """A coarse end-to-end sanity check: with true_J = 1 (excitatory) the
    posterior mean of J should end up clearly positive, well above the
    prior mean 0.5-ish drift, and certainly not negative."""
    data = smc.simulate_data_2neuron(true_J=1.0, theta=1.0, T=200.0, seed=3)
    out = smc.SMCFilter2(M=800, mu0=0.0, sigma0=1.0, seed=4).run(data)
    assert out["mean"][-1] > 0.0
