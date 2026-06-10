# Bayesian Learning of Neural Interactions

A Bayesian framework that learns **who influences whom** in a network of neurons —
its *interaction structure* — observing only the neurons' spike (jump) times. The
neurons are modelled as a continuous-time **point-process interacting particle system**
(coupled stochastic integrate-and-fire units), and we recover the interaction matrix
that couples them.

MVA project for the course *Interactions* (J. Randon-Furling). Includes the full
theory write-up (LaTeX report) and self-contained Python code that reproduces every
figure.

> **Authors:** Marcos Lahoz and Romain Hû.

## What the project does

- Derives the **exact posterior** of the binary interaction matrix `W` from partial
  observation of the spike trains, and how it evolves between and at jumps.
- Proves **posterior consistency** with an explicit non-asymptotic, CLT-based
  convergence rate in a tractable two-neuron case, plus a column-wise consistency
  argument for the general `n`-neuron case.
- **Factorises** the posterior along incoming edges, cutting the combinatorial cost
  from `2^{n(n-1)}` to `n·2^{n-1}`.
- Extends the method to **continuous-valued interactions** with a bootstrap
  **Sequential Monte Carlo** particle filter.
- Validates the theory on simulations (posterior recovery, convergence rate,
  Frobenius error vs. time).

The full derivations and results are in [`paper/main.pdf`](paper/main.pdf).

## Results at a glance

Every figure of the report is reproduced by one script in `code/experiments/`
(all CPU-only; runtimes on a laptop):

| Experiment | Setup | Key result | Runtime |
|---|---|---|:--:|
| Posterior recovery, n = 2 (fig 1) | 20 seeded runs | Posterior mass concentrates on the true `J*` by t ≈ 100 | ~5 s |
| Convergence rate (fig 2) | N = 40 replicates/hypothesis, KL from 5000 trajectories, ε = 0.1 | Empirical rate matches the CLT-based bound (drift μ̂ ≈ 0.074, σ̂² ≈ 0.103 → discrimination scale ≈ 30 spikes) | ~15 s |
| Structure recovery, n = 5 (fig 3) | `W* ~ Ber(0.4)`, T = 1500, factorised posterior | Final Frobenius error ‖Ŵ−W*‖_F ≈ 0.41 | ~30 s |
| Error scaling (fig 4) | n ∈ {3, 5, 7}, 3 replicates | Error decays as ~1/√t, consistent with the theory | ~3 min |
| SMC, continuous interactions (fig 5) | M = 2000 particles, T = 500 | Filter recovers `J* ∈ {0, 1, 2}` from a `N(0.5, 1)` prior | ~20 s |

The factorised posterior cuts the cost from `2^{n(n-1)}` to `n·2^{n-1}` hypotheses —
this is what makes the n = 5–7 experiments tractable at all.

## Repository layout

```
.
├── README.md  LICENSE  requirements.txt  .gitignore
├── code/
│   ├── simulators.py            # IPS simulation + exact factorised filter (n neurons)
│   ├── smc.py                   # bootstrap SMC for a continuous interaction parameter
│   ├── experiments/             # one script per figure (fig1–fig5)
│   ├── figures/                 # generated figure PDFs (used by the report)
│   └── README.md                # detailed run instructions and runtimes
└── paper/
    ├── main.tex                 # LaTeX source
    ├── references.bib
    └── main.pdf                 # compiled report
```

## Quick start

```bash
pip install -r requirements.txt
cd code
python experiments/fig1_posterior_n2.py     # ... fig2 ... fig5
```

Figures are written to `code/figures/`. See [`code/README.md`](code/README.md) for
per-figure runtimes and caching flags, and for how to rebuild the report.

## License

Released under the MIT License — see `LICENSE`.
