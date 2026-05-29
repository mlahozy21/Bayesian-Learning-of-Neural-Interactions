# Bayesian Learning of Neural Interactions

A Bayesian framework that learns **who influences whom** in a network of neurons —
its *interaction structure* — observing only the neurons' spike (jump) times. The
neurons are modelled as a continuous-time **point-process interacting particle system**
(coupled stochastic integrate-and-fire units), and we recover the interaction matrix
that couples them.

MVA project for the course *Interactions* (J. Randon-Furling). Includes the full
theory write-up (LaTeX paper) and self-contained Python code that reproduces every
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

## Repository layout

```
.
├── README.md  LICENSE  requirements.txt  .gitignore
├── code/
│   ├── simulators.py            # IPS simulation + exact factorised filter (n neurons)
│   ├── smc.py                   # bootstrap SMC for a continuous interaction parameter
│   ├── experiments/             # one script per figure (fig1–fig5)
│   ├── figures/                 # generated figure PDFs (used by the paper)
│   └── README.md                # detailed run instructions and runtimes
└── paper/
    ├── main.tex                 # LaTeX source
    ├── references.bib
    └── main.pdf                 # compiled paper
```

## Quick start

```bash
pip install -r requirements.txt
cd code
python experiments/fig1_posterior_n2.py     # ... fig2 ... fig5
```

Figures are written to `code/figures/`. See [`code/README.md`](code/README.md) for
per-figure runtimes and caching flags, and for how to rebuild the paper.

## License

Released under the MIT License — see `LICENSE`.
