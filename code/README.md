# Code — Bayesian Filtering for Interaction-Structure Learning

Companion code for the MVA project on Bayesian filtering of the
interaction matrix of a point-process IPS (course *Interactions*,
J. Randon-Furling).

Authors: Marcos Lahoz, Romain.

## Layout

```
code/
├── simulators.py         # simulation + exact factorised filter for n neurons
├── smc.py                # bootstrap SMC for a continuous interaction parameter
├── experiments/
│   ├── fig1_posterior_n2.py      # posterior trajectories, n=2
│   ├── fig2_convergence_rate.py  # KL slope + CLT hitting-time CDF
│   ├── fig3_posterior_n5.py      # matrix recovery, n=5
│   ├── fig4_error_vs_T.py        # Frobenius error vs time, n ∈ {3,5,7}
│   └── fig5_smc_continuous.py    # SMC posterior of continuous J
├── figures/                      # generated PDFs (consumed by paper/main.tex)
└── cache/                        # npz cache for fig4 replicates
```

## Requirements

Python ≥ 3.10 with `numpy`, `matplotlib`. No other runtime dependency.

```bash
pip install numpy matplotlib
```

## Reproduce every figure

All scripts are self-contained. Run from the `code/` directory:

```bash
python experiments/fig1_posterior_n2.py
python experiments/fig2_convergence_rate.py
python experiments/fig3_posterior_n5.py
python experiments/fig4_error_vs_T.py
python experiments/fig5_smc_continuous.py
```

Figures land in `figures/` as PDFs. Expected runtimes on a laptop
(single core, NumPy only):

| Figure | Time (cold) | Time (cached) |
|-------|------------:|--------------:|
| fig1  |        ~5 s |             — |
| fig2  |       ~15 s |             — |
| fig3  |       ~30 s |             — |
| fig4  |       ~3 min|          ~2 s |
| fig5  |       ~20 s |             — |

`fig4` writes a per-seed `.npz` file under `cache/`. Subsequent runs skip the
simulation. Useful flags:

```bash
python experiments/fig4_error_vs_T.py n=5            # only run the n=5 config
python experiments/fig4_error_vs_T.py seeds=0,1      # subset of seeds
python experiments/fig4_error_vs_T.py --plot-only    # re-plot from cache
```

## Rebuilding the paper

Figures are referenced by the LaTeX source at `../paper/main.tex`. To
rebuild the PDF:

```bash
cd ../paper
pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```
