# RHGD: randomized Hamiltonian gradient descent

Research code for **Hamiltonian Descent Algorithms for Optimization: Accelerated
Rates via Randomized Integration Time**, Qiang Fu and Andre Wibisono,
NeurIPS 2025.

[Paper](https://proceedings.neurips.cc/paper_files/paper/2025/hash/86a8a512b27f49519594ebe89f66d708-Abstract-Conference.html) ·
[PDF](https://proceedings.neurips.cc/paper_files/paper/2025/file/86a8a512b27f49519594ebe89f66d708-Paper-Conference.pdf)

The experiments compare RHGD with gradient descent (GD), Nesterov's accelerated
gradient descent (AGD), and continuized accelerated gradient descent (CAGD) on
quadratic minimization and synthetic logistic regression. The code uses NumPy
and Matplotlib on CPU.

## Installation

Use Python 3.10 or later. From the repository root:

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements.txt
```

`requirements-tested.txt` pins the direct dependency versions used for the
validation checks.

No GPU, external dataset, or notebook environment is required.

## Run an experiment

Check the installation with a small quadratic problem:

```bash
python run_experiment.py quadratic_strong --dimension 8 \
  --iterations 20 --trials 2 --output-dir outputs/check
```

Run the main comparisons with their released defaults:

```bash
python run_experiment.py quadratic_strong
python run_experiment.py quadratic_misspecified
python run_experiment.py quadratic_convex
python run_experiment.py logistic_strong
python run_experiment.py logistic_convex
```

Each command saves PDF figures and `results.json` under `outputs/<experiment>/`.
The JSON contains raw method curves, configuration, and software versions.
The misspecification experiment records each assumed-alpha comparison.
Use `--output-dir` to keep separate configurations. Existing output directories
require `--overwrite`.

| Experiment | Setting | Main parameters |
| --- | --- | --- |
| `quadratic_strong` | Strongly convex quadratic, exact alpha | L=500, kappa=1e7, dimension 100 |
| `quadratic_misspecified` | Same quadratic, assumed alpha in {0.01, 0.1, 1} | L=500, kappa=1e7 |
| `quadratic_convex` | Quadratic with minimum eigenvalue zero | L=500, dimension 100 |
| `logistic_strong` | L2-regularized logistic regression | 500 examples, dimension 100, regularization 1e-4 |
| `logistic_convex` | Unregularized logistic regression | 500 examples, dimension 100 |

For Appendix H settings, choose the condition number, smoothness, or
regularization explicitly. For example:

```bash
python run_experiment.py quadratic_strong --kappa 1e5 \
  --output-dir outputs/quadratic_kappa1e5
python run_experiment.py quadratic_convex --smoothness 5000 \
  --step-size 0.0002 --output-dir outputs/quadratic_L5000
python run_experiment.py logistic_strong --regularization 1e-3 \
  --output-dir outputs/logistic_alpha1e3
```

Changing L does not automatically retune the stepsize. `--step-size` sets eta
for GD/AGD/CAGD and h=sqrt(eta) for RHGD. The logistic experiments retain their
adaptive stepsize rules. See Section 5 and Appendix H for the selected settings;
the runner does not perform a grid search.

`refresh_strong` and `refresh_convex` provide the additional refresh-event
diagnostics. Their reference-process conventions are described in
[REPRODUCIBILITY.md](REPRODUCIBILITY.md).

## Code organization

| File or directory | Purpose |
| --- | --- |
| `algorithms.py` | Problem-bound optimizer implementations |
| `problems.py` | Quadratic generators and iteration budgets |
| `plotting.py` | Loss summaries and refresh-event plots |
| `experiments/` | Problem setup and method comparisons |
| `run_experiment.py` | Command-line configuration, execution, and output |
| `tests/` | Numerical and command-line checks |

All experiments run as Python scripts. The original notebook is available in
the repository's Git history.

## Reproducibility

The synthetic problem seed is 42. The stochastic methods use five generator
seeds by default. Within each trial, methods consume the same generator in the
released order. Raw curves and plotting conventions are documented in
[REPRODUCIBILITY.md](REPRODUCIBILITY.md). The logistic reference optimum is
estimated from the best observed loss, rather than from a separate exact solve.

```bash
python -m unittest discover -s tests -v
```

## Citation

```bibtex
@inproceedings{fu2025hamiltonian,
  title     = {Hamiltonian Descent Algorithms for Optimization: Accelerated Rates via Randomized Integration Time},
  author    = {Fu, Qiang and Wibisono, Andre},
  booktitle = {Advances in Neural Information Processing Systems},
  volume    = {38},
  year      = {2025},
  url       = {https://proceedings.neurips.cc/paper_files/paper/2025/hash/86a8a512b27f49519594ebe89f66d708-Abstract-Conference.html}
}
```

## License

The code is available under the [MIT License](LICENSE).
