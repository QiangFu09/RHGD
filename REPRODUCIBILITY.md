# Reproducibility notes

## Problems and methods

The quadratic objective is xᵀAx/2. The matrix is generated from a Gaussian QR
factorization and a prescribed spectrum, with dimension 100 and L=500 by
default. The strong case uses alpha=L/kappa; the convex case includes a zero
eigenvalue. The misspecification experiment uses assumed alpha values 0.01, 0.1,
and 1 while retaining the same true matrix.

Logistic regression uses 500 synthetic examples with noisy labels and dimension
100 by default. The strong variant adds L2 regularization, with default 1e-4.
The methods retain their released adaptive stepsize rules. See Section 5 and
Appendix H of the paper for experimental settings and tuning grids.

The problem seed is 42. Each stochastic trial constructs `default_rng(seed=i)`
for i=0,...,4. Methods consume that generator sequentially in the released order;
they are not independently reseeded within a trial. `--trials` changes the
number of trials. Changing dimension, condition number, or run length defines
a different experiment and may require retuning the stepsize.

## Saved curves and plotting

The returned loss arrays retain the original before/after-update conventions.
The comparison plots replace the first displayed point with the initial loss.
Raw trial arrays are also saved in `results.json`.

For stochastic methods, the line is the arithmetic mean. The envelope is formed
in log10 space using std(loss)/(mean(loss) ln(10)), with NumPy's population
standard deviation (ddof=0), and transformed back to the original scale. This
delta-method envelope is not a confidence interval.

For regularized logistic regression, f_star is the minimum observed loss across
the compared methods and trials. Shifted losses are clipped below at 1e-20
before log aggregation. The reference is an empirical estimate and depends on
the chosen run budget; it is not an independently computed optimum.

## Additional refresh diagnostics

`refresh_strong` retains the constant-rate reference and the released refresh
probability sqrt(alpha) h. `refresh_convex` retains probability 17/(2k+18) and
the original thinning calculation for its reference process.

The latter uses a fixed proposal rate of 1, while its target intensity
17/(2t+18h) can exceed 1. Consequently the historical thinning plot is a
diagnostic illustration, not an exact simulation of that inhomogeneous Poisson
process in all parameter regimes. It is separate from the main method
comparisons. Correcting the thinning bound would change that reference plot.

## Code organization and validation

The original notebook has been replaced by ordinary Python modules. Numerical
functions are bound to their problem data by the factories in `algorithms.py`;
the experiments no longer rely on notebook execution state. `run_experiment.py`
adds configurable budgets, saved numerical results, and noninteractive PDF
output. It does not launch a grid search or retune stepsizes automatically.

Short CPU checks use Python 3.12.3, NumPy 2.1.2, and Matplotlib 3.10.1;
`requirements-tested.txt` records the direct dependency versions. The paper
reports Python 3.10.12. Full paper experiments have not been rerun as part of
the code reorganization.
