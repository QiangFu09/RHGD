"""Run an RHGD experiment and save its numerical results and figures."""

import argparse
from dataclasses import asdict, dataclass
import importlib
import json
import math
from pathlib import Path
import platform

EXPERIMENTS = ["quadratic_strong", "quadratic_misspecified", "quadratic_convex", "logistic_strong", "logistic_convex", "refresh_strong", "refresh_convex"]


@dataclass
class ExperimentConfig:
    dimension: int = 100
    iterations: int | None = None
    trials: int = 5
    kappa: float = 1e7
    smoothness: float = 500.0
    step_size: float | None = None
    regularization: float = 1e-4


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", choices=EXPERIMENTS)
    parser.add_argument("--dimension", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--kappa", type=float, default=1e7)
    parser.add_argument("--smoothness", type=float, default=500.0)
    parser.add_argument("--step-size", type=float, default=None, help="Override eta (or gamma=h^2 in refresh diagnostics)")
    parser.add_argument("--regularization", type=float, default=1e-4)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if min(args.dimension, args.trials) < 1 or args.dimension < 2 or (args.iterations is not None and args.iterations < 2):
        parser.error("dimension and iterations must be at least 2; trials must be positive")
    numbers = [args.kappa, args.smoothness, args.regularization]
    if args.step_size is not None:
        numbers.append(args.step_size)
    if not all(math.isfinite(value) and value > 0 for value in numbers) or args.kappa < 1:
        parser.error("require finite positive scales and kappa >= 1")
    if args.experiment.startswith("quadratic_strong") or args.experiment in {"quadratic_misspecified", "refresh_strong"}:
        if args.kappa < args.smoothness:
            parser.error("strong quadratic settings require alpha=L/kappa <= 1")
    output = args.output_dir or Path("outputs") / args.experiment
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        parser.error(f"Output directory is not empty: {output}; choose another directory or use --overwrite")
    config = ExperimentConfig(**{key: getattr(args, key) for key in ExperimentConfig.__dataclass_fields__})
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.close("all")
    experiment = importlib.import_module(f"experiments.{args.experiment}")
    results = experiment.run(config)
    output.mkdir(parents=True, exist_ok=True)
    for number in plt.get_fignums():
        plt.figure(number).savefig(output / f"figure_{number}.pdf", bbox_inches="tight")
    plt.close("all")

    def encode(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(type(value).__name__)

    payload = {"experiment": args.experiment, "configuration": asdict(config), "versions": {"python": platform.python_version(), "numpy": np.__version__, "matplotlib": matplotlib.__version__}, "results": results}
    with (output / "results.json").open("w" if args.overwrite else "x") as handle:
        json.dump(payload, handle, indent=2, default=encode, allow_nan=False)
        handle.write("\n")
    print(f"Saved results and figures to {output}")


if __name__ == "__main__":
    main()
