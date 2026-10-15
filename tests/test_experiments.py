import json
from pathlib import Path
import contextlib
import io
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from algorithms import strong_quadratic, regularized_logistic
from problems import generate_spd_matrix_fixed_max
from run_experiment import EXPERIMENTS

def invoke(main, arguments):
    with patch.object(sys, "argv", ["test", *arguments]):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return main()


class ExperimentTests(unittest.TestCase):
    def test_prescribed_quadratic_spectrum(self):
        np.random.seed(42)
        matrix = generate_spd_matrix_fixed_max(8, 1000, 500)
        spectrum = np.linalg.eigvalsh(matrix)
        self.assertAlmostEqual(spectrum[0], 0.5)
        self.assertAlmostEqual(spectrum[-1], 500)
        np.testing.assert_allclose(matrix, matrix.T, atol=1e-12)

    def test_logistic_gradient(self):
        rng = np.random.default_rng(0)
        matrix = rng.normal(size=(10, 4))
        labels = np.where(rng.normal(size=10) > 0, 1.0, -1.0)
        methods = regularized_logistic(matrix, labels, 0.1, 4)
        x = rng.normal(size=4)
        estimates = []
        for i in range(4):
            delta = np.eye(4)[i] * 1e-6
            estimates.append((methods.loss(x + delta) - methods.loss(x - delta)) / 2e-6)
        np.testing.assert_allclose(methods.gradient(x), estimates, rtol=1e-5, atol=1e-7)

    def test_quadratic_gd_descent(self):
        methods = strong_quadratic(np.diag([1.0, 5.0]), 2)
        x, curve = methods.gd(np.ones(2), 10, 0.1)
        self.assertLess(methods.loss(x), methods.loss(np.ones(2)))
        self.assertTrue(np.all(np.diff(curve) <= 0))

    def test_all_cli_experiments(self):
        from run_experiment import main
        with tempfile.TemporaryDirectory() as temporary:
            for name in EXPERIMENTS:
                output = Path(temporary) / name
                command = [name, "--dimension", "8", "--iterations", "20", "--trials", "2", "--output-dir", str(output)]
                invoke(main, command)
                result = json.loads((output / "results.json").read_text())
                self.assertEqual(result["configuration"]["dimension"], 8)
                self.assertEqual(result["results"]["num_iters"], 20)
                self.assertTrue(list(output.glob("*.pdf")))
                if name == "quadratic_misspecified":
                    self.assertEqual(len(result["results"]["comparisons"]), 3)
                with self.assertRaises(SystemExit):
                    invoke(main, command)

    def test_smoothness_override_changes_problem(self):
        from experiments.quadratic_convex import run
        from run_experiment import ExperimentConfig
        import matplotlib.pyplot as plt
        result = run(ExperimentConfig(dimension=8, iterations=20, trials=1, smoothness=5000, step_size=0.0002))
        self.assertEqual(result["L"], 5000)
        plt.close("all")


if __name__ == "__main__":
    unittest.main()
