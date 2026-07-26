"""Lightweight regression tests for the adopted current-model package."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.verify_reproduction import (  # noqa: E402
    load_expected,
    verify_input_manifest,
    verify_portable_inputs,
    verify_saved_metrics,
    verify_spatial_analysis,
)
from libs import diketone_metrics  # noqa: E402
from libs.current_model_support.conformer_helpers import (  # noqa: E402
    discover_conformer_logs,
)


class ReproducibilityTests(unittest.TestCase):
    """Check immutable inputs and paper-level saved metrics without refitting."""

    @classmethod
    def setUpClass(cls) -> None:
        """Load the immutable expected-value contract once for this class."""
        cls.expected = load_expected(ROOT)

    def test_frozen_input_manifest(self) -> None:
        """Require at least one checksum-verified frozen input file."""
        result = verify_input_manifest(ROOT)
        self.assertGreater(result["manifest_files"], 0)
        self.assertGreater(result["manifest_bytes"], 0)

    def test_portable_inputs_build_321_features(self) -> None:
        """Require the documented portable matrix dimensions."""
        result = verify_portable_inputs(ROOT, self.expected)
        self.assertEqual(result["training_rows"], 83)
        self.assertEqual(result["combined_rows"], 161)
        self.assertEqual(result["features"], 321)

    def test_saved_summary_and_outer_metrics(self) -> None:
        """Match saved nested-LOOCV metrics to the immutable contract."""
        result = verify_saved_metrics(ROOT, self.expected)
        outer = self.expected["outer_predictions"]
        self.assertAlmostEqual(result["outer_r2"], outer["r2"], places=9)
        self.assertAlmostEqual(result["outer_rmse"], outer["rmse_kcal_mol"], places=9)
        self.assertAlmostEqual(result["outer_mae"], outer["mae_kcal_mol"], places=9)

    def test_spatial_analysis_is_portable_and_complete(self) -> None:
        """Require the complete spatial audit and six final figures."""
        result = verify_spatial_analysis(ROOT, self.expected)
        self.assertEqual(result["features"], 321)
        self.assertEqual(result["spatial_grids"], 315)
        self.assertEqual(result["figures"], 6)

    def test_gaussian_example_contains_inputs_only(self) -> None:
        """Keep the Gaussian example small and free of generated artefacts."""
        example_dir = ROOT / "examples" / "gaussian"
        with (example_dir / "expected_results.json").open(encoding="utf-8") as handle:
            expected = json.load(handle)
        for name in expected["examples"]:
            path = example_dir / name
            self.assertTrue(path.is_file())
            self.assertLess(path.stat().st_size, 10_000)
            text = path.read_text(encoding="utf-8")
            self.assertIn("\n0 1\n", text)
            self.assertIn("# ", text)
        forbidden = {".log", ".out", ".chk", ".fchk", ".cube", ".rwf"}
        self.assertFalse(
            [path for path in example_dir.rglob("*") if path.suffix.lower() in forbidden]
        )

    def test_partial_gaussian_logs_are_not_conformers(self) -> None:
        """Discover only exact opt<digits>.log files, ordered by conformer ID."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("opt10.log", "opt2.log", "opt2_optimization_partial.log"):
                (root / name).touch()
            self.assertEqual(
                [path.name for path in discover_conformer_logs(root)],
                ["opt2.log", "opt10.log"],
            )

    def test_diketone_equal_rate_limit_is_finite(self) -> None:
        """Require finite canonical network results for twelve equal barriers."""
        peak, final = diketone_metrics.simulate_barrier_network(
            barriers=np.zeros(len(diketone_metrics.ENTRY_ORDER)),
            temperature=298.15,
        )
        self.assertTrue(np.isfinite(list(peak.values())).all())
        self.assertTrue(np.isfinite(list(final.values())).all())
        self.assertAlmostEqual(sum(final.values()), 100.0, places=10)


if __name__ == "__main__":
    unittest.main()
