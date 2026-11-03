import io
import tempfile
import unittest
import sys
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from run_pipeline import main, pipeline_steps, run_step


class TestRunner(unittest.TestCase):
    def test_commands_in_dependency_order(self):
        steps = pipeline_steps("m3")
        self.assertEqual(
            [name for name, _ in steps],
            ["tests", "dataset", "train", "ptq", "prune", "benchmark", "reference", "stage", "ota-cost"],
        )
        self.assertIn("training/models/m3_pruned_int8.tflite", steps[6][1])
        self.assertEqual(steps[7][1][-1], "m3")

    def test_dry_run_does_not_execute_or_check_dependencies(self):
        with patch("run_pipeline.check_dependencies") as check:
            with patch("run_pipeline.run_step") as run:
                with redirect_stdout(io.StringIO()) as output:
                    self.assertEqual(main(["--dry-run"]), 0)
        check.assert_not_called()
        run.assert_not_called()
        self.assertIn("data_pipeline.generate_dataset", output.getvalue())

    def test_failed_step_has_persistent_log(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch("run_pipeline.subprocess.run", return_value=SimpleNamespace(returncode=3)):
                with redirect_stdout(io.StringIO()):
                    with self.assertRaisesRegex(RuntimeError, "exit 3"):
                        run_step("train", ["-m", "training.train_model"], Path(directory))
            self.assertIn("training.train_model", (Path(directory) / "train.log").read_text())


if __name__ == "__main__":
    unittest.main()
