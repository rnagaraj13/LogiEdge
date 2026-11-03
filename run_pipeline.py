#!/usr/bin/env python3
"""Run the offline LogiEdge experiments in dependency order."""

from __future__ import annotations

import argparse
import importlib.util
import os
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REQUIRED_MODULES = (
    "numpy", "sklearn", "tensorflow", "tensorflow_model_optimization",
    "paho", "psutil", "matplotlib",
)


def pipeline_steps(variant):
    model = {
        "m2": "training/models/m2_int8.tflite",
        "m3": "training/models/m3_pruned_int8.tflite",
    }[variant]
    return [
        ("tests", ["-m", "unittest", "discover", "-s", "tests", "-v"]),
        ("dataset", ["-m", "data_pipeline.generate_dataset"]),
        ("train", ["-m", "training.train_model"]),
        ("ptq", ["-m", "training.convert_ptq"]),
        ("prune", ["-m", "training.prune_quantise"]),
        ("benchmark", ["-m", "optimisation.benchmark"]),
        ("reference", ["-m", "monitoring.build_reference", "--model", model]),
        ("stage", ["-m", "training.export_deployment_artifacts", "--variant", variant]),
        ("ota-cost", ["-m", "deployment.ota_cost"]),
    ]


def check_dependencies():
    if sys.version_info[:2] != (3, 11):
        raise RuntimeError(
            "The pinned environment requires Python 3.11. "
            "Create a Python 3.11 virtual environment and install requirements.txt."
        )
    missing = [name for name in REQUIRED_MODULES if importlib.util.find_spec(name) is None]
    if missing:
        raise RuntimeError(
            f"Missing dependencies: {', '.join(missing)}.\n"
            f"Install them in your virtual environment: "
            f"{shlex.join([sys.executable, '-m', 'pip', 'install', '-r', str(ROOT / 'requirements.txt')])}"
        )


def run_step(name, arguments, log_dir):
    command = [sys.executable, "-u", *arguments]
    log_path = log_dir / f"{name}.log"
    print(f"\n[{name}] {shlex.join(command)}\nLog: {log_path}", flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"$ {shlex.join(command)}\n")
        log.flush()
        completed = subprocess.run(
            command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
            check=False,
        )
    if completed.returncode:
        raise RuntimeError(
            f"Step '{name}' failed (exit {completed.returncode}). "
            f"Inspect {log_path}; later steps were not executed."
        )
    print(f"[{name}] completed", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running or writing files.")
    parser.add_argument("--variant", choices=("m2", "m3"), default="m2",
                        help="Model used to build the matching confidence reference and Docker artifacts.")
    args = parser.parse_args(argv)
    steps = pipeline_steps(args.variant)
    if args.dry_run:
        print(f"Project working directory: {ROOT}")
        for name, arguments in steps:
            print(f"{name}: {shlex.join([sys.executable, '-u', *arguments])}")
        return 0
    try:
        check_dependencies()
        log_root = ROOT / "run_logs"
        log_root.mkdir(exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        log_dir = log_root / stamp
        log_dir.mkdir()
        print(f"Run logs: {log_dir}", flush=True)
        for name, arguments in steps:
            run_step(name, arguments, log_dir)
    except (RuntimeError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(
        "\nOffline experiment commands completed. Review the measured metrics; "
        "this does not certify deployment eligibility.\n"
        "Next, from the project directory:\n"
        "  docker compose build inference sync-agent\n"
        "  docker compose up -d\n"
        "  python -m monitoring.drift_monitor\n"
        "In a separate terminal, run the simulator while monitoring is active.\n"
        "Docker/OTA cache, offline replay, Ansible, report, and your own video "
        "still need live verification. See demo/demo_checklist.md."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
