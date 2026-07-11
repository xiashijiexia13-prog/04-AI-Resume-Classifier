"""Run the reproducible data-to-model pipeline with one command."""

import argparse
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent


def run_step(script: str) -> None:
    """Run one pipeline stage and stop immediately if it fails."""
    print(f"\n[pipeline] Running {script}", flush=True)
    subprocess.run([sys.executable, str(ROOT / script)], cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-analysis",
        action="store_true",
        help="Skip EDA chart generation when only retraining the model.",
    )
    args = parser.parse_args()

    if not (ROOT / "data" / "resume_dataset.csv").exists():
        run_step("download_dataset.py")

    run_step("data_cleaning.py")
    if not args.skip_analysis:
        run_step("data_analysis.py")
    run_step("feature_engineering.py")
    run_step("train.py")
    print("\n[pipeline] Complete", flush=True)


if __name__ == "__main__":
    main()
