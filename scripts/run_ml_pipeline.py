#!/usr/bin/env python3
"""Run the ML pipeline notebooks in order.

Usage:
    # Run full pipeline
    python3 scripts/run_ml_pipeline.py

    # Resume from step 3 (1.0 feature_engineering)
    python3 scripts/run_ml_pipeline.py 2

    # List notebooks without running
    python3 scripts/run_ml_pipeline.py --list
"""

import papermill as pm
from pathlib import Path
import sys
from datetime import datetime
import argparse

NOTEBOOKS_DIR = Path(__file__).parent.parent / "notebooks"
OUTPUT_DIR = Path(__file__).parent.parent / "notebooks" / "runs"

# ML pipeline notebooks in execution order
PIPELINE = [
    "0.2 historical_earnings_moves.ipynb",      # Fetch/compute earnings moves
    "0.2c_infer_earnings_timing.ipynb",         # Infer BMO/AMC timing
    "1.0 feature_engineering.ipynb",            # Build ML features
    "1.1 model_training.ipynb",                 # Train quantile models
    "1.2 calibration_analysis.ipynb",           # Analyze calibration
    "1.3 portfolio_simulation.ipynb",           # Portfolio simulation backtest
    "1.4 kelly_position_sizing.ipynb",          # Kelly criterion analysis
]


def list_pipeline():
    """List all notebooks in the pipeline."""
    print("ML Pipeline Notebooks:")
    print("-" * 50)
    for i, notebook in enumerate(PIPELINE):
        print(f"  {i}: {notebook}")
    print("-" * 50)
    print(f"Total: {len(PIPELINE)} notebooks")


def run_pipeline(start_from: int = 0, dry_run: bool = False, use_cache: bool = True):
    """Run the ML pipeline notebooks in order.

    Args:
        start_from: Index of notebook to start from (0-indexed)
        dry_run: If True, only print what would be run
        use_cache: If True, use cached data; if False, refetch from APIs
    """
    if start_from >= len(PIPELINE):
        print(f"Error: start_from ({start_from}) >= number of notebooks ({len(PIPELINE)})")
        sys.exit(1)

    OUTPUT_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = OUTPUT_DIR / timestamp
    run_dir.mkdir(exist_ok=True)

    print(f"ML Pipeline Runner")
    print(f"=" * 60)
    print(f"Start from: step {start_from} ({PIPELINE[start_from]})")
    print(f"Output dir: {run_dir}")
    print(f"Timestamp:  {timestamp}")
    print(f"Use cache:  {use_cache}")
    if dry_run:
        print(f"Mode:       DRY RUN (no execution)")
    print(f"=" * 60)

    for i, notebook in enumerate(PIPELINE[start_from:], start=start_from):
        input_path = NOTEBOOKS_DIR / notebook
        output_path = run_dir / notebook

        print(f"\n{'=' * 60}")
        print(f"[{i + 1}/{len(PIPELINE)}] {notebook}")
        print(f"{'=' * 60}")
        print(f"  Input:  {input_path}")
        print(f"  Output: {output_path}")

        if not input_path.exists():
            print(f"  ERROR: Input notebook not found!")
            sys.exit(1)

        if dry_run:
            print(f"  [DRY RUN] Would execute notebook")
            continue

        try:
            pm.execute_notebook(
                str(input_path),
                str(output_path),
                kernel_name="python3",
                cwd=str(NOTEBOOKS_DIR),  # Set working directory for relative paths
                parameters={"USE_CACHE": use_cache},
            )
            print(f"  ✓ Completed successfully")
        except pm.PapermillExecutionError as e:
            print(f"  ✗ FAILED at cell {e.exec_count}")
            print(f"    Error: {e.ename}: {e.evalue}")
            print(f"    See output notebook for details: {output_path}")
            sys.exit(1)
        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            sys.exit(1)

    print(f"\n{'=' * 60}")
    if dry_run:
        print("DRY RUN complete - no notebooks were executed")
    else:
        print("Pipeline complete!")
        print(f"Outputs saved to: {run_dir}")
    print(f"{'=' * 60}")


def main():
    parser = argparse.ArgumentParser(
        description="Run ML pipeline notebooks in order",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 scripts/run_ml_pipeline.py           # Run full pipeline (uses cache)
  python3 scripts/run_ml_pipeline.py 2         # Start from step 2 (1.0 feature_engineering)
  python3 scripts/run_ml_pipeline.py --refetch # Refetch data from APIs (slow)
  python3 scripts/run_ml_pipeline.py --list    # List notebooks
  python3 scripts/run_ml_pipeline.py --dry-run # Show what would run
        """
    )
    parser.add_argument(
        "start_from",
        type=int,
        nargs="?",
        default=0,
        help="Step index to start from (0-indexed, default: 0)"
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List notebooks in pipeline and exit"
    )
    parser.add_argument(
        "--dry-run", "-n",
        action="store_true",
        help="Print what would be run without executing"
    )
    parser.add_argument(
        "--refetch",
        action="store_true",
        help="Refetch data from APIs instead of using cache"
    )

    args = parser.parse_args()

    if args.list:
        list_pipeline()
    else:
        run_pipeline(
            start_from=args.start_from,
            dry_run=args.dry_run,
            use_cache=not args.refetch,
        )


if __name__ == "__main__":
    main()
