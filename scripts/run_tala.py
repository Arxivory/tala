"""Run the unified TALA assessment pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tala.pipeline import run_tala_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run TALA's unified biomechanics, assessment, "
            "baseline, and RAG pipeline."
        )
    )

    parser.add_argument(
        "--question",
        required=True,
        help="Question for the evidence-grounded RAG explanation.",
    )

    parser.add_argument(
        "--checkin",
        type=Path,
        default=None,
        help="Optional JSON file containing the daily wellness check-in.",
    )

    parser.add_argument(
        "--baseline-scope",
        type=str,
        default=None,
        help=(
            "Optional stable identifier separating baseline histories. "
            "Use the same value for sessions belonging to the same profile."
        ),
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of evidence passages to retrieve.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "results" / "tala_output.json",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    daily_checkin = None

    if args.checkin is not None:
        if not args.checkin.is_file():
            raise FileNotFoundError(
                f"Check-in file not found: {args.checkin}"
            )

        with args.checkin.open("r", encoding="utf-8") as file:
            daily_checkin = json.load(file)

    print("=" * 68)
    print("TALA - UNIFIED ASSESSMENT PIPELINE")
    print("=" * 68)

    result = run_tala_pipeline(
        question=args.question,
        daily_checkin=daily_checkin,
        baseline_scope=args.baseline_scope,
        output_path=args.output,
        top_k=args.top_k,
    )

    assessment = result["assessment"]
    movement = assessment["movement_analysis"]
    baseline = assessment["baseline_comparison"]
    rag = result["rag"]

    print("\nASSESSMENT")
    print("-" * 68)
    print(f"Session: {result['session_id']}")
    print(f"Exercise: {movement['exercise']}")
    print(
        f"Repetitions: "
        f"{movement['repetitions']['count']}"
    )
    print(
        f"Completeness: "
        f"{movement['completeness']['status']}"
    )

    print("\nBASELINE")
    print("-" * 68)
    print(f"Status: {baseline['status']}")

    for name, comparison in baseline["metrics"].items():
        if comparison["status"] == "available":
            print(
                f"{name}: "
                f"current={comparison['current_value']:.4f}, "
                f"reference median={comparison['reference_median']:.4f}, "
                f"difference={comparison['difference_percent']:.2f}%"
            )
        else:
            print(f"{name}: no previous reference available")

    print("\nRAG")
    print("-" * 68)
    print(f"Status: {rag['status']}")

    if rag["answer"]:
        print(rag["answer"])

    print("\nOUTPUT")
    print("-" * 68)
    print(args.output.resolve())
    print("\nTALA pipeline completed.")


if __name__ == "__main__":
    main()