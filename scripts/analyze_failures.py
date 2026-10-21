import json
from pathlib import Path

from src.evaluation.failure_analysis import analyze_failures

BASE_DIR = Path(__file__).resolve().parents[1]
INPUT_PATH = BASE_DIR / "data" / "eval" / "baseline_results.json"
OUTPUT_PATH = BASE_DIR / "data" / "eval" / "failure_analysis.json"


def main() -> None:
    results = json.loads(INPUT_PATH.read_text(encoding="utf-8"))
    report = analyze_failures(results)

    OUTPUT_PATH.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    summary = report["summary"]

    print(f"Questions analyzed: {summary['total_answers']}")
    print(f"Questions requiring review: {summary['answers_requiring_review']}")

    print("\nPotential issue counts:")
    if summary["issue_counts"]:
        for issue, count in summary["issue_counts"].items():
            print(f"- {issue}: {count}")
    else:
        print("- No heuristic flags")

    print(f"\nReport saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
