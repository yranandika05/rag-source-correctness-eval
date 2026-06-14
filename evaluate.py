from pathlib import Path

import pandas as pd


QUESTION_COLUMNS = ["question_id", "question", "intended_source", "category"]
RESULT_COLUMNS = [
    "question_id",
    "question",
    "intended_source",
    "category",
    "method",
    "rank",
    "chunk_id",
    "retrieved_source",
    "score",
    "text_preview",
]


def load_evaluation_questions(path: str | Path) -> list[dict]:
    questions = pd.read_csv(path)
    missing = set(QUESTION_COLUMNS) - set(questions.columns)
    if missing:
        raise ValueError(f"Evaluation file is missing columns: {sorted(missing)}")
    return questions[QUESTION_COLUMNS].fillna("").to_dict(orient="records")


def save_results(path: str | Path, rows: list[dict]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=RESULT_COLUMNS).to_csv(output_path, index=False)


def compute_source_metrics(results_path: str | Path, top_k: int = 5) -> pd.DataFrame:
    results = pd.read_csv(results_path)
    if results.empty:
        return pd.DataFrame()

    missing = set(RESULT_COLUMNS) - set(results.columns)
    if missing:
        raise ValueError(f"Results file is missing columns: {sorted(missing)}")

    results = results[results["intended_source"] != "Ambiguous"].copy()
    if results.empty:
        return pd.DataFrame()

    metric_rows = []
    for method, method_rows in results.groupby("method"):
        for k in range(1, top_k + 1):
            top_rows = method_rows[method_rows["rank"] <= k].copy()
            top_rows["is_correct_source"] = top_rows["retrieved_source"] == top_rows["intended_source"]

            per_question = top_rows.groupby("question_id")["is_correct_source"].any()
            source_accuracy = per_question.mean() if not per_question.empty else 0.0
            wrong_source_rate = 1.0 - top_rows["is_correct_source"].mean() if not top_rows.empty else 0.0

            metric_rows.append(
                {
                    "method": method,
                    "k": k,
                    "source_accuracy_at_k": round(float(source_accuracy), 4),
                    "wrong_source_rate_at_k": round(float(wrong_source_rate), 4),
                }
            )

    return pd.DataFrame(metric_rows).sort_values(["method", "k"])


def compute_ambiguous_source_report(results_path: str | Path, top_k: int = 5) -> pd.DataFrame:
    """Report retrieved source distribution for intentionally ambiguous queries."""
    results = pd.read_csv(results_path)
    if results.empty:
        return pd.DataFrame()

    missing = set(RESULT_COLUMNS) - set(results.columns)
    if missing:
        raise ValueError(f"Results file is missing columns: {sorted(missing)}")

    ambiguous = results[results["intended_source"] == "Ambiguous"].copy()
    if ambiguous.empty:
        return pd.DataFrame()

    rows = []
    for method, method_rows in ambiguous.groupby("method"):
        for k in range(1, top_k + 1):
            top_rows = method_rows[method_rows["rank"] <= k]
            total = len(top_rows)
            if total == 0:
                continue

            source_counts = top_rows["retrieved_source"].fillna("").value_counts().sort_index()
            for retrieved_source, count in source_counts.items():
                rows.append(
                    {
                        "method": method,
                        "k": k,
                        "retrieved_source": retrieved_source,
                        "count": int(count),
                        "share": round(float(count / total), 4),
                    }
                )

    return pd.DataFrame(rows).sort_values(["method", "k", "retrieved_source"])
