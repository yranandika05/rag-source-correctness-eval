import json
from datetime import datetime, timezone
from html import escape
from pathlib import Path

import pandas as pd


RESULTS_DIR = Path("results")


def create_run_directory(
    results_dir: str | Path = RESULTS_DIR,
    run_name: str | None = None,
    overwrite: bool = False,
) -> Path:
    """Create and return a run-specific results directory."""
    base_dir = Path(results_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    if run_name:
        run_dir = base_dir / run_name
        if run_dir.exists() and not overwrite:
            raise FileExistsError(
                f"Run directory already exists: {run_dir}. "
                "Choose a different --run-name or pass --overwrite-run."
            )
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    run_number = 1
    while True:
        run_dir = base_dir / f"run_{run_number:03d}"
        if not run_dir.exists():
            run_dir.mkdir(parents=True)
            return run_dir
        run_number += 1


def save_config(path: str | Path, config: dict) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(config, file, indent=2, sort_keys=True)
        file.write("\n")


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def source_distribution(rows: pd.DataFrame) -> str:
    if rows.empty:
        return ""

    counts = rows["retrieved_source"].fillna("").value_counts().sort_index()
    total = int(counts.sum())
    parts = []
    for source, count in counts.items():
        share = count / total if total else 0
        parts.append(f"{escape(str(source))}: {count} ({share:.0%})")
    return ", ".join(parts)


def generate_question_view_html(results_path: str | Path, output_path: str | Path, top_k: int) -> None:
    results = pd.read_csv(results_path)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    method_order = ["bm25", "dense", "hybrid", "metadata_aware"]
    methods = [method for method in method_order if method in set(results["method"])]
    methods.extend(sorted(set(results["method"]) - set(methods)))

    question_columns = ["question_id", "question", "intended_source", "category"]
    questions = results[question_columns].drop_duplicates().sort_values("question_id")

    html_parts = [
        "<!doctype html>",
        "<html lang=\"en\">",
        "<head>",
        "<meta charset=\"utf-8\">",
        "<title>RAG Source Correctness Question View</title>",
        "<style>",
        "body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 32px; color: #1f2937; background: #f8fafc; }",
        "h1 { margin-bottom: 4px; }",
        ".summary { color: #475569; margin-bottom: 24px; }",
        ".question { background: #ffffff; border: 1px solid #d9e2ec; border-radius: 8px; margin: 0 0 24px; padding: 20px; }",
        ".question-meta { color: #475569; font-size: 14px; margin-bottom: 12px; }",
        ".method { margin-top: 18px; }",
        ".method h3 { margin-bottom: 8px; }",
        ".distribution { color: #475569; font-size: 14px; margin: -2px 0 8px; }",
        "table { width: 100%; border-collapse: collapse; font-size: 14px; table-layout: fixed; }",
        "th, td { border-top: 1px solid #e2e8f0; padding: 8px; vertical-align: top; text-align: left; word-wrap: break-word; }",
        "th { color: #334155; background: #f1f5f9; }",
        ".rank { width: 48px; }",
        ".source { width: 120px; }",
        ".score { width: 100px; }",
        ".chunk { width: 210px; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12px; }",
        ".wrong { background: #fff1f2; }",
        ".badge { display: inline-block; margin-left: 8px; padding: 2px 6px; border-radius: 4px; background: #dc2626; color: #fff; font-size: 12px; font-weight: 700; }",
        ".preview { white-space: normal; }",
        "</style>",
        "</head>",
        "<body>",
        "<h1>Question-Level Retrieval View</h1>",
        f"<div class=\"summary\">Questions: {len(questions)} | Top-k: {top_k}</div>",
    ]

    for question in questions.to_dict(orient="records"):
        question_id = question["question_id"]
        intended_source = str(question["intended_source"])
        question_rows = results[results["question_id"] == question_id]

        html_parts.extend(
            [
                "<section class=\"question\">",
                f"<h2>Q{escape(str(question_id))}: {escape(str(question['question']))}</h2>",
                "<div class=\"question-meta\">"
                f"Intended source: <strong>{escape(intended_source)}</strong> | "
                f"Category: {escape(str(question['category']))}"
                "</div>",
            ]
        )

        for method in methods:
            method_rows = question_rows[question_rows["method"] == method].sort_values("rank").head(top_k)
            if method_rows.empty:
                continue

            html_parts.append("<div class=\"method\">")
            html_parts.append(f"<h3>{escape(method)}</h3>")
            if intended_source == "Ambiguous":
                html_parts.append(
                    f"<div class=\"distribution\">Source distribution: {source_distribution(method_rows)}</div>"
                )
            html_parts.extend(
                [
                    "<table>",
                    "<thead><tr>",
                    "<th class=\"rank\">Rank</th>",
                    "<th class=\"source\">Source</th>",
                    "<th class=\"score\">Score</th>",
                    "<th class=\"chunk\">Chunk ID</th>",
                    "<th>Text Preview</th>",
                    "</tr></thead>",
                    "<tbody>",
                ]
            )

            for row in method_rows.to_dict(orient="records"):
                retrieved_source = str(row.get("retrieved_source", ""))
                is_wrong = intended_source != "Ambiguous" and retrieved_source != intended_source
                row_class = " class=\"wrong\"" if is_wrong else ""
                badge = " <span class=\"badge\">WRONG SOURCE</span>" if is_wrong else ""
                score = row.get("score", "")
                score_text = "" if pd.isna(score) else f"{float(score):.4f}"
                html_parts.extend(
                    [
                        f"<tr{row_class}>",
                        f"<td>{escape(str(row.get('rank', '')))}</td>",
                        f"<td>{escape(retrieved_source)}{badge}</td>",
                        f"<td>{escape(score_text)}</td>",
                        f"<td class=\"chunk\">{escape(str(row.get('chunk_id', '')))}</td>",
                        f"<td class=\"preview\">{escape(str(row.get('text_preview', '')))}</td>",
                        "</tr>",
                    ]
                )

            html_parts.extend(["</tbody>", "</table>", "</div>"])

        html_parts.append("</section>")

    html_parts.extend(["</body>", "</html>"])
    output.write_text("\n".join(html_parts), encoding="utf-8")


def generate_question_view_md(results_path: str | Path, output_path: str | Path, top_k: int) -> None:
    results = pd.read_csv(results_path)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    method_order = ["bm25", "dense", "hybrid", "metadata_aware"]
    methods = [method for method in method_order if method in set(results["method"])]
    methods.extend(sorted(set(results["method"]) - set(methods)))

    question_columns = ["question_id", "question", "intended_source", "category"]
    questions = results[question_columns].drop_duplicates().sort_values("question_id")

    lines = ["# Question-Level Retrieval View", ""]
    for question in questions.to_dict(orient="records"):
        question_id = question["question_id"]
        intended_source = str(question["intended_source"])
        question_rows = results[results["question_id"] == question_id]
        lines.extend(
            [
                f"## Q{question_id}: {question['question']}",
                "",
                f"- intended_source: {intended_source}",
                f"- category: {question['category']}",
                "",
            ]
        )

        for method in methods:
            method_rows = question_rows[question_rows["method"] == method].sort_values("rank").head(top_k)
            if method_rows.empty:
                continue

            lines.extend([f"### {method}", ""])
            if intended_source == "Ambiguous":
                lines.extend([f"Source distribution: {source_distribution(method_rows)}", ""])

            for row in method_rows.to_dict(orient="records"):
                retrieved_source = str(row.get("retrieved_source", ""))
                is_wrong = intended_source != "Ambiguous" and retrieved_source != intended_source
                label = " WRONG SOURCE" if is_wrong else ""
                score = row.get("score", "")
                score_text = "" if pd.isna(score) else f"{float(score):.4f}"
                lines.extend(
                    [
                        f"- rank {row.get('rank', '')}: {retrieved_source}{label}",
                        f"  score: {score_text}",
                        f"  chunk_id: `{row.get('chunk_id', '')}`",
                        f"  preview: {row.get('text_preview', '')}",
                        "",
                    ]
                )

    output.write_text("\n".join(lines), encoding="utf-8")
