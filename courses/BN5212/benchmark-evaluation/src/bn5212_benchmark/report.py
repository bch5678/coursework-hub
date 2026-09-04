"""Self-contained offline HTML reports for one run and multi-model comparisons."""
from __future__ import annotations

import base64
import html
from pathlib import Path
from typing import Any


METRIC_LABELS = {
    "auroc": "AUROC",
    "auprc": "AUPRC",
    "accuracy": "Accuracy",
    "precision": "Precision",
    "recall": "Recall",
    "sensitivity": "Sensitivity",
    "specificity": "Specificity",
    "f1": "F1",
    "brier": "Brier score",
}


def _format(value: Any, digits: int = 3) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _image_uri(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _page(title: str, body: str) -> str:
    safe_title = html.escape(title)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{safe_title}</title>
<style>
:root {{ color-scheme: light; --ink:#172033; --muted:#667085; --line:#d8dee9; --accent:#1769aa; --soft:#f4f7fb; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:#eef2f7; color:var(--ink); font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }}
main {{ max-width:1180px; margin:0 auto; padding:32px 24px 64px; }}
h1 {{ margin:0 0 6px; font-size:32px; }}
h2 {{ margin:34px 0 14px; font-size:21px; }}
p {{ margin:6px 0; }}
.muted {{ color:var(--muted); }}
.panel {{ background:white; border:1px solid var(--line); border-radius:12px; padding:20px; margin-top:16px; box-shadow:0 2px 8px rgba(23,32,51,.04); }}
.cards {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:12px; margin-top:18px; }}
.card {{ background:white; border:1px solid var(--line); border-radius:10px; padding:16px; }}
.card .name {{ color:var(--muted); font-size:13px; }}
.card .value {{ font-size:25px; font-weight:700; margin-top:3px; }}
.card .ci {{ color:var(--muted); font-size:12px; }}
table {{ border-collapse:collapse; width:100%; background:white; }}
th,td {{ border-bottom:1px solid var(--line); padding:9px 10px; text-align:left; vertical-align:top; }}
th {{ background:var(--soft); font-weight:650; }}
.plots {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(400px,1fr)); gap:16px; }}
.plot {{ background:white; border:1px solid var(--line); border-radius:12px; padding:12px; }}
.plot img {{ display:block; width:100%; height:auto; }}
code {{ background:var(--soft); padding:2px 5px; border-radius:4px; overflow-wrap:anywhere; }}
.warning {{ border-left:4px solid #d97706; padding:8px 12px; background:#fff7ed; margin:8px 0; }}
a {{ color:var(--accent); }}
details {{ margin-top:8px; }}
@media (max-width:640px) {{ main {{ padding:20px 12px 40px; }} .plots {{ grid-template-columns:1fr; }} }}
</style>
</head>
<body><main>{body}</main></body>
</html>
"""


def write_run_report(
    output_path: str | Path,
    metrics: dict[str, Any],
    manifest: dict[str, Any],
    plot_paths: dict[str, Path],
) -> None:
    model = metrics["model"]
    test = metrics["splits"]["test"]
    intervals = metrics.get("test_patient_bootstrap", {})
    card_names = ["auroc", "auprc", "sensitivity", "specificity", "f1", "brier"]
    cards = []
    for name in card_names:
        interval = intervals.get(name, {})
        ci = ""
        if interval.get("lower") is not None:
            ci = f"<div class='ci'>95% CI {_format(interval['lower'])}–{_format(interval['upper'])}</div>"
        cards.append(
            f"<div class='card'><div class='name'>{METRIC_LABELS[name]}</div>"
            f"<div class='value'>{_format(test.get(name))}</div>{ci}</div>"
        )

    rows = []
    for name, label in METRIC_LABELS.items():
        interval = intervals.get(name, {})
        ci = "N/A"
        if interval.get("lower") is not None:
            ci = f"{_format(interval['lower'])}–{_format(interval['upper'])}"
        rows.append(
            f"<tr><td>{label}</td><td>{_format(metrics['splits']['val'].get(name))}</td>"
            f"<td>{_format(test.get(name))}</td><td>{ci}</td></tr>"
        )

    warnings = metrics.get("warnings", [])
    warning_html = "".join(f"<div class='warning'>{html.escape(str(item))}</div>" for item in warnings)
    if not warning_html:
        warning_html = "<p class='muted'>No evaluation warnings.</p>"
    plots = "".join(
        f"<div class='plot'><img alt='{html.escape(name)}' src='{_image_uri(path)}'></div>"
        for name, path in plot_paths.items()
    )
    title = f"{model['name']} {model['version']} evaluation"
    body = f"""
<h1>{html.escape(title)}</h1>
<p class="muted">BN5212 in-hospital mortality benchmark · generated {html.escape(manifest['created_utc'])}</p>
<div class="cards">{''.join(cards)}</div>
<h2>Evaluation setup</h2>
<div class="panel">
<table><tbody>
<tr><th>Dataset task</th><td>{html.escape(str(metrics['dataset']['task']))}</td></tr>
<tr><th>Dataset index SHA-256</th><td><code>{html.escape(metrics['dataset']['index_sha256'])}</code></td></tr>
<tr><th>Evaluation unit</th><td>{html.escape(metrics['evaluation_unit'])}</td></tr>
<tr><th>Validation threshold</th><td>{_format(metrics['threshold_selection']['threshold'])} using {html.escape(metrics['threshold_selection']['method'])}</td></tr>
<tr><th>Test observations</th><td>{test['n']} {html.escape(metrics['evaluation_unit'])} records, {test['positives']} positive</td></tr>
<tr><th>Checkpoint SHA-256</th><td><code>{html.escape(str(manifest.get('checkpoint_sha256') or 'not supplied'))}</code></td></tr>
</tbody></table>
</div>
<h2>Metrics</h2>
<div class="panel"><table><thead><tr><th>Metric</th><th>Validation</th><th>Test</th><th>Test 95% patient bootstrap CI</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table></div>
<h2>Test-set plots</h2>
<div class="plots">{plots}</div>
<h2>Warnings</h2>
<div class="panel">{warning_html}</div>
<h2>Reproducibility</h2>
<div class="panel"><table><tbody>
<tr><th>Evaluation Git commit</th><td><code>{html.escape(str(manifest.get('git_commit') or 'unavailable'))}</code></td></tr>
<tr><th>Validation predictions</th><td><code>{html.escape(manifest['inputs']['val_predictions']['sha256'])}</code></td></tr>
<tr><th>Test predictions</th><td><code>{html.escape(manifest['inputs']['test_predictions']['sha256'])}</code></td></tr>
<tr><th>Bootstrap</th><td>{metrics['bootstrap']['samples']} replicates, seed {metrics['bootstrap']['seed']}</td></tr>
<tr><th>Artifacts</th><td><a href="predictions.csv">predictions.csv</a> · <a href="metrics.json">metrics.json</a> · <a href="run_manifest.json">run_manifest.json</a></td></tr>
</tbody></table></div>
"""
    Path(output_path).write_text(_page(title, body), encoding="utf-8")


def write_leaderboard_report(output_path: str | Path, rows: list[dict[str, Any]]) -> None:
    table_rows = []
    for row in rows:
        table_rows.append(
            "<tr>"
            f"<td>{html.escape(str(row['model']))}</td>"
            f"<td>{html.escape(str(row['version']))}</td>"
            f"<td>{_format(row.get('auroc'))}</td>"
            f"<td>{_format(row.get('auprc'))}</td>"
            f"<td>{_format(row.get('sensitivity'))}</td>"
            f"<td>{_format(row.get('specificity'))}</td>"
            f"<td>{_format(row.get('f1'))}</td>"
            f"<td>{row.get('n', 'N/A')}</td>"
            "</tr>"
        )
    body = f"""
<h1>BN5212 benchmark leaderboard</h1>
<p class="muted">Models evaluated through the same benchmark result contract.</p>
<div class="panel"><table><thead><tr><th>Model</th><th>Version</th><th>AUROC</th><th>AUPRC</th><th>Sensitivity</th><th>Specificity</th><th>F1</th><th>N</th></tr></thead>
<tbody>{''.join(table_rows)}</tbody></table></div>
"""
    Path(output_path).write_text(_page("BN5212 benchmark leaderboard", body), encoding="utf-8")
