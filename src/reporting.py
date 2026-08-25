"""Evidence-bound Markdown scientific audit report generator."""
from pathlib import Path
import json

def generate_report(context):
    """Generate statements from actual artifacts and explicitly list automated limitations."""
    q=context["quality"]; perf=context.get("performance"); lines=["# Scientific audit",f"## Dataset\n- Path: `{context['path']}`\n- Rows analyzed after invalid timestamp exclusion: {context['rows']}\n- Datetime: `{context['datetime']}`; target: `{context['target']}`\n- Dominant observed frequency: {q.get('dominant_frequency')}",f"## Data quality\n- Duplicate rows: {q['duplicate_rows']}; duplicate timestamps: {q['duplicate_timestamps']}.\n- Statistical-outlier flags: {q['statistical_outliers']}; physical-anomaly flags: {q['physical_anomalies']}.\n- Notes: {' '.join(q['notes'])}","## Leakage controls\n- Rows were chronologically split into train, validation, then a locked final test.\n- Lag and rolling target features use past observations only; rolling features apply `shift(1)`.\n- Scalers/selectors are fitted only on development/training data.","## Model availability and limitations\n"+"\n".join(f"- {x}" for x in context.get("limitations",[]))]
    if perf is not None and not perf.empty:
        best=perf.loc[perf.RMSE.idxmin()];lines.append(f"## Locked-test results\nThe lowest observed RMSE among models actually evaluated was `{best.model}` ({best.RMSE:.6g}). This descriptive ranking is not a claim of statistical significance.")
    else: lines.append("## Locked-test results\nNo model produced an evaluable locked-test prediction.")
    Path("results/reports/scientific_audit.md").write_text("\n\n".join(lines)+"\n",encoding="utf-8")
