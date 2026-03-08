from __future__ import annotations

import argparse
import html
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd


def _coerce_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    s = series.astype(str).str.strip().str.lower()
    out = s.isin({"1", "true", "yes", "y", "t"})
    return out.where(~series.isna(), False)


def _load_csv_if_exists(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def _resolve_inputs(outdir: Path) -> Dict[str, Path]:
    mapping = {
        "overall": outdir / "overall.csv",
        "overall_master": outdir / "overall_by_variant_tf_h.csv",
        "per_symbol_year": outdir / "per_symbol_year.csv",
        "per_symbol_year_master": outdir / "per_symbol_year_by_variant_tf_h.csv",
        "best_selection": outdir / "best_selection.csv",
        "best_master": outdir / "best_horizon_selection.csv",
        "lift": outdir / "lift_vs_baseline.csv",
        "policy_results": outdir / "policy_results.csv",
        "run_logs": outdir / "run_logs.csv",
        "barrier_results": outdir / "barrier_results.csv",
        "claim_verification": outdir / "claim_verification.csv",
        "config_audit": outdir / "config_audit.csv",
        "baseline_vs_prod": outdir / "baseline_vs_prod.csv",
    }
    return mapping


def _pick_existing(primary: Path, fallback: Path) -> Path:
    return primary if primary.exists() else fallback


def _add_derived_metrics(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    numeric_cols = [
        "mean_ret",
        "median_ret",
        "p25_ret",
        "p75_ret",
        "mean_mfe",
        "mean_mae",
        "up_rate",
        "good_rate",
        "worst_year_mean_ret",
        "worst_year_good_rate",
        "n_total",
        "n_valid",
        "horizon_h",
        "min_symbol_year_n_valid",
    ]
    for c in numeric_cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    for c in ["mean_ret", "median_ret", "p25_ret", "p75_ret", "mean_mfe", "mean_mae", "worst_year_mean_ret"]:
        if c in out.columns:
            out[f"{c}_bps"] = out[c] * 10000.0
    if "up_rate" in out.columns:
        out["down_rate"] = 1.0 - out["up_rate"]
    return out


def _ensure_eligibility(
    overall: pd.DataFrame,
    min_n_valid_global: int,
    min_n_valid_per_symbol_year: int,
) -> pd.DataFrame:
    if overall.empty:
        return overall
    out = overall.copy()
    if "selection_eligible" in out.columns:
        out["selection_eligible"] = _coerce_bool(out["selection_eligible"])
        return out
    out["n_valid"] = pd.to_numeric(out.get("n_valid"), errors="coerce")
    out["min_symbol_year_n_valid"] = pd.to_numeric(out.get("min_symbol_year_n_valid"), errors="coerce")
    out["selection_eligible"] = (
        (out["n_valid"] >= float(min_n_valid_global))
        & (out["min_symbol_year_n_valid"] >= float(min_n_valid_per_symbol_year))
    )
    return out


def _pick_row(
    g: pd.DataFrame,
    sort_cols: Sequence[str],
    ascending: Sequence[bool],
    constraint: Optional[pd.Series] = None,
) -> Optional[pd.Series]:
    w = g.copy()
    if constraint is not None:
        w = w[constraint].copy()
    if w.empty:
        return None
    ranked = w.sort_values(list(sort_cols), ascending=list(ascending), kind="mergesort", na_position="last")
    return ranked.iloc[0]


def compute_objective_selection(
    overall: pd.DataFrame,
    objective_c_worst_year_min_bps: float,
) -> pd.DataFrame:
    if overall.empty:
        return pd.DataFrame()
    work = overall.copy()
    if "level" in work.columns:
        work["level"] = work["level"].astype(str).str.lower()
        work = work[work["level"] == "executed"].copy()
    if work.empty:
        return pd.DataFrame()

    key_cols = ["variant", "timeframe", "session_mode"]
    rows: List[Dict[str, object]] = []
    thr_ret = float(objective_c_worst_year_min_bps) / 10000.0
    for key_vals, g in work.groupby(key_cols, dropna=False, sort=True):
        if not isinstance(key_vals, tuple):
            key_vals = (key_vals,)
        key_map = dict(zip(key_cols, key_vals))
        g = g.sort_values("horizon_h", kind="mergesort")
        g["selection_eligible"] = _coerce_bool(g["selection_eligible"])
        g_elig = g[g["selection_eligible"]].copy()

        # Objective A: maximize mean_ret
        a = _pick_row(
            g_elig,
            sort_cols=["mean_ret", "worst_year_mean_ret", "good_rate", "horizon_h"],
            ascending=[False, False, False, True],
        )
        # Objective B: maximize worst_year_mean_ret
        b = _pick_row(
            g_elig,
            sort_cols=["worst_year_mean_ret", "mean_ret", "good_rate", "horizon_h"],
            ascending=[False, False, False, True],
        )
        # Objective C: maximize mean_ret with robustness threshold
        c = _pick_row(
            g_elig,
            sort_cols=["mean_ret", "worst_year_mean_ret", "good_rate", "horizon_h"],
            ascending=[False, False, False, True],
            constraint=(pd.to_numeric(g_elig["worst_year_mean_ret"], errors="coerce") >= thr_ret),
        )

        for obj_name, row in [
            ("A_max_mean_ret", a),
            ("B_max_worst_year_mean_ret", b),
            ("C_mean_ret_subject_to_worst_year_min", c),
        ]:
            if row is None:
                rows.append(
                    {
                        **key_map,
                        "objective": obj_name,
                        "selected_horizon_h": pd.NA,
                        "status": "no_eligible_candidate",
                        "mean_ret_bps": np.nan,
                        "worst_year_mean_ret_bps": np.nan,
                        "up_rate": np.nan,
                        "good_rate": np.nan,
                    }
                )
            else:
                rows.append(
                    {
                        **key_map,
                        "objective": obj_name,
                        "selected_horizon_h": int(row["horizon_h"]),
                        "status": "selected",
                        "mean_ret_bps": float(pd.to_numeric(row.get("mean_ret"), errors="coerce") * 10000.0),
                        "worst_year_mean_ret_bps": float(
                            pd.to_numeric(row.get("worst_year_mean_ret"), errors="coerce") * 10000.0
                        ),
                        "up_rate": float(pd.to_numeric(row.get("up_rate"), errors="coerce")),
                        "good_rate": float(pd.to_numeric(row.get("good_rate"), errors="coerce")),
                    }
                )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out.sort_values(["variant", "timeframe", "session_mode", "objective"], kind="mergesort").reset_index(
        drop=True
    )
    return out


def _fmt_cell(v: object) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        if not np.isfinite(v):
            return ""
        return f"{v:.6f}"
    if isinstance(v, (np.floating,)):
        vv = float(v)
        if not np.isfinite(vv):
            return ""
        return f"{vv:.6f}"
    if isinstance(v, (np.integer, int)):
        return str(int(v))
    if pd.isna(v):
        return ""
    return html.escape(str(v))


def _table_html(df: pd.DataFrame, table_id: str, title: str, columns: Sequence[str]) -> str:
    cols = [c for c in columns if c in df.columns]
    if not cols:
        return f"<h3>{html.escape(title)}</h3><p>No columns available.</p>"
    rows: List[str] = []
    for _, r in df[cols].iterrows():
        tds = "".join([f"<td>{_fmt_cell(r[c])}</td>" for c in cols])
        rows.append(f"<tr>{tds}</tr>")
    header = "".join([f"<th data-col='{i}'>{html.escape(c)}</th>" for i, c in enumerate(cols)])
    body = "\n".join(rows)
    return (
        f"<div class='table-wrap'>"
        f"<h3>{html.escape(title)}</h3>"
        f"<input class='table-search' data-target='{table_id}' placeholder='Search table...' />"
        f"<table id='{table_id}' class='report-table'>"
        f"<thead><tr>{header}</tr></thead>"
        f"<tbody>{body}</tbody>"
        f"</table>"
        f"</div>"
    )


def _build_html(
    outdir: Path,
    overall: pd.DataFrame,
    per_sy: pd.DataFrame,
    best_obj: pd.DataFrame,
    lift: pd.DataFrame,
    policy: pd.DataFrame,
    claim: pd.DataFrame,
    config_audit: pd.DataFrame,
    barrier: pd.DataFrame,
    baseline_vs_prod: pd.DataFrame,
    meta: Dict[str, object],
) -> str:
    overview_cards = [
        ("Rows (overall)", int(len(overall))),
        ("Rows (symbol-year)", int(len(per_sy))),
        ("Rows (barrier 1:3)", int(len(barrier))),
        ("Rows (policy)", int(len(policy))),
    ]
    card_html = "".join(
        [f"<div class='card'><div class='k'>{html.escape(k)}</div><div class='v'>{v}</div></div>" for k, v in overview_cards]
    )

    claim_status = "MISSING"
    claim_badge = "warn"
    if not claim.empty and "status" in claim.columns:
        statuses = claim["status"].astype(str).str.lower()
        if len(statuses) > 0 and (statuses == "pass").all():
            claim_status = "PASS"
            claim_badge = "pass"
        else:
            claim_status = "FAIL"
            claim_badge = "fail"

    top_sections: List[str] = []
    top_sections.append(
        "<section class='top-section'>"
        "<h2>Claim Verification</h2>"
        f"<div class='badge {claim_badge}'>{claim_status}</div>"
        f"<div class='meta'><pre>{html.escape(json.dumps(meta, indent=2, sort_keys=True))}</pre></div>"
        f"<div class='cards'>{card_html}</div>"
        + _table_html(
            claim,
            "tbl_claim_verification",
            "Claim: prod / executed / session_on (m15)",
            [
                "horizon_h",
                "n_valid",
                "up_rate_pct",
                "mean_ret_bps",
                "worst_year_mean_ret_bps",
                "expected_n_valid",
                "expected_up_rate_pct",
                "status",
            ],
        )
        + _table_html(
            baseline_vs_prod,
            "tbl_baseline_vs_prod",
            "Baseline vs Prod (Same Slice)",
            [
                "variant",
                "horizon_h",
                "n_valid",
                "up_rate_pct",
                "mean_ret_bps",
                "worst_year_mean_ret_bps",
            ],
        )
        + "</section>"
    )
    top_sections.append(
        "<section class='top-section'>"
        "<h2>Config Audit</h2>"
        + _table_html(
            config_audit,
            "tbl_config_audit",
            "Effective Toggles by Variant",
            [
                "variant",
                "trade_mode_label",
                "entry_mode_label",
                "useVolEntryGate",
                "minEntryVolRatio",
                "gate_vol_entry_at",
                "bosAtrBuffer",
                "maxWaitBars",
                "oneTradeAtTime",
                "cooldownBars",
            ],
        )
        + "</section>"
    )

    by_var = overall.sort_values(["variant", "timeframe", "session_mode", "horizon_h"], kind="mergesort")
    by_tf = overall.sort_values(["timeframe", "variant", "session_mode", "horizon_h"], kind="mergesort")
    by_sess = overall.sort_values(["session_mode", "variant", "timeframe", "horizon_h"], kind="mergesort")
    by_symbol = per_sy.sort_values(["symbol", "year", "variant", "timeframe", "session_mode", "horizon_h"], kind="mergesort")
    by_year = per_sy.sort_values(["year", "symbol", "variant", "timeframe", "session_mode", "horizon_h"], kind="mergesort")

    common_cols = [
        "variant",
        "timeframe",
        "session_mode",
        "level",
        "horizon_h",
        "n_valid",
        "selection_eligible",
        "up_rate",
        "mean_ret_bps",
        "worst_year_mean_ret_bps",
        "good_rate",
        "mean_mfe_bps",
        "mean_mae_bps",
    ]

    sections: List[str] = []
    sections.append(
        "<section id='tab-variant' class='tab-content active'>"
        "<h2>Variant</h2>"
        + _table_html(by_var, "tbl_by_variant", "Metrics by Variant", common_cols)
        + "</section>"
    )
    sections.append(
        "<section id='tab-timeframe' class='tab-content'>"
        "<h2>Timeframe</h2>"
        + _table_html(by_tf, "tbl_by_timeframe", "Metrics by Timeframe", common_cols)
        + "</section>"
    )
    sections.append(
        "<section id='tab-session' class='tab-content'>"
        "<h2>Session Mode</h2>"
        + _table_html(by_sess, "tbl_by_session", "Metrics by Session Mode", common_cols)
        + "</section>"
    )
    sections.append(
        "<section id='tab-symbol' class='tab-content'>"
        "<h2>Symbol</h2>"
        + _table_html(
            by_symbol,
            "tbl_by_symbol",
            "Per Symbol Rows",
            [
                "symbol",
                "year",
                "variant",
                "timeframe",
                "session_mode",
                "level",
                "horizon_h",
                "n_valid",
                "up_rate",
                "mean_ret_bps",
                "good_rate",
            ],
        )
        + "</section>"
    )
    sections.append(
        "<section id='tab-year' class='tab-content'>"
        "<h2>Year</h2>"
        + _table_html(
            by_year,
            "tbl_by_year",
            "Per Year Rows",
            [
                "year",
                "symbol",
                "variant",
                "timeframe",
                "session_mode",
                "level",
                "horizon_h",
                "n_valid",
                "up_rate",
                "mean_ret_bps",
                "good_rate",
            ],
        )
        + "</section>"
    )
    sections.append(
        "<section id='tab-barrier' class='tab-content'>"
        "<h2>Barrier 1:3</h2>"
        + _table_html(
            barrier,
            "tbl_barrier",
            "True Trade Win Rate (Separate From Up Rate)",
            [
                "variant",
                "timeframe",
                "session_mode",
                "n_trades",
                "win_rate_1to3",
                "avg_R",
                "worst_year_win_rate",
                "selected_horizon_h",
                "pricing_mode",
            ],
        )
        + "</section>"
    )
    if not policy.empty and {"variant", "timeframe", "session_mode", "policy", "run_id"}.issubset(policy.columns):
        policy_view = policy.sort_values(["variant", "timeframe", "session_mode", "policy", "run_id"], kind="mergesort")
    else:
        policy_view = policy.copy()
    sections.append(
        "<section id='tab-policy' class='tab-content'>"
        "<h2>Policy Tests</h2>"
        + _table_html(
            policy_view,
            "tbl_policy",
            "Time-Only Policy Evaluation",
            [
                "variant",
                "timeframe",
                "session_mode",
                "policy",
                "x_bps",
                "y_bps",
                "n_valid",
                "mean_policy_ret_bps",
                "win_rate",
                "down_rate",
                "worst_year_mean_policy_ret_bps",
                "early_exit_rate",
                "selection_eligible",
            ],
        )
        + "</section>"
    )

    tabs = [
        ("Variant", "tab-variant"),
        ("Timeframe", "tab-timeframe"),
        ("Session Mode", "tab-session"),
        ("Symbol", "tab-symbol"),
        ("Year", "tab-year"),
        ("Barrier 1:3", "tab-barrier"),
        ("Policy Tests", "tab-policy"),
    ]
    tabs_html = "".join(
        [
            f"<button class='tab-btn{' active' if i == 0 else ''}' data-tab='{tid}'>{html.escape(lbl)}</button>"
            for i, (lbl, tid) in enumerate(tabs)
        ]
    )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>DFD05 Forward-Horizon Report</title>
  <style>
    :root {{
      --bg: #f8fafc;
      --panel: #ffffff;
      --ink: #0f172a;
      --muted: #475569;
      --line: #cbd5e1;
      --accent: #14532d;
      --accent2: #166534;
    }}
    body {{
      margin: 0;
      font-family: "Segoe UI", "Trebuchet MS", sans-serif;
      color: var(--ink);
      background: radial-gradient(circle at 10% 0%, #dcfce7 0%, #f8fafc 40%);
    }}
    .container {{
      max-width: 1500px;
      margin: 24px auto;
      padding: 0 16px 48px;
    }}
    h1 {{ margin: 0 0 8px; }}
    .subtitle {{ color: var(--muted); margin-bottom: 16px; }}
    .tabs {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-bottom: 16px;
    }}
    .tab-btn {{
      border: 1px solid var(--line);
      background: var(--panel);
      color: var(--ink);
      padding: 8px 12px;
      border-radius: 10px;
      cursor: pointer;
      font-weight: 600;
    }}
    .tab-btn.active {{
      background: linear-gradient(90deg, var(--accent), var(--accent2));
      color: #fff;
      border-color: var(--accent2);
    }}
    .top-section {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 16px;
      box-shadow: 0 10px 24px rgba(15, 23, 42, 0.06);
      margin-bottom: 16px;
    }}
    .badge {{
      display: inline-block;
      font-weight: 700;
      border-radius: 999px;
      padding: 6px 12px;
      margin-bottom: 8px;
      border: 1px solid var(--line);
    }}
    .badge.pass {{ background: #16a34a; color: #fff; border-color: #15803d; }}
    .badge.fail {{ background: #dc2626; color: #fff; border-color: #b91c1c; }}
    .badge.warn {{ background: #ca8a04; color: #111827; border-color: #a16207; }}
    .tab-content {{
      display: none;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 16px;
      box-shadow: 0 10px 24px rgba(15, 23, 42, 0.06);
    }}
    .tab-content.active {{ display: block; }}
    .cards {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 10px;
      margin-bottom: 16px;
    }}
    .card {{
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 10px;
      background: #f1f5f9;
    }}
    .card .k {{ color: var(--muted); font-size: 12px; }}
    .card .v {{ font-size: 20px; font-weight: 700; }}
    .meta pre {{
      overflow-x: auto;
      border: 1px solid var(--line);
      background: #f8fafc;
      padding: 8px;
      border-radius: 8px;
      white-space: pre-wrap;
      font-size: 12px;
      color: var(--muted);
    }}
    .table-wrap {{ margin: 18px 0; }}
    .table-search {{
      margin: 8px 0 10px;
      padding: 7px 10px;
      width: min(420px, 100%);
      border: 1px solid var(--line);
      border-radius: 8px;
      font-size: 13px;
    }}
    table.report-table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 12px;
    }}
    .report-table th, .report-table td {{
      border: 1px solid var(--line);
      padding: 6px 8px;
      white-space: nowrap;
      text-align: right;
    }}
    .report-table th {{
      position: sticky;
      top: 0;
      background: #e2e8f0;
      cursor: pointer;
    }}
    .report-table th:first-child, .report-table td:first-child,
    .report-table th:nth-child(2), .report-table td:nth-child(2),
    .report-table th:nth-child(3), .report-table td:nth-child(3),
    .report-table th:nth-child(4), .report-table td:nth-child(4) {{
      text-align: left;
    }}
    @media (max-width: 900px) {{
      .report-table {{ font-size: 11px; }}
      .report-table th, .report-table td {{ padding: 5px 6px; }}
    }}
  </style>
</head>
<body>
  <div class="container">
    <h1>DFD05 Forward-Horizon Evaluation</h1>
    <div class="subtitle">Generated UTC: {html.escape(datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S'))}</div>
    {''.join(top_sections)}
    <div class="tabs">{tabs_html}</div>
    {''.join(sections)}
  </div>
  <script>
    (function() {{
      const btns = document.querySelectorAll('.tab-btn');
      const tabs = document.querySelectorAll('.tab-content');
      btns.forEach((b) => {{
        b.addEventListener('click', () => {{
          btns.forEach(x => x.classList.remove('active'));
          tabs.forEach(x => x.classList.remove('active'));
          b.classList.add('active');
          const target = document.getElementById(b.dataset.tab);
          if (target) target.classList.add('active');
        }});
      }});

      function parseValue(txt) {{
        const t = (txt || '').replace(/,/g, '').replace(/%/g, '').trim();
        const n = Number(t);
        if (!Number.isNaN(n)) return n;
        return t.toLowerCase();
      }}

      function sortTable(table, colIdx) {{
        const tbody = table.tBodies[0];
        const rows = Array.from(tbody.querySelectorAll('tr'));
        const th = table.querySelectorAll('th')[colIdx];
        const asc = !(th.dataset.asc === 'true');
        table.querySelectorAll('th').forEach(x => delete x.dataset.asc);
        th.dataset.asc = asc ? 'true' : 'false';
        rows.sort((ra, rb) => {{
          const va = parseValue(ra.children[colIdx]?.innerText || '');
          const vb = parseValue(rb.children[colIdx]?.innerText || '');
          if (typeof va === 'number' && typeof vb === 'number') {{
            return asc ? va - vb : vb - va;
          }}
          const sa = String(va), sb = String(vb);
          return asc ? sa.localeCompare(sb) : sb.localeCompare(sa);
        }});
        rows.forEach(r => tbody.appendChild(r));
      }}

      document.querySelectorAll('table.report-table').forEach((table) => {{
        table.querySelectorAll('th').forEach((th, idx) => {{
          th.addEventListener('click', () => sortTable(table, idx));
        }});
      }});

      document.querySelectorAll('.table-search').forEach((input) => {{
        const target = document.getElementById(input.dataset.target);
        if (!target) return;
        input.addEventListener('input', () => {{
          const q = input.value.toLowerCase();
          target.querySelectorAll('tbody tr').forEach((tr) => {{
            const txt = tr.innerText.toLowerCase();
            tr.style.display = txt.includes(q) ? '' : 'none';
          }});
        }});
      }});
    }})();
  </script>
</body>
</html>
"""


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Build interactive HTML report from DFD05 evaluation outputs.")
    ap.add_argument("--outdir", required=True, help="Directory containing evaluation outputs.")
    ap.add_argument(
        "--objective_c_worst_year_min_bps",
        type=float,
        default=0.0,
        help="Objective C constraint: worst_year_mean_ret must be >= this threshold in bps.",
    )
    ap.add_argument("--min_n_valid_global", type=int, default=200, help="Eligibility fallback threshold.")
    ap.add_argument(
        "--min_n_valid_per_symbol_year",
        type=int,
        default=10,
        help="Eligibility fallback threshold for min symbol-year sample.",
    )
    return ap


def run_build_html_report(args: argparse.Namespace) -> Dict[str, Path]:
    outdir = Path(args.outdir)
    if not outdir.exists():
        raise FileNotFoundError(f"Outdir does not exist: {outdir}")
    paths = _resolve_inputs(outdir)

    overall_path = _pick_existing(paths["overall"], paths["overall_master"])
    per_sy_path = _pick_existing(paths["per_symbol_year"], paths["per_symbol_year_master"])
    best_path = _pick_existing(paths["best_selection"], paths["best_master"])

    overall = _add_derived_metrics(_load_csv_if_exists(overall_path))
    per_sy = _add_derived_metrics(_load_csv_if_exists(per_sy_path))
    best_raw = _load_csv_if_exists(best_path)
    lift = _add_derived_metrics(_load_csv_if_exists(paths["lift"]))
    policy = _load_csv_if_exists(paths["policy_results"])
    run_logs = _load_csv_if_exists(paths["run_logs"])
    barrier = _load_csv_if_exists(paths["barrier_results"])
    claim = _load_csv_if_exists(paths["claim_verification"])
    config_audit = _load_csv_if_exists(paths["config_audit"])
    baseline_vs_prod = _load_csv_if_exists(paths["baseline_vs_prod"])

    overall = _ensure_eligibility(
        overall=overall,
        min_n_valid_global=int(args.min_n_valid_global),
        min_n_valid_per_symbol_year=int(args.min_n_valid_per_symbol_year),
    )
    if not lift.empty:
        for c in [
            "delta_mean_ret",
            "delta_worst_year_mean_ret",
            "delta_good_rate",
            "delta_worst_year_good_rate",
            "delta_up_rate",
        ]:
            if c in lift.columns:
                lift[c] = pd.to_numeric(lift[c], errors="coerce")
                if "ret" in c:
                    lift[f"{c}_bps"] = lift[c] * 10000.0
                else:
                    lift[f"{c}_pp"] = lift[c] * 100.0
    if not barrier.empty:
        for c in ["n_trades", "win_rate_1to3", "avg_R", "worst_year_win_rate", "selected_horizon_h"]:
            if c in barrier.columns:
                barrier[c] = pd.to_numeric(barrier[c], errors="coerce")
    if not claim.empty:
        for c in ["horizon_h", "n_valid", "up_rate_pct", "mean_ret_bps", "worst_year_mean_ret_bps", "expected_n_valid", "expected_up_rate_pct"]:
            if c in claim.columns:
                claim[c] = pd.to_numeric(claim[c], errors="coerce")

    best_obj = compute_objective_selection(
        overall=overall,
        objective_c_worst_year_min_bps=float(args.objective_c_worst_year_min_bps),
    )
    best_obj_path = outdir / "best_horizon_objectives.csv"
    best_obj.to_csv(best_obj_path, index=False)

    meta = {
        "outdir": str(outdir),
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        "objective_c_worst_year_min_bps": float(args.objective_c_worst_year_min_bps),
        "min_n_valid_global": int(args.min_n_valid_global),
        "min_n_valid_per_symbol_year": int(args.min_n_valid_per_symbol_year),
        "overall_rows": int(len(overall)),
        "per_symbol_year_rows": int(len(per_sy)),
        "best_rows": int(len(best_raw)),
        "lift_rows": int(len(lift)),
        "policy_rows": int(len(policy)),
        "run_log_rows": int(len(run_logs)),
        "barrier_rows": int(len(barrier)),
        "claim_rows": int(len(claim)),
        "config_audit_rows": int(len(config_audit)),
    }

    html_text = _build_html(
        outdir=outdir,
        overall=overall,
        per_sy=per_sy,
        best_obj=best_obj,
        lift=lift,
        policy=policy,
        claim=claim,
        config_audit=config_audit,
        barrier=barrier,
        baseline_vs_prod=baseline_vs_prod,
        meta=meta,
    )
    html_path = outdir / "index.html"
    html_path.write_text(html_text, encoding="utf-8")

    meta_path = outdir / "report_metadata.json"
    meta_path.write_text(json.dumps(meta, sort_keys=True, indent=2), encoding="utf-8")
    return {"index_html": html_path, "best_horizon_objectives": best_obj_path, "report_metadata": meta_path}


def main() -> None:
    args = build_arg_parser().parse_args()
    outputs = run_build_html_report(args)
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
