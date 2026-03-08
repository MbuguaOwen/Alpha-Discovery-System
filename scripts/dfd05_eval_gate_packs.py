from __future__ import annotations

import argparse
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd


DEFAULT_PACKS_24H: Dict[str, List[str]] = {
    "Pack_A": [
        "daily_adx >= 42",
        "daily_di_ok == 1",
        "osc_change_pct >= 70",
        "bars_gap >= 16",
        "vol_ratio_entry >= 1.5",
    ],
    "Pack_B": [
        "daily_adx >= 42",
        "daily_di_ok == 1",
        "osc_change_pct >= 70",
        "bars_gap >= 16",
        "vol_ratio_entry >= 1.5",
        "vol_spike_ok_entry == 1",
    ],
    "Pack_C": [
        "daily_adx >= 42",
        "daily_di_ok == 1",
        "osc_change_pct >= 70",
        "bars_gap >= 16",
        "vol_ratio_entry >= 1.5",
        "vol_behavior_ok_entry == 1",
    ],
    "Pack_D": [
        "daily_adx >= 42",
        "daily_di_ok == 1",
        "osc_change_pct >= 70",
        "bars_gap >= 16",
        "vol_ratio_entry >= 1.5",
        "vol_spike_ok_entry == 1",
        "vol_behavior_ok_entry == 1",
    ],
    "Pack_E": [
        "daily_adx >= 42",
        "daily_di_ok == 1",
        "osc_change_pct >= 70",
        "bars_gap >= 16",
        "vol_ratio_entry >= 1.5",
        "vol_spike_ok_entry == 1",
        "vol_behavior_ok_entry == 1",
        "session_ok_pivot == 1",
    ],
}


def _load_packs(path: str | None) -> Dict[str, List[str]]:
    if not path:
        return DEFAULT_PACKS_24H
    try:
        import yaml  # type: ignore
    except ModuleNotFoundError as exc:
        raise SystemExit("PyYAML is required for --packs_yaml") from exc
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if isinstance(raw, dict) and "packs" in raw and isinstance(raw["packs"], dict):
        raw = raw["packs"]
    if not isinstance(raw, dict):
        raise SystemExit("Invalid packs YAML format. Expected mapping of pack name -> rules list.")
    out: Dict[str, List[str]] = {}
    for k, v in raw.items():
        if not isinstance(v, list):
            continue
        out[str(k)] = [str(r) for r in v]
    if not out:
        raise SystemExit("No valid packs found in YAML.")
    return out


def _pick_horizon_cols(df: pd.DataFrame, h: int) -> dict[str, str]:
    suf = f"{h}h"
    tp_res = next((c for c in df.columns if c.startswith("tp_first_resolved_") and c.endswith(suf)), None)
    sl_res = next((c for c in df.columns if c.startswith("sl_first_resolved_") and c.endswith(suf)), None)
    tp = next((c for c in df.columns if c.startswith("tp_first_") and c.endswith(suf)), None)
    sl = next((c for c in df.columns if c.startswith("sl_first_") and c.endswith(suf)), None)
    nh = next((c for c in df.columns if c.startswith("no_hit_") and c.endswith(suf)), None)
    trunc = next((c for c in df.columns if c.startswith("is_truncated_") and c.endswith(suf)), None)
    if not tp or not sl or not nh or not trunc:
        raise SystemExit(f"Missing required horizon columns for {h}h.")
    return {
        "tp": tp_res or tp,
        "sl": sl_res or sl,
        "no_hit": nh,
        "trunc": trunc,
    }


def _derive_year(df: pd.DataFrame) -> pd.Series:
    if "event_time_ms" in df.columns:
        dt = pd.to_datetime(df["event_time_ms"], unit="ms", utc=True, errors="coerce")
    else:
        dt = pd.to_datetime(df.get("time"), utc=True, errors="coerce")
    return dt.dt.year.astype("Int64")


def _infer_tf_runid(path: Path, frame: pd.DataFrame) -> tuple[str, str]:
    m = re.match(r"^labeled_dfd05_(?P<tf>[^_]+)_(?P<rid>.+)$", path.stem)
    if m:
        return m.group("tf"), m.group("rid")
    tf = "unknown"
    if "timeframe" in frame.columns:
        vals = frame["timeframe"].dropna().astype(str).unique().tolist()
        if len(vals) == 1:
            tf = vals[0]
    rid = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return tf, rid


def _apply_rule(df: pd.DataFrame, rule: str) -> pd.Series:
    if " >= " in rule:
        lhs, rhs = rule.split(" >= ", 1)
        if lhs not in df.columns:
            return pd.Series(False, index=df.index)
        x = pd.to_numeric(df[lhs], errors="coerce")
        return (np.isfinite(x.values) & (x >= float(rhs.strip())))
    if " == " in rule:
        lhs, rhs = rule.split(" == ", 1)
        if lhs not in df.columns:
            return pd.Series(False, index=df.index)
        rhs_tok = rhs.strip()
        if rhs_tok.lower() in {"true", "false"}:
            target = 1.0 if rhs_tok.lower() == "true" else 0.0
        else:
            target = float(rhs_tok)
        x = pd.to_numeric(df[lhs], errors="coerce")
        return x == target
    return pd.Series(False, index=df.index)


def _exp_r_total(df: pd.DataFrame, tp_col: str, sl_col: str, rr: float) -> float:
    if len(df) == 0:
        return float("nan")
    tp = float(pd.to_numeric(df[tp_col], errors="coerce").mean())
    sl = float(pd.to_numeric(df[sl_col], errors="coerce").mean())
    return float(rr * tp - sl)


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate DFD05 gate packs")
    ap.add_argument("--labeled", required=True, help="Path to labeled parquet")
    ap.add_argument("--horizon", type=int, default=24, help="Horizon hours to evaluate")
    ap.add_argument("--rr", type=float, default=3.0, help="RR multiplier for expR_total")
    ap.add_argument("--packs_yaml", required=False, help="Optional YAML with pack definitions")
    ap.add_argument("--outdir", default="data/reports", help="Output directory")
    args = ap.parse_args()

    labeled_path = Path(args.labeled)
    df = pd.read_parquet(labeled_path)
    cols = _pick_horizon_cols(df, args.horizon)
    df = df[df[cols["trunc"]].fillna(0) == 0].copy().reset_index(drop=True)
    df["_year"] = _derive_year(df)
    packs = _load_packs(args.packs_yaml)

    rows: list[dict[str, Any]] = []
    for pack_name, rules in packs.items():
        mask = pd.Series(True, index=df.index)
        for rule in rules:
            mask &= _apply_rule(df, rule=rule)
        sub = df.loc[mask].copy()

        tp_rate = float(pd.to_numeric(sub[cols["tp"]], errors="coerce").mean()) if len(sub) else float("nan")
        sl_rate = float(pd.to_numeric(sub[cols["sl"]], errors="coerce").mean()) if len(sub) else float("nan")
        no_hit_rate = float(pd.to_numeric(sub[cols["no_hit"]], errors="coerce").mean()) if len(sub) else float("nan")
        hit_rate = float(1.0 - no_hit_rate) if not np.isnan(no_hit_rate) else float("nan")
        exp_total = _exp_r_total(sub, tp_col=cols["tp"], sl_col=cols["sl"], rr=args.rr)

        sy = []
        for (sym, yr), g in sub.groupby(["symbol", "_year"], dropna=False):
            g_tp = float(pd.to_numeric(g[cols["tp"]], errors="coerce").mean()) if len(g) else float("nan")
            g_sl = float(pd.to_numeric(g[cols["sl"]], errors="coerce").mean()) if len(g) else float("nan")
            g_nh = float(pd.to_numeric(g[cols["no_hit"]], errors="coerce").mean()) if len(g) else float("nan")
            g_hit = float(1.0 - g_nh) if not np.isnan(g_nh) else float("nan")
            g_exp = _exp_r_total(g, tp_col=cols["tp"], sl_col=cols["sl"], rr=args.rr)
            sy.append(
                {
                    "section": "per_symbol_year",
                    "pack": pack_name,
                    "symbol": str(sym) if pd.notna(sym) else None,
                    "year": int(yr) if pd.notna(yr) else None,
                    "n": float(len(g)),
                    "tp_rate": g_tp,
                    "sl_rate": g_sl,
                    "no_hit_rate": g_nh,
                    "hit_rate": g_hit,
                    "expR_total": g_exp,
                    "worst_year_expR": np.nan,
                    "trades_per_symbol_year": np.nan,
                    "rules": "; ".join(rules),
                }
            )
        sy_df = pd.DataFrame(sy)
        worst_year = float(sy_df.groupby("year", dropna=False)["expR_total"].mean().min()) if len(sy_df) else float("nan")
        trades_per_symbol_year = float(sy_df["n"].mean()) if len(sy_df) else float("nan")

        rows.append(
            {
                "section": "overall",
                "pack": pack_name,
                "symbol": None,
                "year": None,
                "n": float(len(sub)),
                "tp_rate": tp_rate,
                "sl_rate": sl_rate,
                "no_hit_rate": no_hit_rate,
                "hit_rate": hit_rate,
                "expR_total": exp_total,
                "worst_year_expR": worst_year,
                "trades_per_symbol_year": trades_per_symbol_year,
                "rules": "; ".join(rules),
            }
        )
        rows.extend(sy)

    out_df = pd.DataFrame(rows)
    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tf, rid = _infer_tf_runid(labeled_path, df)
    out_path = out_dir / f"dfd05_gate_packs_{tf}_{rid}.csv"
    out_df.to_csv(out_path, index=False)
    print(f"gate_packs: {out_path}")


if __name__ == "__main__":
    main()
