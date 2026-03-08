from __future__ import annotations

import argparse
from pathlib import Path


def run_prepare_templates(base_dir: Path) -> dict[str, Path]:
    base_dir.mkdir(parents=True, exist_ok=True)
    templates_dir = base_dir / "templates"
    templates_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = base_dir / "manifest_template.yaml"
    trades_tpl = templates_dir / "trades_export_template.csv"
    signals_tpl = templates_dir / "signals_export_template.csv"
    summary_tpl = templates_dir / "summary_export_template.csv"
    readme_path = base_dir / "README.md"

    manifest_path.write_text(
        """
imports:
  config_pack: pine16_exact_prod_screenshot
  timeframe: m15
  tv_strategy_name: "16. Divergence Feature Discovery System - LONG ONLY (Baseline + Toggles) [v6 FIXED + Session Gate]"
  files:
    trades:
      - trades_*.csv
    signals:
      - signals_*.csv
    summary:
      - summary_*.csv
notes:
  timezone_requirement: "UTC timestamps required. If export is local, convert to UTC before ingest."
  required_truth: EXACT_PINE_EXPORTED
""".strip()
        + "\n",
        encoding="utf-8",
    )

    trades_tpl.write_text(
        "entry_time_utc,exit_time_utc,entry_price,exit_price,side,symbol,result_r,result_pct,bars_held,atr_entry,sl_price,tp_price\n",
        encoding="utf-8",
    )
    signals_tpl.write_text(
        "bar_time_utc,entry_time_utc,entry_price,side,symbol\n",
        encoding="utf-8",
    )
    summary_tpl.write_text(
        "symbol,n_trades,win_rate,net_r,expectancy_r,profit_factor,year\n",
        encoding="utf-8",
    )

    readme_path.write_text(
        """
# TradingView Export Staging

Place TradingView exports in this folder before running ingestion.

Expected files:
- Trades export CSV (filename includes `trade`)
- Optional signals/alerts CSV (filename includes `signal`, `alert`, or `event`)
- Optional strategy summary CSV (filename includes `summary`, `performance`, or `tester`)

Then run:

```bash
python -m scripts.pine16_ingest_tv_exports --imports-dir tv_exports/
```
""".strip()
        + "\n",
        encoding="utf-8",
    )

    return {
        "manifest_template": manifest_path,
        "trades_template": trades_tpl,
        "signals_template": signals_tpl,
        "summary_template": summary_tpl,
        "readme": readme_path,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Prepare TradingView export templates for Pine16 exact ingestion.")
    ap.add_argument("--imports-dir", default="tv_exports", help="Directory to place templates and import files.")
    return ap


def main() -> None:
    args = build_arg_parser().parse_args()
    outputs = run_prepare_templates(Path(args.imports_dir))
    for k, v in outputs.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()

