from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common import DEFAULT_CSV, OUTPUTS_DIR, pick_config_column, read_csv_robust, resolve_path, safe_filename


def summarize_metric_by_config(df: pd.DataFrame, metric_col: str) -> pd.DataFrame:
    config_col = pick_config_column(df)
    if metric_col not in df.columns:
        raise ValueError(f"Metrica nao encontrada no CSV: {metric_col}")

    work = df[[config_col, metric_col]].copy()
    work[metric_col] = pd.to_numeric(work[metric_col], errors="coerce")
    work = work.dropna(subset=[config_col, metric_col])

    return (
        work.groupby(config_col)[metric_col]
        .agg(n="count", mean="mean", std="std")
        .reset_index()
        .rename(columns={"mean": f"mean_{safe_filename(metric_col).lower()}", "std": f"std_{safe_filename(metric_col).lower()}"})
        .sort_values(config_col)
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Calcula media e desvio padrao por configuracao.")
    parser.add_argument("--csv", default=str(DEFAULT_CSV), help="CSV consolidado de entrada.")
    parser.add_argument("--metric", default="Acc", help="Nome da metrica a resumir.")
    parser.add_argument(
        "--output",
        default=None,
        help="CSV de saida. Se omitido, salva em outputs/descritivas/.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    df = read_csv_robust(args.csv)
    summary = summarize_metric_by_config(df, args.metric)

    default_output = OUTPUTS_DIR / "descritivas" / f"{safe_filename(args.metric).lower()}_mean_std_por_config.csv"
    output_path = resolve_path(args.output) if args.output else default_output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_path, index=False)

    print(f"\n=== Media e desvio padrao de {args.metric} por configuracao ===\n")
    print(summary.to_string(index=False))
    print(f"\nResumo salvo em: {output_path}")


if __name__ == "__main__":
    main()
