from __future__ import annotations

import argparse
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.libqsturng import psturng
from statsmodels.stats.oneway import anova_oneway


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common import (
    DEFAULT_CSV,
    OUTPUTS_DIR,
    detect_metric_columns,
    pick_config_column,
    read_csv_robust,
    resolve_path,
    validate_metric_columns,
)


DEFAULT_ALPHA = 0.05
DEFAULT_METRICS = ["F1", "ROC_AUC", "Rec", "Prec"]


def prepare_metric_frame(df: pd.DataFrame, group_col: str, metric_col: str) -> pd.DataFrame:
    work = df[[group_col, metric_col]].copy()
    work[metric_col] = pd.to_numeric(work[metric_col], errors="coerce")
    return work.dropna(subset=[group_col, metric_col])


def welch_anova(df: pd.DataFrame, group_col: str, metric_col: str) -> dict[str, float]:
    clean = prepare_metric_frame(df, group_col, metric_col)
    groups = [group[metric_col].to_numpy() for _, group in clean.groupby(group_col)]
    result = anova_oneway(groups, use_var="unequal", welch_correction=True)

    statistic = float(result.statistic)
    df_num = float(result.df_num)
    df_denom = float(result.df_denom)
    eta2 = (df_num * statistic) / (df_num * statistic + df_denom) if (df_num * statistic + df_denom) else np.nan

    return {
        "F": statistic,
        "p": float(result.pvalue),
        "df1": df_num,
        "df2": df_denom,
        "eta2_approx": eta2,
    }


def games_howell(df: pd.DataFrame, group_col: str, metric_col: str, alpha: float) -> pd.DataFrame:
    clean = prepare_metric_frame(df, group_col, metric_col)
    grouped = clean.groupby(group_col)[metric_col]

    summary = grouped.agg(["count", "mean", "var"]).rename(columns={"count": "n", "mean": "mean", "var": "var"})
    groups = summary.index.tolist()
    group_count = len(groups)

    rows: list[dict[str, object]] = []
    for group_a, group_b in combinations(groups, 2):
        n1, n2 = summary.loc[group_a, "n"], summary.loc[group_b, "n"]
        m1, m2 = summary.loc[group_a, "mean"], summary.loc[group_b, "mean"]
        v1, v2 = summary.loc[group_a, "var"], summary.loc[group_b, "var"]

        se = np.sqrt(v1 / n1 + v2 / n2)
        diff = m1 - m2
        t_value = np.abs(diff) / se if se != 0 else np.nan

        numerator = (v1 / n1 + v2 / n2) ** 2
        denominator = (v1**2) / (n1**2 * (n1 - 1)) + (v2**2) / (n2**2 * (n2 - 1))
        df_ws = numerator / denominator if denominator != 0 else np.nan

        q_value = t_value * np.sqrt(2) if np.isfinite(t_value) else np.nan
        p_value = np.asarray(psturng(q_value, group_count, df_ws)).squeeze()
        p_value = float(p_value) if np.isfinite(p_value) else np.nan

        rows.append(
            {
                "dv": metric_col,
                "group1": group_a,
                "group2": group_b,
                "mean1": m1,
                "mean2": m2,
                "diff_mean": diff,
                "se": se,
                "t": t_value,
                "df": df_ws,
                "q": q_value,
                "p": p_value,
                f"significant(alpha={alpha})": (p_value < alpha) if np.isfinite(p_value) else False,
            }
        )

    return pd.DataFrame(rows).sort_values(["p", "group1", "group2"], ascending=[True, True, True])


def descriptives(df: pd.DataFrame, group_col: str, metric_col: str) -> pd.DataFrame:
    clean = prepare_metric_frame(df, group_col, metric_col)
    return (
        clean.groupby(group_col)[metric_col]
        .agg(n="count", mean="mean", std="std", median="median")
        .reset_index()
        .sort_values("mean", ascending=False)
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Executa Welch ANOVA e Games-Howell.")
    parser.add_argument("--csv", default=str(DEFAULT_CSV), help="CSV consolidado de entrada.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Diretorio de saida. Se omitido, usa outputs/anova/<stem>_metrics.",
    )
    parser.add_argument(
        "--metrics",
        nargs="*",
        default=DEFAULT_METRICS,
        help="Lista de metricas para analisar.",
    )
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA, help="Nivel de significancia.")
    parser.add_argument(
        "--auto-metrics",
        action="store_true",
        help="Ignora --metrics e detecta automaticamente as metricas numericas.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    csv_path = resolve_path(args.csv)
    df = read_csv_robust(csv_path)
    group_col = pick_config_column(df)

    if args.auto_metrics:
        metric_columns = detect_metric_columns(df, group_col)
    else:
        metric_columns = validate_metric_columns(df, args.metrics)

    output_dir = resolve_path(args.output_dir) if args.output_dir else OUTPUTS_DIR / "anova" / f"{csv_path.stem}_metrics"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nCSV: {csv_path}")
    print(f"Coluna de grupo (config): {group_col}")
    print(f"Metricas: {metric_columns}")
    print(f"Saida em: {output_dir}\n")

    summary_rows: list[dict[str, object]] = []

    for metric_col in metric_columns:
        desc = descriptives(df, group_col, metric_col)
        desc_path = output_dir / f"descritivas_{metric_col}.csv"
        desc.to_csv(desc_path, index=False)

        anova_result = welch_anova(df, group_col, metric_col)
        summary_rows.append(
            {
                "dv": metric_col,
                "F": anova_result["F"],
                "df1": anova_result["df1"],
                "df2": anova_result["df2"],
                "p": anova_result["p"],
                "eta2_approx": anova_result["eta2_approx"],
                f"reject_H0(alpha={args.alpha})": anova_result["p"] < args.alpha,
            }
        )

        print(f"=== Welch ANOVA | {metric_col} ===")
        print(
            f"F({anova_result['df1']:.3f}, {anova_result['df2']:.3f}) = "
            f"{anova_result['F']:.6f} | p = {anova_result['p']:.6g} | eta2~ {anova_result['eta2_approx']:.4f}"
        )

        pairwise = games_howell(df, group_col, metric_col, args.alpha)
        pairwise_path = output_dir / f"games_howell_{metric_col}.csv"
        pairwise.to_csv(pairwise_path, index=False)

        significant_col = f"significant(alpha={args.alpha})"
        significant_count = int(pairwise[significant_col].sum())
        print(
            f"Pos-hoc Games-Howell salvo: {pairwise_path.name} | "
            f"Comparacoes significativas (p<{args.alpha}): {significant_count}\n"
        )

    summary = pd.DataFrame(summary_rows).sort_values("p")
    summary_path = output_dir / "welch_anova_summary.csv"
    summary.to_csv(summary_path, index=False)

    print("=== Resumo Welch ANOVA (todas as metricas) ===")
    print(summary.to_string(index=False))
    print(f"\nResumo salvo em: {summary_path}\n")


if __name__ == "__main__":
    main()
