from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


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
    safe_filename,
    validate_metric_columns,
)


DEFAULT_ALPHA = 0.05
DEFAULT_METRICS = ["Acc", "Prec", "Rec", "F1", "ROC_AUC"]


def normalidade_por_metrica(
    df: pd.DataFrame,
    config_col: str,
    metric_col: str,
    out_dir_metric: Path,
    alpha: float,
) -> pd.DataFrame:
    out_dir_metric.mkdir(parents=True, exist_ok=True)

    work = df[[config_col, metric_col]].copy()
    work[metric_col] = pd.to_numeric(work[metric_col], errors="coerce")
    work = work.dropna(subset=[config_col, metric_col])

    rows: list[dict[str, object]] = []
    configs = sorted(work[config_col].unique(), key=lambda value: str(value))

    for cfg in configs:
        samples = work.loc[work[config_col] == cfg, metric_col].to_numpy(dtype=float)

        if len(samples) < 3:
            rows.append(
                {
                    "config": cfg,
                    "metric": metric_col,
                    "n": len(samples),
                    "mean": float(np.mean(samples)) if len(samples) else np.nan,
                    "std": float(np.std(samples, ddof=1)) if len(samples) > 1 else np.nan,
                    "shapiro_W": np.nan,
                    "shapiro_p": np.nan,
                    f"normal_alpha_{alpha}": np.nan,
                    "note": "n < 3 (Shapiro nao aplicavel)",
                }
            )
            continue

        shapiro_w, shapiro_p = stats.shapiro(samples)

        plt.figure()
        plt.hist(samples, bins=10)
        plt.title(f"Histograma {metric_col} - {cfg} (n={len(samples)})")
        plt.xlabel(metric_col)
        plt.ylabel("Frequencia")
        plt.tight_layout()
        plt.savefig(out_dir_metric / f"hist_{safe_filename(cfg)}.png", dpi=150)
        plt.close()

        plt.figure()
        (osm, osr), (slope, intercept, _r) = stats.probplot(samples, dist="norm", plot=None)
        plt.scatter(osm, osr)
        plt.plot(osm, slope * np.array(osm) + intercept)
        plt.title(f"QQ-plot {metric_col} - {cfg} (Shapiro p={shapiro_p:.4g})")
        plt.xlabel("Quantis teoricos")
        plt.ylabel(f"Quantis observados ({metric_col})")
        plt.tight_layout()
        plt.savefig(out_dir_metric / f"qq_{safe_filename(cfg)}.png", dpi=150)
        plt.close()

        rows.append(
            {
                "config": cfg,
                "metric": metric_col,
                "n": len(samples),
                "mean": float(np.mean(samples)),
                "std": float(np.std(samples, ddof=1)) if len(samples) > 1 else 0.0,
                "shapiro_W": float(shapiro_w),
                "shapiro_p": float(shapiro_p),
                f"normal_alpha_{alpha}": bool(shapiro_p >= alpha),
                "note": "",
            }
        )

    out_df = pd.DataFrame(rows).sort_values(
        [f"normal_alpha_{alpha}", "shapiro_p"],
        ascending=[True, True],
        na_position="first",
    )
    out_path = out_dir_metric / "resumo_normalidade.csv"
    out_df.to_csv(out_path, index=False)
    return out_df


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Executa teste de normalidade por metrica.")
    parser.add_argument("--csv", default=str(DEFAULT_CSV), help="CSV consolidado de entrada.")
    parser.add_argument(
        "--output-dir",
        default=str(OUTPUTS_DIR / "normalidade" / "metricas"),
        help="Diretorio base para salvar os resultados.",
    )
    parser.add_argument(
        "--metrics",
        nargs="*",
        default=DEFAULT_METRICS,
        help="Lista de metricas. Se vazio, usa todas as metricas numericas detectadas.",
    )
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA, help="Nivel de significancia.")
    parser.add_argument(
        "--auto-metrics",
        action="store_true",
        help="Ignora --metrics e detecta automaticamente todas as colunas numericas validas.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    df = read_csv_robust(args.csv)
    config_col = pick_config_column(df)

    if args.auto_metrics:
        metric_columns = detect_metric_columns(df, config_col)
    else:
        metric_columns = validate_metric_columns(df, args.metrics)

    out_dir = resolve_path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Coluna de configuracao detectada: {config_col}")
    print(f"Metricas selecionadas: {metric_columns}")

    for metric_col in metric_columns:
        metric_dir = out_dir / safe_filename(metric_col)
        summary = normalidade_por_metrica(df, config_col, metric_col, metric_dir, args.alpha)

        print("\n" + "=" * 70)
        print(f"[{metric_col}] Resumo salvo em: {metric_dir / 'resumo_normalidade.csv'}")
        columns = ["config", "metric", "n", "mean", "std", "shapiro_p", f"normal_alpha_{args.alpha}", "note"]
        print(summary[columns].to_string(index=False))


if __name__ == "__main__":
    main()
