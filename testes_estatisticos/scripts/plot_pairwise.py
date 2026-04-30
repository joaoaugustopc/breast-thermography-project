from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common import OUTPUTS_DIR, resolve_path


DEFAULT_ORDER = [
    "Original",
    "FixedTemp",
    "FixedSize",
    "BreastMarkerSeg",
    "BreastSeg",
    "NoMarker",
    "Moved",
]

DEFAULT_LABEL_MAP = {
    "Original": "ORIG",
    "FixedTemp": "FT",
    "FixedSize": "FS",
    "BreastMarkerSeg": "SEG+TAG",
    "BreastSeg": "SEG",
    "NoMarker": "WT",
    "Moved": "MV",
}


def save_figure(fig: plt.Figure, output_path: Path) -> None:
    suffix = output_path.suffix.lower() or ".png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, format=suffix.lstrip("."))
    plt.close(fig)


def plot_lower_triangle_pvalues_from_pairwise_csv(
    pairwise_csv: str | Path,
    output_path: str | Path,
    *,
    alpha: float = 0.05,
    title: str = "Matriz triangular inferior - p-valores",
    group1_col: str = "group1",
    group2_col: str = "group2",
    p_col: str = "p",
    order: list[str] | None = None,
    label_map: dict[str, str] | None = None,
) -> pd.DataFrame:
    pairwise_df = pd.read_csv(pairwise_csv)

    groups = (
        order
        if order is not None
        else sorted(pd.unique(pd.concat([pairwise_df[group1_col], pairwise_df[group2_col]], ignore_index=True)).tolist())
    )
    display_groups = [label_map.get(group, group) for group in groups] if label_map else groups

    p_matrix = pd.DataFrame(np.ones((len(groups), len(groups))), index=groups, columns=groups)
    for _, row in pairwise_df.iterrows():
        group_a, group_b, p_value = row[group1_col], row[group2_col], row[p_col]
        if group_a in p_matrix.index and group_b in p_matrix.columns:
            p_matrix.loc[group_a, group_b] = p_value
            p_matrix.loc[group_b, group_a] = p_value

    matrix = p_matrix.to_numpy(dtype=float, copy=True)
    matrix[np.triu(np.ones_like(matrix, dtype=bool), k=0)] = np.nan

    fig, ax = plt.subplots(figsize=(9, 8))
    ax.imshow(matrix, cmap="Blues", vmin=0.0, vmax=1.0)
    ax.set_xticks(np.arange(len(groups)))
    ax.set_yticks(np.arange(len(groups)))
    ax.set_xticklabels(display_groups, rotation=45, ha="right", fontsize=16)
    ax.set_yticklabels(display_groups, fontsize=16)
    ax.set_title(title)

    for row_index in range(len(groups)):
        for col_index in range(len(groups)):
            if row_index <= col_index or not np.isfinite(matrix[row_index, col_index]):
                continue

            p_value = matrix[row_index, col_index]
            stars = "*" if p_value < alpha else ""
            text_color = "white" if p_value > 0.8 else "black"
            formatted = f"{p_value:.3g}".replace(".", ",")
            ax.text(col_index, row_index, f"{formatted}{stars}", ha="center", va="center", fontsize=14, color=text_color)

    ax.set_xlim(-0.5, len(groups) - 0.5)
    ax.set_ylim(len(groups) - 0.5, -0.5)
    save_figure(fig, Path(output_path))
    return p_matrix


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Gera matriz triangular inferior a partir do Games-Howell.")
    parser.add_argument(
        "--anova-dir",
        default=str(OUTPUTS_DIR / "anova" / "todosResultados_metrics"),
        help="Diretorio com os CSVs do Games-Howell.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(OUTPUTS_DIR / "graficos"),
        help="Diretorio para salvar as figuras.",
    )
    parser.add_argument(
        "--metrics",
        nargs="*",
        default=["F1", "ROC_AUC"],
        help="Metricas para converter em figura triangular.",
    )
    parser.add_argument("--alpha", type=float, default=0.05, help="Nivel de significancia.")
    parser.add_argument(
        "--save-matrix",
        action="store_true",
        help="Tambem salva a matriz simetrica de p-valores em CSV.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    anova_dir = resolve_path(args.anova_dir)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for metric in args.metrics:
        pairwise_csv = anova_dir / f"games_howell_{metric}.csv"
        output_path = output_dir / f"games_howell_{metric}_triangular.svg"
        matrix = plot_lower_triangle_pvalues_from_pairwise_csv(
            pairwise_csv=pairwise_csv,
            output_path=output_path,
            title=f"Games-Howell - {metric}",
            alpha=args.alpha,
            order=DEFAULT_ORDER,
            label_map=DEFAULT_LABEL_MAP,
        )

        print(f"Figura salva em: {output_path}")
        if args.save_matrix:
            matrix_path = output_dir / f"games_howell_{metric}_pvalue_matrix.csv"
            matrix.to_csv(matrix_path)
            print(f"Matriz salva em: {matrix_path}")


if __name__ == "__main__":
    main()
