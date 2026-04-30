from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
OUTPUTS_DIR = ROOT_DIR / "outputs"
DEFAULT_CSV = DATA_DIR / "todosResultados.csv"

CONFIG_CANDIDATES = [
    "config",
    "configuracao",
    "configuração",
    "configuration",
    "exp",
    "experimento",
    "experiment",
    "setup",
    "setting",
    "modelo",
    "model",
    "grupo",
    "group",
    "nome",
    "name",
]


def resolve_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT_DIR / candidate


def safe_filename(value: str) -> str:
    text = re.sub(r"[^\w\-_. ]+", "_", str(value), flags=re.UNICODE)
    text = text.strip().replace(" ", "_")
    return text[:120] if len(text) > 120 else text


def pick_config_column(df: pd.DataFrame) -> str:
    lowered = {column.lower(): column for column in df.columns}

    for candidate in CONFIG_CANDIDATES:
        if candidate in lowered:
            return lowered[candidate]

    for candidate in CONFIG_CANDIDATES:
        for lowered_name, original_name in lowered.items():
            if candidate in lowered_name:
                return original_name

    object_columns = [column for column in df.columns if df[column].dtype == "object"]
    if object_columns:
        return object_columns[0]

    raise ValueError("Nao foi possivel detectar a coluna de configuracao experimental.")


def read_csv_robust(csv_path: str | Path) -> pd.DataFrame:
    resolved = resolve_path(csv_path)
    for sep in [None, ";", "|", "\t", ","]:
        try:
            if sep is None:
                return pd.read_csv(resolved)
            return pd.read_csv(resolved, sep=sep)
        except Exception:
            continue

    raise ValueError(f"Falha ao ler o CSV: {resolved}. Verifique separador e encoding.")


def detect_metric_columns(df: pd.DataFrame, config_col: str) -> list[str]:
    ignored = {
        config_col.lower(),
        "seed",
        "seeds",
        "fold",
        "kfold",
        "arquivo",
        "file",
        "path",
        "run",
        "id",
        "idx",
    }

    metric_columns: list[str] = []
    for column in df.columns:
        lowered = column.lower()
        if lowered in ignored:
            continue
        if pd.api.types.is_numeric_dtype(df[column]):
            metric_columns.append(column)

    return metric_columns


def validate_metric_columns(df: pd.DataFrame, metric_columns: list[str]) -> list[str]:
    missing = [column for column in metric_columns if column not in df.columns]
    if missing:
        raise ValueError(f"Metricas nao encontradas no CSV: {missing}")
    return metric_columns
