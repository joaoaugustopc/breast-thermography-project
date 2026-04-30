# evaluate_all_models.py
import glob
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

import tensorflow as tf

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from src.training.evaluation import prep_test_data
else:
    from .evaluation import prep_test_data


DEFAULT_INPUT_GLOB = "./eval_results/*_per_run.csv"
DEFAULT_OUTPUT_CSV = "todosResultados.csv"


def safe_roc_auc(y_true, y_score):
    y_true = np.asarray(y_true).astype(int)
    if len(np.unique(y_true)) < 2:
        return np.nan
    return roc_auc_score(y_true, y_score)


def evaluate_one_run(
    message: str,
    raw_root: str,
    angle: str = "Frontal",
    fold: int = 0,
    resize: bool = True,
    resize_to: int = 224,
    resize_method: str = "BlackPadding",
    segmenter_model: str = "none",
    seg_model_path: str = "",
    rgb: bool = False,
    channel_method: str = "MapaCalor",
    yolo_marker_source: str = "manual_masks",
):
    """
    Avalia UM modelo (uma seed/run), usando o split salvo daquele message+fold.
    Retorna dict com métricas.
    """
    split_json = f"splits/{message}_{angle}_F{fold}.json"
    model_path = f"modelos/Vgg_16/{message}_{angle}_F{fold}.h5"

    if not os.path.isfile(split_json):
        raise FileNotFoundError(f"Split não encontrado: {split_json}")
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Modelo não encontrado: {model_path}")

    X_test, y_test = prep_test_data(
        raw_root=raw_root,
        angle=angle,
        split_json=split_json,
        resize=resize,
        resize_method=resize_method,
        resize_to=resize_to,
        segmenter=segmenter_model,
        seg_model_path=seg_model_path,
        rgb=rgb,
        channel_method=channel_method,
        yolo_marker_source=yolo_marker_source,
        marcadores=0,
        fold=fold,
    )

    model = tf.keras.models.load_model(model_path, compile=False)
    y_prob = model.predict(X_test, verbose=0).ravel()
    y_pred = (y_prob > 0.5).astype(int)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = safe_roc_auc(y_test, y_prob)

    return {
        "message": message,
        "raw_root": raw_root,
        "angle": angle,
        "fold": fold,
        "acc": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": auc,
        "n_test": int(len(y_test)),
        "test_pos": int(np.sum(np.asarray(y_test) == 1)),
        "test_neg": int(np.sum(np.asarray(y_test) == 0)),
    }


def run_experiment_block(
    exp_name: str,
    message_fmt: str,
    raw_root: str,
    t_values=range(30),
    angle="Frontal",
    fold=0,
    resize=True,
    resize_to=224,
    resize_method="BlackPadding",
    segmenter_model="none",
    seg_model_path="",
    rgb=False,
    channel_method="MapaCalor",
    yolo_marker_source="manual_masks",
    out_dir="eval_results",
):
    """
    Roda t0..t29 (ou o range que você quiser) e salva CSV + resumo.
    """
    os.makedirs(out_dir, exist_ok=True)

    rows = []
    for t in t_values:
        message = message_fmt.format(t=t)
        print(f"[{exp_name}] avaliando {message} ...")

        row = evaluate_one_run(
            message=message,
            raw_root=raw_root,
            angle=angle,
            fold=fold,
            resize=resize,
            resize_to=resize_to,
            resize_method=resize_method,
            segmenter_model=segmenter_model,
            seg_model_path=seg_model_path,
            rgb=rgb,
            channel_method=channel_method,
            yolo_marker_source=yolo_marker_source,
        )
        row["experiment"] = exp_name
        row["t"] = int(t)
        rows.append(row)

    df = pd.DataFrame(rows)

    csv_path = os.path.join(out_dir, f"{exp_name}_per_run.csv")
    df.to_csv(csv_path, index=False)

    summary = df[["acc", "precision", "recall", "f1", "roc_auc"]].agg(["mean", "std"])
    summary_path = os.path.join(out_dir, f"{exp_name}_summary.csv")
    summary.to_csv(summary_path)

    print(f"[{exp_name}] salvo: {csv_path}")
    print(f"[{exp_name}] salvo: {summary_path}")

    return df


def eval_results():
   
    experiments = [
        {
            "exp_name": "DatasetSeg2classes",
            "message_fmt": "Vgg_AUG_CV_DatasetSeg2classes_17_01_t{t}",
            "raw_root": "filtered_raw_dataset",
            "segmenter_model": "yolo",
            "seg_model_path": "runs/segment/train39/weights/best.pt",
        }
    ]

    all_dfs = []
    for cfg in experiments:
        df = run_experiment_block(
            exp_name=cfg["exp_name"],
            message_fmt=cfg["message_fmt"],
            raw_root=cfg["raw_root"],
            t_values=range(30),
            angle="Frontal",
            fold=0,
            resize=True,
            resize_to=224,
            resize_method="BlackPadding",
            segmenter_model=cfg["segmenter_model"],
            seg_model_path=cfg["seg_model_path"],
            rgb=False,
            channel_method="MapaCalor",
            out_dir="eval_results",
        )
        all_dfs.append(df)

    df_all = pd.concat(all_dfs, ignore_index=True)
    summary_all = df_all.groupby("experiment")[["acc", "precision", "recall", "f1", "roc_auc"]].agg(["mean", "std"])
    summary_all.to_csv("eval_results/ALL_experiments_summary.csv")

    print("OK: eval_results/ALL_experiments_summary.csv")


def convert_per_run(df: pd.DataFrame, source_file: str) -> pd.DataFrame:
    out = df.copy()

    out["Configuração"] = out["message"].astype(str).str.replace(r"_t\d+$", "", regex=True)
    out["Seed"] = out["t"].astype(int)
    out["Fold"] = out["fold"].astype(int)

    out["Acc"] = out["acc"].astype(float)
    out["Prec"] = out["precision"].astype(float)
    out["Rec"] = out["recall"].astype(float)
    out["F1"] = out["f1"].astype(float)
    out["ROC_AUC"] = out["roc_auc"].astype(float)
    out["Arquivo"] = source_file

    out = out[["Configuração", "Seed", "Fold", "Acc", "Prec", "Rec", "F1", "ROC_AUC", "Arquivo"]]

    for col in ["Acc", "Prec", "Rec", "F1", "ROC_AUC"]:
        out[col] = out[col].round(4)

    return out


def aggregate_eval_results(input_glob=DEFAULT_INPUT_GLOB, output_csv=DEFAULT_OUTPUT_CSV):
    files = sorted(glob.glob(input_glob, recursive=True))
    if not files:
        raise SystemExit(f"Nenhum arquivo encontrado com o padrão: {input_glob}")

    parts = [convert_per_run(pd.read_csv(path), path) for path in files]
    final = pd.concat(parts, ignore_index=True)
    final = final.sort_values(["Configuração", "Fold", "Seed"], kind="mergesort").reset_index(drop=True)

    final.to_csv(output_csv, index=False)
    print(f"OK: gerado {output_csv} com {len(final)} linhas a partir de {len(files)} arquivos.")
    return final


if __name__ == "__main__":
    eval_results()
    aggregate_eval_results()
