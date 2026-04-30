import gc
import json
import os
from pathlib import Path

import cv2
import numpy as np
import seaborn as sns
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from tensorflow.keras import backend as K

from src.analysis.eigencam import run_eigencam_for_predictions
from src.analysis.eigencam_prep import prep_eigencam_data
from src.training.classification import (
    apply_optional_segmenter,
    build_split,
    load_classification_dataset,
    normalize_split_with_bounds,
    prepare_images_for_model,
    resize_split,
)


def prep_test_data(
    raw_root,
    angle,
    split_json,
    resize=True,
    resize_method="GrayPadding",
    resize_to=224,
    segmenter="none",
    seg_model_path="",
    rgb=False,
    channel_method="MapaCalor",
    yolo_marker_source="manual_masks",
    seed=13388,
    marcadores=0,
    fold=0,
):
    """
    Prepara X_test seguindo o mesmo preprocessamento salvo no split de treino.
    """
    if yolo_marker_source not in {"yolo", "manual_masks"}:
        raise ValueError("yolo_marker_source deve ser 'yolo' ou 'manual_masks'")

    use_manual_marker_masks = segmenter == "yolo" and yolo_marker_source == "manual_masks"
    dataset = load_classification_dataset(
        raw_root,
        angle,
        exclude_segmentation_ids=True,
        segmentation_images_dir="data/Termografias_Dataset_Segmentação/images",
        load_marker_masks=use_manual_marker_masks,
        mask_root="Frontal-mask-txt",
    )

    with open(split_json, "r") as f:
        split = json.load(f)

    tr_idx = np.array(split["train_idx"])
    va_idx = np.array(split["val_idx"])
    te_idx = np.array(split["test_idx"])
    mn, mx = split["mn_train_pixel"], split["mx_train_pixel"]
    split_data = build_split(dataset, tr_idx, va_idx, te_idx)

    # with open(f"test_idx{fold}", "w") as f:
    #     json.dump({"ids_data": dataset.ids_data[te_idx].tolist()}, f)

    split_data = normalize_split_with_bounds(split_data, mn, mx)

    if resize:
        split_data = resize_split(split_data, resize_method, resize_to)

    split_data = apply_optional_segmenter(split_data, segmenter, seg_model_path)
    X_test = split_data.X_test
    y_test = split_data.y_test

    if rgb:
        X_test = (X_test * 255).astype(np.uint8)

        if channel_method == "MapaCalor":
            imgs_test = []
            for img in X_test:
                img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                imgs_test.append(img)
            X_test = np.array(imgs_test, dtype=np.uint8)
        else:
            X_test = np.stack((X_test,) * 3, axis=-1)

    return X_test, y_test


def _plot_and_save_cm(cm, classes, title, out_png):
    plt.figure(figsize=(4, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        square=True,
        xticklabels=classes,
        yticklabels=classes,
        cmap="Blues",
    )
    plt.xlabel("Predito")
    plt.ylabel("Real")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()


def evaluate_model_cm(
    model_path,
    output_path,
    split_json,
    raw_root,
    message,
    angle="Frontal",
    resize=True,
    resize_to=224,
    resize_method="BlackPadding",
    segmenter="none",
    seg_model_path="",
    classes=("Healthy", "Sick"),
    rgb=False,
    channel_method="MapaCalor",
    yolo_marker_source="manual_masks",
    marcadores=0,
    fold=0,
):
    """
    Avalia o modelo salvo no fold especificado e gera matriz de confusão.
    """
    os.makedirs(output_path, exist_ok=True)

    X_test, y_test = prep_test_data(
        raw_root,
        angle,
        split_json,
        resize,
        resize_method,
        resize_to,
        segmenter,
        seg_model_path,
        rgb=rgb,
        channel_method=channel_method,
        yolo_marker_source=yolo_marker_source,
        marcadores=marcadores,
        fold=fold,
    )

    model = tf.keras.models.load_model(model_path, compile=False)
    X_test = prepare_images_for_model(X_test, model.__class__.__name__, channel_method)

    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob > 0.5).astype(int).ravel()

    cm = confusion_matrix(y_test, y_pred)
    out_png = os.path.join(output_path, f"cm_{message}_{angle}.png")
    _plot_and_save_cm(cm, classes, f"Confusion Matrix – {message}", out_png=out_png)

    acc = accuracy_score(y_test, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_test,
        y_pred,
        average="binary",
        zero_division=0,
    )

    out_txt = os.path.join(output_path, f"resultado_{message}_{angle}.txt")
    with open(out_txt, "a") as f:
        f.write(f"Acc={acc:.4f}  Prec={prec:.4f}  Rec={rec:.4f}  F1={f1:.4f}\n")
        f.write(classification_report(y_test, y_pred, target_names=classes, zero_division=0))
        f.write("\n")

    K.clear_session()
    gc.collect()
    return y_pred


def evaluate_fold_with_eigencam(
    model_path,
    split_json,
    output_base,
    raw_root,
    message,
    X,
    y,
    ids_data,
    angle="Frontal",
    classes=("Healthy", "Sick"),
    resize=True,
    resize_to=224,
    resize_method="BlackPadding",
    segmenter="none",
    seg_model_path="",
    rgb=False,
    channel_method="MapaCalor",
    yolo_marker_source="manual_masks",
    marcadores=0,
    fold=0,
):
    """
    Avalia um fold e gera mapas EigenCAM separados em acertos e erros.
    """
    output_base = Path(output_base)
    out_dir_cm = output_base / "Confusion_Matrix"
    out_dir_cm.mkdir(parents=True, exist_ok=True)

    cm_message = f"{message}_F{fold}"
    y_pred = evaluate_model_cm(
        model_path=model_path,
        output_path=str(out_dir_cm),
        split_json=split_json,
        raw_root=raw_root,
        message=cm_message,
        angle=angle,
        classes=classes,
        rgb=rgb,
        resize_method=resize_method,
        resize=resize,
        resize_to=resize_to,
        segmenter=segmenter,
        seg_model_path=seg_model_path,
        yolo_marker_source=yolo_marker_source,
        marcadores=marcadores,
        fold=fold,
    )

    X_test, y_test, ids_test = prep_eigencam_data(
        X,
        y,
        ids_data,
        split_json=split_json,
        resize=resize,
        resize_method=resize_method,
        resize_to=resize_to,
        segmenter=segmenter,
        seg_model_path=seg_model_path,
        fold=fold,
    )

    cam_dirs = run_eigencam_for_predictions(
        imgs=X_test,
        y_true=y_test,
        y_pred=y_pred,
        ids=ids_test,
        model_path=model_path,
        output_root=output_base / "CAM_results",
        message=cm_message,
    )

    print(f"[OK] {message} | fold {fold} -> {cam_dirs['erros']}")
    return {
        "message": message,
        "fold": fold,
        "model_path": model_path,
        "split_json": split_json,
        "y_pred": y_pred,
        "y_test": y_test,
        "ids_test": ids_test,
        "cam_dirs": cam_dirs,
    }


def evaluate_experiments_with_eigencam(
    experiments,
    model_dirs,
    output_base,
    raw_root,
    angle,
    X,
    y,
    ids_data,
    classes=("Healthy", "Sick"),
    resize=True,
    resize_to=224,
    rgb=False,
    channel_method="MapaCalor",
    yolo_marker_source="manual_masks",
    marcadores=0,
    folds=5,
):
    """
    Reproduz o pipeline de avaliacao + EigenCAM para uma lista de experimentos.
    """
    results = []
    output_base = Path(output_base)

    for exp in experiments:
        resize_method = exp["resize_method"]
        message = exp["message"]
        segmenter = exp.get("segment", "none")
        seg_model_path = exp.get("segmenter_path", "")

        backbone_key = "resnet" if message.upper().startswith("RESNET") else "vgg"
        model_dir = model_dirs[backbone_key]

        for fold in range(folds):
            model_path = f"{model_dir}/{message}_{angle}_F{fold}.h5"
            split_json = f"splits/{message}_{angle}_F{fold}.json"

            result = evaluate_fold_with_eigencam(
                model_path=model_path,
                split_json=split_json,
                output_base=output_base,
                raw_root=raw_root,
                message=message,
                X=X,
                y=y,
                ids_data=ids_data,
                angle=angle,
                classes=classes,
                resize=resize,
                resize_to=resize_to,
                resize_method=resize_method,
                segmenter=segmenter,
                seg_model_path=seg_model_path,
                rgb=rgb,
                channel_method=channel_method,
                yolo_marker_source=yolo_marker_source,
                marcadores=marcadores,
                fold=fold,
            )
            results.append(result)

    return results
