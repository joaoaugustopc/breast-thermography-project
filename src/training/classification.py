from dataclasses import dataclass
from datetime import datetime
import json
import os
import time
from pathlib import Path
import gc
import cv2
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tensorflow.keras.applications.resnet import preprocess_input as resnet_preprocess_input
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_preprocess_input
import torch
from ultralytics import YOLO

from src.training.segmentation import segment_with_yolo, unet_segmenter
from utils.data_prep import (
    augment_train_fold,
    augment_train_fold_with_masks,
    listar_imgs_nao_usadas,
    load_raw_images,
    load_raw_images_ufpe,
    load_raw_images_with_masks,
    make_tvt_splits,
    normalize,
    tf_letterbox,
    tf_letterbox_black,
)
from utils.stats import plot_convergence


@dataclass
class ClassificationDataset:
    X: np.ndarray
    y: np.ndarray
    patient_ids: np.ndarray
    ids_data: np.ndarray
    masks: np.ndarray | None = None


@dataclass
class ClassificationSplit:
    X_tr: np.ndarray
    y_tr: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    masks_tr: np.ndarray | None = None
    masks_val: np.ndarray | None = None
    masks_test: np.ndarray | None = None


PRETRAINED_MODEL_PREPROCESSORS = {
    "Vgg_16_pre_trained": vgg_preprocess_input,
    "resnet50_pre_trained": resnet_preprocess_input,
}

def clear_memory():
    tf.keras.backend.clear_session()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def model_name(model):
    return model if isinstance(model, str) else model.__name__


def is_ufpe_dataset(raw_root):
    return "ufpe" in str(raw_root).lower()


def load_classification_dataset(
    raw_root,
    angle,
    exclude_segmentation_ids=True,
    segmentation_images_dir="data/Termografias_Dataset_Segmentação/images",
    load_marker_masks=False,
    mask_root="Frontal-mask-txt",
):
    if is_ufpe_dataset(raw_root):
        X, y, patient_ids = load_raw_images_ufpe(os.path.join(raw_root, angle), exclude=False)
        print(f"Carregando imagens da UFPE: {X.shape}, {y.shape}, {len(patient_ids)} pacientes")
        return ClassificationDataset(X, y, patient_ids, np.asarray(patient_ids), masks=None)

    exclude_set = None
    if exclude_segmentation_ids:
        exclude_set = listar_imgs_nao_usadas(segmentation_images_dir, angle)

    if load_marker_masks:
        X, y, patient_ids, _filenames, ids_data, masks = load_raw_images_with_masks(
            os.path.join(raw_root, angle),
            mask_root,
            exclude=exclude_segmentation_ids,
            exclude_set=exclude_set,
        )
    else:
        X, y, patient_ids, _filenames, ids_data = load_raw_images(
            os.path.join(raw_root, angle),
            exclude=exclude_segmentation_ids,
            exclude_set=exclude_set,
        )
        masks = None

    print(f"Carregando imagens: {X.shape}, {y.shape}, {len(patient_ids)} pacientes")
    return ClassificationDataset(X, y, patient_ids, np.asarray(ids_data), masks=masks)


def write_seed_log(message, seed, path="modelos/random_seed.txt"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as f:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"{timestamp} | {message}: | SEMENTE: {seed}\n")


def write_augmentation_log(fold, X_tr, y_tr, path="modelos/random_seed.txt"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as f:
        f.write(
            f"Shape de treinamento fold {fold} apos o aumento de dados: {X_tr.shape} | "
            f"Saudaveis: {X_tr[y_tr==0].shape[0]} | Doentes: {X_tr[y_tr==1].shape[0]} \n"
        )


def save_split_metadata(message, angle, fold, tr_idx, va_idx, te_idx, ids_data, mn, mx, output_dir="splits"):
    os.makedirs(output_dir, exist_ok=True)
    split_file = os.path.join(output_dir, f"{message}_{angle}_F{fold}.json")
    with open(split_file, "w") as f:
        json.dump({
            "train_idx": tr_idx.tolist(),
            "val_idx": va_idx.tolist(),
            "test_idx": te_idx.tolist(),
            "train_ids": ids_data[tr_idx].tolist(),
            "val_ids": ids_data[va_idx].tolist(),
            "test_ids": ids_data[te_idx].tolist(),
            "mn_train_pixel": float(mn),
            "mx_train_pixel": float(mx),
        }, f)


def build_split(dataset, tr_idx, va_idx, te_idx):
    masks = dataset.masks
    return ClassificationSplit(
        X_tr=dataset.X[tr_idx],
        y_tr=dataset.y[tr_idx],
        X_val=dataset.X[va_idx],
        y_val=dataset.y[va_idx],
        X_test=dataset.X[te_idx],
        y_test=dataset.y[te_idx],
        masks_tr=None if masks is None else masks[tr_idx],
        masks_val=None if masks is None else masks[va_idx],
        masks_test=None if masks is None else masks[te_idx],
    )


def normalize_split(split):
    mn, mx = split.X_tr.min(), split.X_tr.max()
    split.X_tr = normalize(split.X_tr, mn, mx)
    split.X_val = normalize(split.X_val, mn, mx)
    split.X_test = normalize(split.X_test, mn, mx)
    return split, mn, mx


def normalize_split_with_bounds(split, mn, mx):
    split.X_tr = normalize(split.X_tr, mn, mx)
    split.X_val = normalize(split.X_val, mn, mx)
    split.X_test = normalize(split.X_test, mn, mx)
    return split


def augment_training_split(split, n_aug, seed, dataset_kind):
    if n_aug <= 0:
        return split

    if dataset_kind == "ufpe" or split.masks_tr is None:
        split.X_tr, split.y_tr = augment_train_fold(
            split.X_tr, split.y_tr, n_aug=n_aug, seed=seed, dataset=dataset_kind
        )
    else:
        split.X_tr, split.masks_tr, split.y_tr = augment_train_fold_with_masks(
            split.X_tr, split.masks_tr, split.y_tr, n_aug=n_aug, seed=seed
        )
    return split


def _expand_channel(arr):
    return np.expand_dims(arr, axis=-1)


def _resize_images(images, resize_method, resize_to, mask=False):
    mode = "nearest" if mask else "bilinear"
    if resize_method == "GrayPadding" and not mask:
        resized = tf_letterbox(images, resize_to)
    elif resize_method == "GrayPadding" and mask:
        resized = tf_letterbox_black(images, resize_to, mode=mode)
    elif resize_method == "BlackPadding":
        resized = tf_letterbox_black(images, resize_to, mode=mode)
    elif resize_method == "Distorcido":
        resized = tf.image.resize(images, (resize_to, resize_to), method=mode)
    else:
        raise ValueError("resize_method deve ser 'GrayPadding', 'BlackPadding' ou 'Distorcido'")

    return tf.clip_by_value(resized, 0, 1).numpy().squeeze(axis=-1)


def resize_split(split, resize_method="BlackPadding", resize_to=224):
    split.X_tr = _resize_images(_expand_channel(split.X_tr), resize_method, resize_to)
    split.X_val = _resize_images(_expand_channel(split.X_val), resize_method, resize_to)
    split.X_test = _resize_images(_expand_channel(split.X_test), resize_method, resize_to)

    if split.masks_tr is not None:
        split.masks_tr = _resize_images(_expand_channel(split.masks_tr), resize_method, resize_to, mask=True)
        split.masks_val = _resize_images(_expand_channel(split.masks_val), resize_method, resize_to, mask=True)
        split.masks_test = _resize_images(_expand_channel(split.masks_test), resize_method, resize_to, mask=True)
    return split


def apply_optional_segmenter(split, segmenter, seg_model_path):
    if segmenter in (None, "none"):
        return split

    if segmenter == "unet":
        split.X_tr, split.X_val, split.X_test = unet_segmenter(
            split.X_tr, split.X_val, split.X_test, seg_model_path
        )
        print("Segmentacao com UNet concluida.")
        return split

    if segmenter == "yolo":
        if split.masks_tr is None:
            split.X_tr, split.X_val, split.X_test = segment_with_yolo(
                split.X_tr,
                split.X_val,
                split.X_test,
                seg_model_path,
            )
            print("Segmentacao com YOLO concluida no modo original.")
        else:
            split.X_tr, split.X_val, split.X_test = segment_with_yolo(
                split.X_tr,
                split.X_val,
                split.X_test,
                split.masks_tr,
                split.masks_val,
                split.masks_test,
                seg_model_path,
            )
            print("Segmentacao com YOLO concluida com mascaras manuais do marcador.")
        return split

    raise ValueError("segmenter deve ser 'none', 'unet' ou 'yolo'")


def heatmap_to_rgb(images, colormap=cv2.COLORMAP_JET):
    rgb_images = []
    for img in images:
        img_uint8 = img.astype(np.uint8)
        img_bgr = cv2.applyColorMap(img_uint8, colormap)
        rgb_images.append(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    return np.array(rgb_images)


def to_three_channels(images, channel_method="MapaCalor"):
    images = (images * 255).astype(np.uint8)
    if channel_method == "MapaCalor":
        return heatmap_to_rgb(images)
    return np.stack((images,) * 3, axis=-1)


def prepare_images_for_model(images, model, channel_method="MapaCalor"):
    name = model_name(model)
    preprocessor = PRETRAINED_MODEL_PREPROCESSORS.get(name)
    if preprocessor is None:
        return images
    return preprocessor(to_three_channels(images, channel_method))


def prepare_split_for_model(split, model, channel_method="MapaCalor"):
    split.X_tr = prepare_images_for_model(split.X_tr, model, channel_method)
    split.X_val = prepare_images_for_model(split.X_val, model, channel_method)
    split.X_test = prepare_images_for_model(split.X_test, model, channel_method)
    return split


def save_split_to_png(images, labels, split_name, root="dataset_fold"):
    out_base = Path(root) / split_name
    out_base.mkdir(parents=True, exist_ok=True)

    for idx, (img, cls) in enumerate(zip(images, labels)):
        if img.dtype != np.uint8:
            img = np.clip(img * 255, 0, 255).astype(np.uint8)
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)

        class_dir = out_base / str(cls)
        class_dir.mkdir(parents=True, exist_ok=True)

        fname = class_dir / f"{idx:06d}.png"
        cv2.imwrite(str(fname), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    print(f"{split_name} salvo em {out_base}")


def build_keras_model(model):
    name = model_name(model)
    if name in {"Vgg_16", "Vgg_16_pre_trained"}:
        print("VGG")
        return model().model

    print("ResNet")
    return model()


def train_yolo_classifier(split, fold, seed, epochs=100, patience=50, batch=16):
    dataset_root = f"dataset_fold_{fold+1}"
    save_split_to_png(split.X_tr, split.y_tr, "train", root=dataset_root)
    save_split_to_png(split.X_val, split.y_val, "val", root=dataset_root)
    save_split_to_png(split.X_test, split.y_test, "test", root=dataset_root)

    model_f = YOLO("yolov8s-cls.pt")
    start_time = time.time()
    model_f.train(
        data=dataset_root,
        epochs=epochs,
        patience=patience,
        batch=batch,
        optimizer="AdamW",
        workers=0,
        pretrained=False,
        amp=False,
        deterministic=True,
        seed=seed,
        project="runs/classify",
        name=f"YOLOv8_cls_fold_{fold+1}_seed_{seed}",
    )
    end_time = time.time()

    metrics = model_f.val(
        data=dataset_root,
        project="runs/classify/val",
        name=f"fold_{fold+1}_seed_{seed}",
        save_json=True,
    )

    results_to_save = {
        "top1_accuracy": metrics.top1,
        "top5_accuracy": metrics.top5,
        "fitness": metrics.fitness,
        "training_time_formatted": f"{end_time - start_time:.2f} s",
        "all_metrics": metrics.results_dict,
        "speed": metrics.speed,
    }
    out_json = f"runs/classify/val/fold_{fold+1}_seed_{seed}/results_fold_{fold+1}_seed_{seed}.json"
    with open(out_json, "w") as f:
        json.dump(results_to_save, f, indent=4)


def train_keras_classifier(
    split,
    model,
    message,
    angle,
    fold,
    batch,
    epochs=500,
    early_stop_patience=50,
    early_stop_min_delta=0.01,
):
    keras_model = build_keras_model(model)
    name = model_name(model)
    ckpt = f"modelos/{name}/{message}_{angle}_F{fold}.h5"
    log_txt = f"history/{name}/{message}_{angle}.txt"
    os.makedirs(os.path.dirname(ckpt), exist_ok=True)
    os.makedirs(os.path.dirname(log_txt), exist_ok=True)

    start_time = time.time()
    history = keras_model.fit(
        split.X_tr,
        split.y_tr,
        epochs=epochs,
        validation_data=(split.X_val, split.y_val),
        batch_size=batch,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=early_stop_patience,
                min_delta=early_stop_min_delta,
                restore_best_weights=True,
            ),
            tf.keras.callbacks.ModelCheckpoint(ckpt, monitor="val_loss", save_best_only=True),
        ],
        verbose=2,
        shuffle=True,
    )
    end_time = time.time()

    best = tf.keras.models.load_model(ckpt, compile=False)
    y_pred = (best.predict(split.X_test) > 0.5).astype(int).ravel()

    acc = accuracy_score(split.y_test, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        split.y_test, y_pred, average="binary", zero_division=0
    )

    with open(log_txt, "a") as f:
        f.write(
            f"Fold {fold:02d}  "
            f"Acc={acc:.4f}  "
            f"Prec={prec:.4f}  "
            f"Rec={rec:.4f}  "
            f"F1={f1:.4f}\n"
            f"Tempo de treinamento={end_time - start_time:.2f} s\n"
        )

    plot_convergence(history, name, angle, fold, message)


def run_with_oom_retries(fn, fold, max_retries=2):
    clear_memory()
    for attempt in range(1, max_retries + 1):
        try:
            fn()
            break
        except (tf.errors.ResourceExhaustedError, RuntimeError) as e:
            error_text = str(e).lower()
            if (
                "out of memory" not in error_text
                and "oom" not in error_text
                and "failed to allocate memory" not in error_text
            ):
                raise

            os.makedirs("logs", exist_ok=True)
            with open("logs/oom_errors.txt", "a") as f:
                f.write(f"[Fold {fold+1}] OOM na tentativa {attempt}\n")

            if attempt == max_retries:
                with open("logs/oom_errors.txt", "a") as f:
                    f.write("Maximo de tentativas atingido. Abortando ...")
                raise
            clear_memory()


def train_model_cv(model, raw_root, message, angle="Frontal", k=5,
                   resize=True, resize_method="BlackPadding", resize_to=224, n_aug=0,
                   batch=8, seed=42, segmenter="none", seg_model_path="",
                   yolo_marker_source="manual_masks",
                   channel_method="MapaCalor", test_type=0, val_size=0.25,
                   max_retries=2, exclude_segmentation_ids=True,
                   segmentation_images_dir="Termografias_Dataset_Segmentação/images",
                   mask_root="Frontal-mask-txt", keras_epochs=500,
                   early_stop_patience=50, early_stop_min_delta=0.01,
                   yolo_epochs=100, yolo_patience=50, yolo_batch=16):
    _ = test_type
    if yolo_marker_source not in {"yolo", "manual_masks"}:
        raise ValueError("yolo_marker_source deve ser 'yolo' ou 'manual_masks'")

    use_manual_marker_masks = segmenter == "yolo" and yolo_marker_source == "manual_masks"
    if use_manual_marker_masks and is_ufpe_dataset(raw_root):
        print("UFPE nao possui mascaras manuais do marcador neste pipeline; usando YOLO no modo original.")

    dataset = load_classification_dataset(
        raw_root,
        angle,
        exclude_segmentation_ids=exclude_segmentation_ids,
        segmentation_images_dir=segmentation_images_dir,
        load_marker_masks=use_manual_marker_masks and not is_ufpe_dataset(raw_root),
        mask_root=mask_root,
    )
    dataset_kind = "ufpe" if is_ufpe_dataset(raw_root) else "uff"
    print(f"Saudaveis: {np.sum(dataset.y==0)}, Doentes: {np.sum(dataset.y==1)}")
    write_seed_log(message, seed)

    split_iterator = enumerate(
        make_tvt_splits(
            dataset.X,
            dataset.y,
            dataset.patient_ids,
            k=k,
            val_size=val_size,
            seed=seed,
        )
    )

    for fold, (tr_idx, va_idx, te_idx) in split_iterator:
        def run_fold():
            split = build_split(dataset, tr_idx, va_idx, te_idx)

            print(f"Shape de treinamento fold {fold} antes do aumento de dados: {split.X_tr.shape}")
            print(f"Shape de validacao fold {fold}: {split.X_val.shape}")
            print(f"Shape de teste fold {fold}: {split.X_test.shape}")

            split, mn, mx = normalize_split(split)
            save_split_metadata(message, angle, fold, tr_idx, va_idx, te_idx, dataset.ids_data, mn, mx)

            split = augment_training_split(split, n_aug, seed, dataset_kind)
            if n_aug > 0:
                write_augmentation_log(fold, split.X_tr, split.y_tr)

            if resize:
                split = resize_split(split, resize_method, resize_to)

            split = apply_optional_segmenter(split, segmenter, seg_model_path)

            if model == "yolo":
                print("Modelo YOLO selecionado.")
                train_yolo_classifier(split, fold, seed, epochs=yolo_epochs, patience=yolo_patience, batch=yolo_batch)
                return

            split = prepare_split_for_model(split, model, channel_method)
            train_keras_classifier(
                split,
                model,
                message,
                angle,
                fold,
                batch,
                epochs=keras_epochs,
                early_stop_patience=early_stop_patience,
                early_stop_min_delta=early_stop_min_delta,
            )

        run_with_oom_retries(run_fold, fold, max_retries=max_retries)
