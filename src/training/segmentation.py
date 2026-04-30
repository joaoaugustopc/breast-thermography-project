from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from ultralytics import YOLO

from src.models.u_net import unet_model
from utils.data_prep import load_imgs_masks_Black_Padding
from utils.stats import precision_score_, recall_score_, accuracy_score_, dice_coef_, iou_




def unet_segmenter(data_train, data_valid, data_test, path_model):
    model = tf.keras.models.load_model(path_model)
    train_predictions = model.predict(data_train, batch_size=4)
    valid_predictions = model.predict(data_valid, batch_size=4)
    test_predictions = model.predict(data_test, batch_size=4)

    masks_train = (train_predictions > 0.5).astype(np.uint8)
    masks_valid = (valid_predictions > 0.5).astype(np.uint8)
    masks_test = (test_predictions > 0.5).astype(np.uint8)

    masks_train = np.squeeze(masks_train, axis=-1)
    masks_valid = np.squeeze(masks_valid, axis=-1)
    masks_test = np.squeeze(masks_test, axis=-1)

    segmented_images_train = data_train * masks_train
    segmented_images_valid = data_valid * masks_valid
    segmented_images_test = data_test * masks_test

    return segmented_images_train, segmented_images_valid, segmented_images_test


def segment_with_yolo(
    X_train,
    X_valid,
    X_test,
    M_train=None,
    M_valid=None,
    M_test=None,
    model_path=None,
):
    """
    Segmenta X_train, X_valid e X_test usando YOLO-Seg.

    Modos suportados:
    - Apenas YOLO (assinatura antiga):
        segment_with_yolo(X_train, X_valid, X_test, model_path)
    - YOLO + mascaras externas:
        segment_with_yolo(X_train, X_valid, X_test, M_train, M_valid, M_test, model_path)

    Quando mascaras externas sao fornecidas, elas sao unidas a mascara de mama
    predita pela YOLO. Quando nao sao fornecidas, a YOLO usa diretamente as
    classes retornadas pelo modelo.
    """

    if model_path is None and isinstance(M_train, (str, Path)) and M_valid is None and M_test is None:
        model_path = M_train
        M_train = M_valid = M_test = None

    if model_path is None:
        raise ValueError("model_path deve ser informado para a segmentacao com YOLO.")

    def prepare_image(img):
        """Prepara imagem para o YOLO: uint8 RGB 3 canais, redimensionada"""
        if img.dtype != np.uint8:
            temp = (img * 255).astype(np.uint8)
        else:
            temp = img.copy()
        if temp.ndim == 2:
            temp = cv2.cvtColor(temp, cv2.COLOR_GRAY2BGR)
        elif temp.ndim == 3 and temp.shape[2] == 1:
            temp = np.repeat(temp, 3, axis=-1)
        return temp

    def prepare_optional_masks(masks, num_images):
        if masks is None:
            return [None] * num_images
        if len(masks) != num_images:
            raise ValueError("A quantidade de mascaras deve corresponder ao numero de imagens.")
        return masks

    def build_external_mask(mask, width, height):
        if mask is None:
            return np.zeros((height, width), dtype=np.uint8)

        if mask.ndim == 3 and mask.shape[2] == 1:
            mask = mask[..., 0]

        binary_mask = (
            (mask > 0.5).astype(np.uint8)
            if mask.dtype != np.uint8
            else (mask > 0).astype(np.uint8)
        )
        return cv2.resize(binary_mask, (width, height), interpolation=cv2.INTER_NEAREST)

    PEITO_ID = 0
    MARCADOR_ID = 1

    def segment_batch(images, external_masks, model):
        segmented = []
        external_masks = prepare_optional_masks(external_masks, len(images))

        for img, external_mask in zip(images, external_masks):
            original = img
            img_prepared = prepare_image(original)
            H, W = img_prepared.shape[:2]

            results = model.predict(img_prepared, verbose=False)
            res = results[0]

            has_masks = (
                res.masks is not None and
                res.masks.data is not None and
                len(res.masks.data) > 0
            )

            union_mask = np.zeros((H, W), dtype=np.uint8)
            target_classes = {PEITO_ID} if external_mask is not None else {PEITO_ID, MARCADOR_ID}

            if has_masks:
                masks_np = res.masks.data.cpu().numpy()
                classes = res.boxes.cls.cpu().numpy().astype(int)

                for m, c in zip(masks_np, classes):
                    if c in target_classes:
                        m_bin = (m > 0.5).astype(np.uint8)
                        m_resized = cv2.resize(m_bin, (W, H), interpolation=cv2.INTER_NEAREST)
                        union_mask |= m_resized

            if union_mask.max() > 0 and external_mask is not None:
                union_mask |= build_external_mask(external_mask, W, H)

            if union_mask.max() > 0:
                mask_float = (union_mask > 0).astype(np.float32)
                if original.ndim == 2:
                    segmented_img = original * mask_float
                elif original.ndim == 3 and original.shape[2] == 1:
                    segmented_img = original * mask_float[..., None]
                else:
                    segmented_img = original * mask_float[..., None]
            else:
                print("Nao encontrou instancias das classes esperadas na YOLO.")
                segmented_img = original

            segmented.append(segmented_img.astype(np.float32))

        return np.array(segmented)

    # Carrega modelo YOLO
    model = YOLO(model_path)

    # Segmenta os três conjuntos
    seg_train = segment_batch(X_train, M_train, model)
    seg_valid = segment_batch(X_valid, M_valid, model)
    seg_test  = segment_batch(X_test, M_test, model)


    def squeeze_channel(x):
        if x.ndim == 4 and x.shape[-1] == 1:
            return x[..., 0]
        return x

    seg_train = squeeze_channel(seg_train)
    seg_valid = squeeze_channel(seg_valid)
    seg_test  = squeeze_channel(seg_test)

    return seg_train, seg_valid, seg_test


def evaluate_segmentation(model_path, x_val, y_val):
    model = tf.keras.models.load_model(model_path)
    pred = (model.predict(x_val) > 0.5).astype(np.uint8)
    true = (y_val > 0.5).astype(np.uint8)

    pred = np.squeeze(pred, axis=-1)

    metrics = {
        "precision": precision_score_(true, pred),
        "recall": recall_score_(true, pred),
        "accuracy": accuracy_score_(true, pred),
        "dice": dice_coef_(true, pred),
        "iou": iou_(true, pred),
    }
    return metrics


def evaluate_yolo_on_folder(model_path, ds_root, split="val", imgsz=(224, 224), thr=0.5):
    """
    ds_root/
      images/val/*.png
      masks/val/*.png
    """
    model = YOLO(model_path)
    img_dir = Path(ds_root) / "images" / split
    msk_dir = Path(ds_root) / "masks" / split

    y_true, y_pred = [], []

    for img_file in img_dir.glob("*"):
        name = img_file.stem
        msk_gt = cv2.imread(str(msk_dir / f"{name}.png"), cv2.IMREAD_GRAYSCALE)
        msk_gt = cv2.resize(msk_gt, imgsz, interpolation=cv2.INTER_NEAREST)
        msk_gt = (msk_gt > 127).astype(np.uint8)

        img = cv2.imread(str(img_file))
        img = cv2.resize(img, imgsz)
        res = model.predict(img, verbose=False)
        canvas = np.zeros(imgsz, np.uint8)
        if res and len(res[0].masks):
            for mask in res[0].masks.data:
                mask = cv2.resize(mask.cpu().numpy(), imgsz)
                canvas |= (mask > thr).astype(np.uint8)

        y_true.append(msk_gt.ravel())
        y_pred.append(canvas.ravel())

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)

    metrics = {
        "precision": precision_score_(y_true, y_pred),
        "recall": recall_score_(y_true, y_pred),
        "accuracy": accuracy_score_(y_true, y_pred),
        "dice": dice_coef_(y_true, y_pred),
        "iou": iou_(y_true, y_pred),
    }
    return metrics


def train_unet_segmentation(
    message,
    angle="Frontal",
    img_dir="data/Termografias_Dataset_Segmentação_Frontal_txt_rounded/images",
    mask_dir="TagsMasks",
    resize_to=224,
    batch_size=8,
    epochs=500,
    patience=50,
):
    print("TREINANDO UNET")
    imgs_train, imgs_valid, masks_train, masks_valid = load_imgs_masks_Black_Padding(
        angle,
        img_dir,
        mask_dir,
        True,
        True,
        resize_to,
    )

    model = unet_model()
    model.summary()

    earlystop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        min_delta=0.01,
        patience=patience,
        verbose=1,
        mode="auto",
    )
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        f"modelos/unet/{message}.h5",
        monitor="val_loss",
        verbose=1,
        save_best_only=True,
        save_weights_only=False,
        mode="auto",
    )

    history = model.fit(
        imgs_train,
        masks_train,
        epochs=epochs,
        validation_data=(imgs_valid, masks_valid),
        callbacks=[checkpoint, earlystop],
        batch_size=batch_size,
        verbose=1,
        shuffle=True,
    )

    plt.figure(figsize=(10, 6))
    plt.plot(history.history["loss"], label="Training Loss")
    plt.title(f"Training Loss Convergence for unet - {angle}")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"unet_loss_convergence_{message}.png")
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title(f"Validation Loss Convergence for unet - {angle}")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"unet_val_loss_convergence_{message}.png")
    plt.close()

    return history
