from pathlib import Path

import cv2
import tensorflow as tf
from tqdm import tqdm

from utils.data_prep import tf_letterbox, tf_letterbox_black, yolo_data


def resize_imgs_masks_dataset(
    img_dir: str,
    mask_dir: str,
    output_base: str,
    target: int = 224,
    resize_method="BlackPadding",
):
    """
    Resize paired RGB images and binary masks into an aligned square dataset.

    Output layout:
        output_base/images/*.jpg
        output_base/masks/*.png
    """
    img_dir = Path(img_dir)
    mask_dir = Path(mask_dir)
    out_img = Path(output_base) / "images"
    out_mask = Path(output_base) / "masks"
    out_img.mkdir(parents=True, exist_ok=True)
    out_mask.mkdir(parents=True, exist_ok=True)

    for img_path in tqdm(sorted(img_dir.glob("*.jpg")), desc="Redimensionando"):
        stem = img_path.stem
        mask_path = mask_dir / f"{stem}.png"
        if not mask_path.exists():
            print(f"[aviso] Mascara ausente para {stem} - pulando.")
            continue

        img = tf.image.decode_jpeg(tf.io.read_file(str(img_path)), channels=3)
        mask = tf.image.decode_png(tf.io.read_file(str(mask_path)), channels=1)

        img = tf.image.convert_image_dtype(img, tf.float32)
        mask = tf.image.convert_image_dtype(mask, tf.float32)

        if resize_method == "BlackPadding":
            img_lb = tf_letterbox_black(tf.expand_dims(img, 0), target=target, mode="bilinear")
            mask_lb = tf_letterbox_black(tf.expand_dims(mask, 0), target=target, mode="nearest")
        elif resize_method == "Distorcido":
            img_lb = tf.image.resize(tf.expand_dims(img, 0), (target, target), method="bilinear")
            mask_lb = tf.image.resize(tf.expand_dims(mask, 0), (target, target), method="nearest")
        elif resize_method == "GrayPadding":
            img_lb = tf_letterbox(tf.expand_dims(img, 0), target=target, mode="bilinear")
            mask_lb = tf_letterbox_black(tf.expand_dims(mask, 0), target=target, mode="nearest")
        else:
            raise ValueError("resize_method deve ser 'BlackPadding', 'GrayPadding' ou 'Distorcido'")

        img_lb = tf.squeeze(img_lb, 0)
        mask_lb = tf.squeeze(mask_lb, 0)

        img_uint8 = tf.image.convert_image_dtype(img_lb, tf.uint8, saturate=True)
        mask_bin = tf.cast(mask_lb > 0.5, tf.uint8) * 255

        tf.io.write_file(str(out_img / f"{stem}.jpg"), tf.io.encode_jpeg(img_uint8, quality=95))
        tf.io.write_file(str(out_mask / f"{stem}.png"), tf.io.encode_png(mask_bin))

    print(f"\nConcluido! Novas pastas:\n  imagens -> {out_img}\n  mascaras -> {out_mask}")


def resize_imgs_masks_dataset_png(
    img_dir: str,
    mask_dir: str,
    output_base: str,
    target: int = 640,
    resize_method="BlackPadding",
):
    """
    Redimensiona imagens PNG e mascaras alinhadas.
    """
    img_dir = Path(img_dir)
    mask_dir = Path(mask_dir)
    out_img = Path(output_base) / "images"
    out_mask = Path(output_base) / "masks"
    out_img.mkdir(parents=True, exist_ok=True)
    out_mask.mkdir(parents=True, exist_ok=True)

    for img_path in tqdm(sorted(img_dir.glob("*.png")), desc="Redimensionando"):
        stem = img_path.stem
        mask_path = mask_dir / f"{stem}.png"
        if not mask_path.exists():
            print(f"[aviso] Mascara ausente para {stem} - pulando.")
            continue

        img = tf.image.decode_png(tf.io.read_file(str(img_path)), channels=1, dtype=tf.uint16)
        mask = tf.image.decode_png(tf.io.read_file(str(mask_path)), channels=1)

        img = tf.image.convert_image_dtype(img, tf.float32)
        mask = tf.image.convert_image_dtype(mask, tf.float32)

        if resize_method == "BlackPadding":
            img_lb = tf_letterbox_black(tf.expand_dims(img, 0), target=target, mode="bilinear")
            mask_lb = tf_letterbox_black(tf.expand_dims(mask, 0), target=target, mode="nearest")
        elif resize_method == "Distorcido":
            img_lb = tf.expand_dims(img, 0)
            img_lb = tf.image.resize(img_lb, (target, target), method="bilinear")
            mask_lb = tf.expand_dims(mask, 0)
            mask_lb = tf.image.resize(mask_lb, (target, target), method="nearest")
        elif resize_method == "GrayPadding":
            img_lb = tf_letterbox(tf.expand_dims(img, 0), target=target, mode="bilinear")
            mask_lb = tf_letterbox_black(tf.expand_dims(mask, 0), target=target, mode="nearest")
        else:
            raise ValueError(f"resize_method desconhecido: {resize_method}")

        img_lb = tf.squeeze(img_lb, 0)
        mask_lb = tf.squeeze(mask_lb, 0)

        img_uint16 = tf.image.convert_image_dtype(img_lb, tf.uint16, saturate=True)
        mask_bin = tf.cast(mask_lb > 0.5, tf.uint8) * 255

        tf.io.write_file(str(out_img / f"{stem}.png"), tf.io.encode_png(img_uint16))
        tf.io.write_file(str(out_mask / f"{stem}.png"), tf.io.encode_png(mask_bin))

    print(f"\nConcluido! Novas pastas:\n  imagens -> {out_img}\n  mascaras -> {out_mask}")


def resize_imgs_two_masks_dataset(
    img_dir: str,
    mask_breast_dir: str,
    mask_marker_dir: str,
    output_base: str,
    target: int = 640,
    resize_method: str = "BlackPadding",
    min_val_mask: float = 0.5,
):
    """
    Redimensiona imagens JPG e duas mascaras preservando o alinhamento.
    """
    img_dir = Path(img_dir)
    mask_breast_dir = Path(mask_breast_dir)
    mask_marker_dir = Path(mask_marker_dir)

    out_img = Path(output_base) / "images"
    out_mb = Path(output_base) / "masks_breast"
    out_mm = Path(output_base) / "masks_marker"
    for path in (out_img, out_mb, out_mm):
        path.mkdir(parents=True, exist_ok=True)

    for img_path in tqdm(sorted(img_dir.glob("*.jpg")), desc="Redimensionando"):
        stem = img_path.stem
        mb_path = mask_breast_dir / f"{stem}.png"
        mm_path = mask_marker_dir / f"{stem}.png"

        if not (mb_path.exists() and mm_path.exists()):
            print(f"[aviso] Mascara faltando para {stem} - pulando.")
            continue

        img = tf.image.decode_jpeg(tf.io.read_file(str(img_path)), channels=3)
        mb = tf.image.decode_png(tf.io.read_file(str(mb_path)), channels=1)
        mm = tf.image.decode_png(tf.io.read_file(str(mm_path)), channels=1)

        img = tf.image.convert_image_dtype(img, tf.float32)
        mb = tf.image.convert_image_dtype(mb, tf.float32)
        mm = tf.image.convert_image_dtype(mm, tf.float32)

        if resize_method == "BlackPadding":
            img_lb = tf_letterbox_black(tf.expand_dims(img, 0), target, mode="bilinear")
            mb_lb = tf_letterbox_black(tf.expand_dims(mb, 0), target, mode="nearest")
            mm_lb = tf_letterbox_black(tf.expand_dims(mm, 0), target, mode="nearest")
        elif resize_method == "GrayPadding":
            img_lb = tf_letterbox(tf.expand_dims(img, 0), target, mode="bilinear")
            mb_lb = tf_letterbox_black(tf.expand_dims(mb, 0), target, mode="nearest")
            mm_lb = tf_letterbox_black(tf.expand_dims(mm, 0), target, mode="nearest")
        elif resize_method == "Distorcido":
            img_lb = tf.image.resize(tf.expand_dims(img, 0), (target, target), method="bilinear")
            mb_lb = tf.image.resize(tf.expand_dims(mb, 0), (target, target), method="nearest")
            mm_lb = tf.image.resize(tf.expand_dims(mm, 0), (target, target), method="nearest")
        else:
            raise ValueError(f"resize_method desconhecido: {resize_method}")

        img_lb = tf.squeeze(img_lb, 0)
        mb_lb = tf.squeeze(mb_lb, 0)
        mm_lb = tf.squeeze(mm_lb, 0)

        img_uint8 = tf.image.convert_image_dtype(img_lb, tf.uint8, saturate=True)
        mb_bin = tf.cast(mb_lb > min_val_mask, tf.uint8) * 255
        mm_bin = tf.cast(mm_lb > min_val_mask, tf.uint8) * 255

        tf.io.write_file(str(out_img / f"{stem}.jpg"), tf.io.encode_jpeg(img_uint8, quality=95))
        tf.io.write_file(str(out_mb / f"{stem}.png"), tf.io.encode_png(mb_bin))
        tf.io.write_file(str(out_mm / f"{stem}.png"), tf.io.encode_png(mm_bin))

    print(f"\nConcluido!\n  Imagens -> {out_img}\n  Breast -> {out_mb}\n  Marker -> {out_mm}")


def unir_mascaras(pasta_breast, pasta_marker, pasta_saida):
    """
    Une mascaras de duas pastas diferentes e salva a uniao em uma pasta final.
    """
    output_dir = Path(pasta_saida)
    output_dir.mkdir(parents=True, exist_ok=True)

    for breast_path in Path(pasta_breast).iterdir():
        if breast_path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}:
            continue

        marker_path = Path(pasta_marker) / breast_path.name
        if not marker_path.exists():
            print(f"Aviso: nao encontrei {breast_path.name} em {pasta_marker}, pulando...")
            continue

        mask_breast = cv2.imread(str(breast_path), 0)
        mask_marker = cv2.imread(str(marker_path), 0)

        if mask_breast is None or mask_marker is None:
            print(f"Aviso: nao consegui carregar {breast_path.name}, pulando...")
            continue

        combined = cv2.bitwise_or(mask_breast, mask_marker)
        _, combined = cv2.threshold(combined, 127, 255, cv2.THRESH_BINARY)
        cv2.imwrite(str(output_dir / breast_path.name), combined)

    print(f"Mascaras unidas foram salvas em: {pasta_saida}")


def prepare_yolo_segmentation_dataset(
    angle: str,
    img_dir: str,
    mask_dir: str,
    resized_output_dir: str,
    yolo_output_dir: str,
    target: int = 224,
    resize_method: str = "BlackPadding",
    augment: bool = True,
):

    resize_imgs_masks_dataset(
        img_dir=img_dir,
        mask_dir=mask_dir,
        output_base=resized_output_dir,
        target=target,
        resize_method=resize_method,
    )
    yolo_data(
        angle,
        str(Path(resized_output_dir) / "images"),
        str(Path(resized_output_dir) / "masks"),
        yolo_output_dir,
        augment,
    )
