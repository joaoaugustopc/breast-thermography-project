from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.utils import custom_object_scope

from src.models.resNet_34 import ResidualUnit


def run_eigencam(imgs, labels, masks=None, model_path="", out_dir="cam_out", layer_name=None, ids=None):
    """
    imgs  : np.ndarray (N,H,W,1) normalizado 0-1
    masks : np.ndarray (N,H,W) binario OU None
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    num_imgs, height, width, _ = imgs.shape
    if masks is not None and masks.shape != (num_imgs, height, width):
        raise ValueError("Mascaras e imagens tem shapes incompatíveis")

    with custom_object_scope({"ResidualUnit": ResidualUnit}):
        model = tf.keras.models.load_model(model_path, compile=False)

    if layer_name is None:
        for layer in reversed(model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                layer_name = layer.name
                break
    print(f"[INFO] Usando camada '{layer_name}'")

    feat = model.get_layer(layer_name).output
    cam_model = tf.keras.Model(model.input, {"logits": model.output, "feat": feat})

    def calc_cam(img1):
        out = cam_model(img1[None])
        feat_tensor = tf.transpose(out["feat"], [0, 3, 1, 2])
        feat_tensor = tf.cast(feat_tensor, tf.float32)
        singular, left, right = tf.linalg.svd(feat_tensor)
        cam = left[..., :1] @ singular[..., :1, None] @ tf.transpose(right[..., :1], [0, 1, 3, 2])
        cam = tf.reduce_sum(cam, 1)[0]
        cam -= tf.reduce_min(cam)
        cam /= tf.reduce_max(cam) + 1e-8
        cam = cv2.resize(cam.numpy(), (width, height))
        return cam, out["logits"][0].numpy()

    def to_rgb(gray):
        gray255 = (gray.squeeze() * 255).astype(np.uint8)
        return np.repeat(gray255[..., None], 3, axis=2)

    def heatmap_rgb(cam):
        color = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
        return color[:, :, ::-1]

    def mix(base, heat, alpha=0.4):
        return np.uint8((1 - alpha) * base + alpha * heat)

    def overlap(cam, mask, thr=0.5):
        hot = cam >= thr
        return (hot & mask).sum() / (hot.sum() + 1e-6)

    scores = []
    for index in range(num_imgs):
        cam, logit = calc_cam(imgs[index])

        rgb = to_rgb(imgs[index])
        heat = heatmap_rgb(cam)
        overlay = mix(rgb, heat)

        diagnostic = "Health" if labels[index] == 0 else "Sick"
        img_id = ids[index] if ids is not None else index
        path = Path(out_dir) / diagnostic / f"id_{img_id}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(overlay).save(path)

        if masks is not None:
            score = overlap(cam, masks[index])
            scores.append(score)
            print(f"img_{img_id}: prob={logit.squeeze():.3f}  overlap={score:.3f}")
        else:
            print(f"img_{img_id}: prob={logit.squeeze():.3f}")

    if scores:
        print(f"\nOverlap medio: {np.mean(scores):.3f} +- {np.std(scores):.3f}")


def run_eigencam_for_predictions(
    imgs,
    y_true,
    y_pred,
    ids,
    model_path,
    output_root,
    message,
):
    """
    Gera mapas EigenCAM separados em acertos e erros.
    """
    hits = y_pred == y_true
    miss = y_pred != y_true
    output_root = Path(output_root)

    results = {}
    for bucket, mask in (("Acertos", hits), ("Erros", miss)):
        out_dir = output_root / bucket / message
        out_dir.mkdir(parents=True, exist_ok=True)
        results[bucket.lower()] = str(out_dir)

        if np.any(mask):
            run_eigencam(
                imgs=imgs[mask],
                labels=y_true[mask],
                ids=ids[mask],
                model_path=model_path,
                out_dir=str(out_dir),
            )
        else:
            print(f"[INFO] Nenhuma amostra em {bucket} para {message}.")

    return results
