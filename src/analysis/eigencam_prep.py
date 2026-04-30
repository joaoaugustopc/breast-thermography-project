import json

import numpy as np

from src.training.classification import (
    ClassificationDataset,
    apply_optional_segmenter,
    build_split,
    normalize_split_with_bounds,
    resize_split,
)


def prep_eigencam_data(
    X,
    y,
    ids_data,
    split_json,
    resize=True,
    resize_method="GrayPadding",
    resize_to=224,
    segmenter="none",
    seg_model_path="",
    fold=0,
):
    """
    Prepara X_test para geracao de mapas EigenCAM a partir de um split salvo.
    """
    dataset = ClassificationDataset(
        X=np.asarray(X),
        y=np.asarray(y),
        patient_ids=np.asarray(ids_data),
        ids_data=np.asarray(ids_data),
        masks=None,
    )

    with open(split_json, "r") as f:
        split = json.load(f)

    tr_idx = np.array(split["train_idx"])
    va_idx = np.array(split["val_idx"])
    te_idx = np.array(split["test_idx"])
    mn, mx = split["mn_train_pixel"], split["mx_train_pixel"]
    split_data = build_split(dataset, tr_idx, va_idx, te_idx)

    split_data = normalize_split_with_bounds(split_data, mn, mx)

    if resize:
        split_data = resize_split(split_data, resize_method, resize_to)

    split_data = apply_optional_segmenter(split_data, segmenter, seg_model_path)
    X_test = np.expand_dims(split_data.X_test, axis=-1)
    y_test = split_data.y_test
    ids_test = dataset.ids_data[te_idx]

    return X_test, y_test, ids_test
