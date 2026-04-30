"""Utilities for converting and restoring temperature datasets."""

import json
import os
import re
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

from utils.data_prep import load_temp_matrix


def txt_to_image(txt_file, output_image_path):
    with open(txt_file, "r") as file:
        lines = file.readlines()

    lines = [line.replace(" ", ";") for line in lines]
    data = [list(map(float, line.strip().split(";"))) for line in lines]
    image_array = np.array(data, dtype=np.float32)
    image_array = cv2.normalize(image_array, None, 0, 255, cv2.NORM_MINMAX)
    image_array = image_array.astype(np.uint8)

    image = Image.fromarray(image_array)
    image.save(output_image_path)
    print(f"Imagem salva em: {output_image_path}")


def segment_and_save_pngdataset(
    model_path,
    input_dir,
    output_dir,
    ext_txt=".txt",
    ext_img=".png",
):
    """
    Segmenta imagens convertidas de arquivos .txt ou .png usando um modelo YOLO.
    """
    model = YOLO(model_path)
    os.makedirs(output_dir, exist_ok=True)

    for file in os.listdir(input_dir):
        if file.endswith(ext_txt):
            txt_path = os.path.join(input_dir, file)
            img_path = os.path.join(input_dir, f"{os.path.splitext(file)[0]}{ext_img}")

            txt_to_image(txt_path, img_path)

            img = cv2.imread(img_path)
            if img is None:
                print(f"[Erro] Nao foi possivel carregar a imagem: {img_path}")
                continue

            height, width, _ = img.shape
            results = model(img)

            for result in results:
                if result.masks is None:
                    continue
                for index, mask in enumerate(result.masks.data):
                    mask_np = mask.cpu().numpy() * 255
                    mask_resized = cv2.resize(mask_np, (width, height))
                    segmented = cv2.bitwise_and(
                        img,
                        img,
                        mask=mask_resized.astype(np.uint8),
                    )
                    out_path = os.path.join(
                        output_dir,
                        f"{os.path.splitext(file)[0]}_seg_{index}{ext_img}",
                    )
                    cv2.imwrite(out_path, segmented)
                    print(f"[Salvo] {out_path}")


def transform_temp_img_png16(input_folder, output_folder, mn=None, mx=None):
    os.makedirs(output_folder, exist_ok=True)
    limites = {}

    for fname in os.listdir(input_folder):
        if fname.endswith(".txt"):
            path = os.path.join(input_folder, fname)
            temperatura = load_temp_matrix(path)

            if mn is None and mx is None:
                temp_min, temp_max = float(temperatura.min()), float(temperatura.max())
                limites[fname] = {"min": temp_min, "max": temp_max}
                norm = ((temperatura - temp_min) / (temp_max - temp_min) * 65535).astype(np.uint16)
            else:
                limites[fname] = {"min": mn, "max": mx}
                norm = ((temperatura - mn) / (mx - mn) * 65535).astype(np.uint16)

            out_name = os.path.splitext(fname)[0] + ".png"
            cv2.imwrite(os.path.join(output_folder, out_name), norm)

    with open(os.path.join(output_folder, "limites.json"), "w") as file:
        json.dump(limites, file, indent=4)


def recuperar_img(input_folder, output_folder):
    """
    Converte imagens PNG 16-bit de volta para matrizes de temperatura.
    """
    os.makedirs(output_folder, exist_ok=True)

    json_path = os.path.join(input_folder, "limites.json")
    if not os.path.isfile(json_path):
        raise FileNotFoundError(f"JSON de limites nao encontrado em: {json_path}")

    with open(json_path, "r") as file:
        limites = json.load(file)

    for fname in os.listdir(input_folder):
        if not fname.lower().endswith(".png"):
            continue

        path = os.path.join(input_folder, fname)
        editada = cv2.imread(path, cv2.IMREAD_UNCHANGED)

        if editada is None:
            print(f"Aviso: nao foi possivel ler {fname}. Pulando.")
            continue

        if editada.dtype == np.uint8:
            gray_8 = cv2.cvtColor(editada, cv2.COLOR_BGR2GRAY)
            editada = gray_8.astype(np.uint16) * 257

        if editada.ndim == 3:
            editada = editada[:, :, 0]

        original_txt = os.path.splitext(fname)[0] + ".txt"
        if original_txt not in limites:
            print(
                f"AVISO: chave {original_txt} nao encontrada. "
                f"Chaves do JSON: {list(limites.keys())[:10]}"
            )
            continue

        temp_min = limites[original_txt]["min"]
        temp_max = limites[original_txt]["max"]
        recuperada = editada.astype(np.float32) / 65535.0 * (temp_max - temp_min) + temp_min

        out_name = os.path.splitext(fname)[0] + ".txt"
        np.savetxt(os.path.join(output_folder, out_name), recuperada, fmt="%.6f")

def gerar_limites_originais_txt(input_folder, output_json):
    """
    Gera limites min/max de cada arquivo TXT com matriz de temperatura.
    """
    limites = {}

    for fname in os.listdir(input_folder):
        if fname.lower().endswith(".txt"):
            path = os.path.join(input_folder, fname)

            with open(path, "r") as file:
                first_line = file.readline()

            delimiter = None
            if ";" in first_line:
                delimiter = ";"
            elif "," in first_line:
                delimiter = ","

            if delimiter:
                temperatura = np.loadtxt(path, delimiter=delimiter)
            else:
                temperatura = np.loadtxt(path)

            temp_min, temp_max = float(temperatura.min()), float(temperatura.max())
            limites[fname] = {"min": temp_min, "max": temp_max}
            print(f"Adicionado: {fname} -> min: {temp_min}, max: {temp_max}")

    if not limites:
        print("AVISO: Nenhum arquivo TXT foi processado! Verifique o diretorio.")

    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w") as file:
        json.dump(limites, file, indent=4)

    print(f"Arquivo de limites salvo em: {output_json}")


def get_imgs_lim_seg_data(input_folder):
    """
    Salva os limites das imagens do dataset de segmentacao.
    """
    pattern = re.compile(
        r"^T0*(\d+)\.\d+\.\d+\.[A-Z]\.(\d{4}-\d{2}-\d{2})\.\d{2}\.png$",
        re.IGNORECASE,
    )

    with open("limites_raw/Frontal/healthy/limites.json", "r") as file:
        healthy_limits = json.load(file)
    with open("limites_raw/Frontal/sick/limites.json", "r") as file:
        sick_limits = json.load(file)

    limitesdump = {}
    for file in os.listdir(input_folder):
        match = pattern.match(file)
        if not match:
            continue

        img_id = match.group(1)
        date = match.group(2)
        key = f"{img_id}_img_Static-Frontal_{date}.txt"

        if healthy_limits.get(key) is not None:
            print(f"LIMITES1: {healthy_limits[key]['min']} | {healthy_limits[key]['max']}")
            limitesdump[file.replace(".png", ".txt")] = {
                "min": healthy_limits[key]["min"],
                "max": healthy_limits[key]["max"],
            }
        elif sick_limits.get(key) is not None:
            print(f"LIMITES2: {sick_limits[key]['min']} | {sick_limits[key]['max']}")
            limitesdump[file.replace(".png", ".txt")] = {
                "min": sick_limits[key]["min"],
                "max": sick_limits[key]["max"],
            }
        else:
            print(f"NAO encontrou {key}")

    path = os.path.join(input_folder, "limites.json")
    with open(path, "w") as file:
        json.dump(limitesdump, file, indent=4)
