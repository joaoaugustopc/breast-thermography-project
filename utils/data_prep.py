from include.imports import *
import os
import random
import re
import shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold, train_test_split

#Semente utilizada para criação dos datasets < USAR APENAS QUANDO FOR CRIAR UM NOVO DATASET
#-> EXISTE UMA DEFINIÇÃO DE SEMENTE NO ARQUIVO MAIN.PY >
"""
SEED = 36f
np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)
"""
#Arrays Numpy
def load_data(angulo, folder = ""):

    imagens_train = np.load(f"{folder}/imagens_train_{angulo}.npy")
    labels_train = np.load(f"{folder}/labels_train_{angulo}.npy")
    imagens_valid = np.load(f"{folder}/imagens_valid_{angulo}.npy")
    labels_valid = np.load(f"{folder}/labels_valid_{angulo}.npy")
    imagens_test = np.load(f"{folder}/imagens_test_{angulo}.npy")
    labels_test = np.load(f"{folder}/labels_test_{angulo}.npy")


    return imagens_train, labels_train, imagens_valid, labels_valid, imagens_test, labels_test


def preprocess(image, max, min):
    image = (image - min) / (max - min)
    return image


_pattern_id = re.compile(r'^(\d+)_.*_(\d{4}-\d{2}-\d{2})')
CLASS_LABELS = (("healthy", 0), ("sick", 1))

def extract_id_data(filename):

    m = _pattern_id.match(filename)
    if not m:
        raise ValueError(f"Nome inesperado: {filename}")
    
    id = int(m.group(1))
    data = m.group(2)
    id_data = f"{id}_{data}"
    # Extrair o ID a partir do nome do arquivo
    return id, data, id_data

"""
Função para converter os arquivos de texto em arrays numpy
"""
def to_array(directory, exclude = False, exclude_set = None):

    arquivos = os.listdir(directory)

    print(arquivos)

    healthy_path = os.path.join(directory, 'healthy')
    sick_path = os.path.join(directory, 'sick')

    healthy = os.listdir(healthy_path)
    sick = os.listdir(sick_path)

    imagens = []
    labels = []
    ids = []
    max_value = 0
    min_value = 1000

    for arquivo in healthy:

        if exclude and arquivo in exclude_set:
            print(f"Excluindo {arquivo}")
            continue

        path = os.path.join(healthy_path, arquivo)
        try:
          with open(path, 'r') as f:
            primeira_linha = f.readline()
            if ';' in primeira_linha:
              delimiter = ';'
            else:
              delimiter = ' '
          imagem = np.loadtxt(path, delimiter=delimiter)
          imagens.append(imagem)
          labels.append(0)
          _id, _data, _id_data = extract_id_data(arquivo)
          ids.append(_id)
        except ValueError as e:
          print(e)
          print(arquivo)
          continue

    for arquivo in sick:

        if exclude and arquivo in exclude_set:
            print(f"Excluindo {arquivo}")
            continue

        path = os.path.join(sick_path, arquivo)
        try:
          with open(path, 'r') as f:
            primeira_linha = f.readline()
            if ';' in primeira_linha:
              delimiter = ';'
            else:
              delimiter = ' '
            f.seek(0)
          imagem = np.loadtxt(path, delimiter=delimiter)
          imagens.append(imagem)
          labels.append(1)
          _id, _data, _id_data = extract_id_data(arquivo)
          ids.append(_id)
        except ValueError as e:
          print(e)
          print(arquivo)
          continue

  
    unique_ids = set()
    ids_unicos = []
    imagens_unicas = []
    labels_unicos = []

    consulta_count = {}

    mult_consultas_img = []
    mult_consultas_label = []
    mult_consultas_id = []

    #contar o número de consultas por paciente  
    for id in ids:
        if id in consulta_count:
            consulta_count[id] += 1
        else:
            consulta_count[id] = 1
    

    for imagem, label, id in zip(imagens, labels, ids):
        if consulta_count[id] == 1:
            imagens_unicas.append(imagem)
            labels_unicos.append(label)
            ids_unicos.append(id)
        else:
            mult_consultas_img.append(imagem)
            mult_consultas_label.append(label)
            mult_consultas_id.append(id)

      
    total_imgs = len(imagens)
    total_imgs_unicas = len(imagens_unicas)
    total_mult_imgs = len(mult_consultas_img)

    mult_porcent = (total_mult_imgs / total_imgs)

    train_percent = max(0.0,0.6 - mult_porcent)

    imagens_unicas = np.array(imagens_unicas)
    labels_unicos = np.array(labels_unicos)

    # Primeiro split: 60% treino, 40% restante
    imagens_train, imagens_rest, labels_train, labels_rest = train_test_split(imagens_unicas, labels_unicos, test_size= 1 - train_percent, shuffle = True)

    imagens_train = np.concatenate((imagens_train, mult_consultas_img), axis=0)
    labels_train = np.concatenate((labels_train, mult_consultas_label), axis=0)

    permutation = np.random.permutation(len(imagens_train))
    imagens_train = imagens_train[permutation]
    labels_train = labels_train[permutation]

    # Segundo split: 50% validação, 50% teste do restante (que é 20% cada do total original)

    imagens_valid, imagens_test, labels_valid, labels_test = train_test_split(imagens_rest, labels_rest, test_size=0.5, shuffle = True)


    for i in range(len(imagens_train)):
        max_value = max(max_value, np.max(imagens_train[i]))
        min_value = min(min_value, np.min(imagens_train[i]))


    for i in range(len(imagens_train)):
        imagens_train[i] = preprocess(imagens_train[i], max_value, min_value)

    for i in range(len(imagens_valid)):
        imagens_valid[i] = preprocess(imagens_valid[i], max_value, min_value)

    for i in range(len(imagens_test)):
        imagens_test[i] = preprocess(imagens_test[i], max_value, min_value)

    return imagens_train, labels_train, imagens_valid, labels_valid, imagens_test, labels_test


def format_data(directory_raw, output_dir = "output_dir", exclude = False, exclude_path = ""):
    for angle in ["Frontal", "Right45", "Right90", "Left45", "Left90"]:

        if exclude:
            list = set()

            list = listar_imgs_nao_usadas(exclude_path, angle)
            
            imagens_train, labels_train, imagens_valid, labels_valid, imagens_test, labels_test = to_array(f"{directory_raw}/{angle}", True, list)

        else:
            imagens_train, labels_train, imagens_valid, labels_valid, imagens_test, labels_test = to_array(f"{directory_raw}/{angle}")

        print("ANGLE:",angle)
        print("Train shape:",imagens_train.shape)
        print(labels_train.shape)
        print("valid shape:",imagens_valid.shape)
        print(labels_valid.shape)
        print("test shape:",imagens_test.shape)
        print(labels_test.shape)

        print("Train Healthy:",len(labels_train[labels_train == 0]))
        print("Train Sick:",len(labels_train[labels_train == 1]))
        print("Valid Healthy:",len(labels_valid[labels_valid == 0]))
        print("Valid Sick:",len(labels_valid[labels_valid == 1]))
        print("Test Healthy:",len(labels_test[labels_test == 0]))
        print("Test Sick:",len(labels_test[labels_test == 1]))

        if not os.path.exists(f"{output_dir}"):
            os.makedirs(f"{output_dir}")

        np.save(f"{output_dir}/imagens_train_{angle}.npy", imagens_train)
        np.save(f"{output_dir}/labels_train_{angle}.npy", labels_train)
        np.save(f"{output_dir}/imagens_valid_{angle}.npy", imagens_valid)
        np.save(f"{output_dir}/labels_valid_{angle}.npy", labels_valid)
        np.save(f"{output_dir}/imagens_test_{angle}.npy", imagens_test)
        np.save(f"{output_dir}/labels_test_{angle}.npy", labels_test)    


def apply_mask():
    masks = os.listdir('masks')

    regex = re.compile(r'.*1\.S.*')

    mascaras = [mask for mask in masks if regex.match(mask)]

    max_value = 0
    img_max = None

    for mask in mascaras:
        img = Image.open(os.path.join('masks', mask))
        img = np.array(img)

        n_true = np.count_nonzero(img)

        if n_true > max_value:
            max_value = n_true
            img_max = img


    imagens_train = np.load('np_dataset/imagens_train_Frontal.npy')

    lista = []

    for imagem in imagens_train:

        masked = np.ma.masked_array(imagem, ~img_max)
        filled = np.ma.filled(masked, 0)
        lista.append(filled)

    masked_array = np.array(lista)

    np.save('imagens_train_Frontal_masked.npy', masked_array)

    
# Função para salvar imagens e rótulos em arquivos NumPy
def save_numpy_data(images, labels, output_dir, position):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Salvar imagens e labels em arquivos separados
    np.save(os.path.join(output_dir, f"imagens_train_aug_{position}"), images)
    np.save(os.path.join(output_dir, f"labels_train_aug_{position}"), labels)
    print(f"Imagens e labels salvos em {output_dir}")


# Funções lambda para cada transformação separada
random_flip =  lambda s=None: keras.layers.RandomFlip("horizontal", seed = s)
random_rotation =  lambda s=None: keras.layers.RandomRotation(0.04, interpolation="bilinear",seed=s)
random_rotation_mask =  lambda s=None: keras.layers.RandomRotation(0.04, interpolation="nearest",seed=s)
random_zoom =  lambda s=None: keras.layers.RandomZoom(0.2, 0.2, interpolation="bilinear",seed=s)
random_zoom_mask =  lambda s=None: keras.layers.RandomZoom(0.2, 0.2, interpolation="nearest",seed=s)
random_brightness =  lambda s=None: keras.layers.RandomBrightness(factor=0.3, value_range=(0.0, 1.0),seed=s)
random_contrast =  lambda s=None: keras.layers.RandomContrast(factor=0.3,seed=s)

# Função para aplicar as transformações individualmente
def apply_transformation(image, transformation, seed_value = None):
    return transformation(seed_value)(image).numpy()

"""
    Params:
    train: conjuntos de imagens de treino original
    labels: conjunto de labels de treino original
    num_augmented_copies: números de vezes para expandir o dataset
    resize: bool -> para redimensionar
    targert_size: int -> novo tamanho para redimensionar
    
    return:
    all_imagens, all_labels : imagens e labels originais + aug
"""
def apply_augmentation_and_expand(train, labels, num_augmented_copies, seed =42, resize=False, target_size=0):

    print("Aumentando o dataset com cópias aumentadas...")
    print("Shape original das imagens:", train.shape)

    #VALUE_SEED = int(time.time() * 1000) % 15000
    #VALUE_SEED = 12274
    random.seed(seed)

    
    train = np.expand_dims(train, axis=-1)
    
    all_images = []
    all_labels = []
    
    
    for image, label in zip(train, labels):
        all_images.append(image)
        all_labels.append(label)
    
    #transformations = [random_rotation, random_zoom, random_brightness, random_contrast]
    transformations = [random_rotation, random_zoom, random_brightness]


    
    for _ in range(num_augmented_copies):
        i = 0
        for image, label in zip(train, labels):
            for transformation in transformations:  
                
                seed = random.randint(0, 10000)
                
                random.seed(seed)
                
                augmented_image = apply_transformation(image, transformation, seed)
                
                
                if random.random() > 0.5:  # 50% de chance de aplicar o flip
                    random.seed(seed)
                    augmented_image = apply_transformation(augmented_image, random_flip, seed)
                

                all_images.append(augmented_image)
                all_labels.append(label)
                
               
    
    all_images = np.array([img for img in all_images])
    all_labels = np.array(all_labels)
    
    
    if resize:
        all_images = tf.image.resize_with_pad(all_images, target_size, target_size, method="bilenear")
    
    
    all_images = np.squeeze(all_images, axis=-1)
        
    print(all_images.shape)
    print(all_labels[all_labels == 1].shape)
    print(all_labels[all_labels == 0].shape)
    
    #teste
    #visualize_augmentation(all_images[:10], all_images[156:166], 10)

    print("Shape das imagens aumentadas:", all_images.shape)
    
    
    return all_images, all_labels

def apply_augmentation_and_expand_with_masks(train_imgs, train_masks, labels, num_augmented_copies, seed =42, resize=False, target_size=0):

    print("Aumentando o dataset com cópias aumentadas...")
    print("Shape original das imagens:", train_imgs.shape)
    print("Shape original das mascaras:", train_masks.shape)


    #VALUE_SEED = int(time.time() * 1000) % 15000
    #VALUE_SEED = 12274
    random.seed(seed)

    
    train_imgs = np.expand_dims(train_imgs, axis=-1)
    train_masks = np.expand_dims(train_masks, axis=-1)
    
    
    # listas para armazenar as imagens e labels
    all_images = []
    all_masks = []
    all_labels = []
    
    #adicionar as imagens e labels originais
    for image, mask, label in zip(train_imgs, train_masks, labels):
        all_images.append(image)
        all_masks.append(mask)
        all_labels.append(label)
    
    #transformations = [random_rotation, random_zoom, random_brightness, random_contrast]
    transformations = [random_rotation, random_zoom, random_brightness]
    transformations_mask = { 
        random_rotation: random_rotation_mask,
        random_zoom: random_zoom_mask,
    }


    #aplicar cada transformação separado
    for _ in range(num_augmented_copies):
        i = 0
        for image, mask, label in zip(train_imgs, train_masks, labels):
            for transformation in transformations:  # Aplicar cada transformação separadamente
                
                seed = random.randint(0, 10000)
                
                random.seed(seed)
                
                augmented_image = apply_transformation(image, transformation, seed)

                if transformation != random_brightness:
                    random.seed(seed)
                    augmented_mask = apply_transformation(mask, transformations_mask[transformation], seed)
                else:
                    augmented_mask = mask
                
                # Aleatoriamente decidir se vai aplicar random_flip
                if random.random() > 0.5:  # 50% de chance de aplicar o flip
                    random.seed(seed)
                    augmented_image = apply_transformation(augmented_image, random_flip, seed)
                    random.seed(seed)
                    augmented_mask = apply_transformation(augmented_mask, random_flip, seed)
                

                all_images.append(augmented_image)
                all_masks.append(augmented_mask)
                all_labels.append(label)
    
    #convertendo em np
    all_images = np.array([img for img in all_images])
    all_masks = np.array([mask for mask in all_masks])
    all_labels = np.array(all_labels)
    
    #se passado como parametro
    if resize:
        all_images = tf.image.resize_with_pad(all_images, target_size, target_size, method="bilinear")
        all_masks = tf.image.resize_with_pad(all_masks, target_size, target_size, method="nearest")
    
    
    all_images = np.squeeze(all_images, axis=-1)
    all_masks = np.squeeze(all_masks, axis=-1)

    all_masks = (all_masks > 0.5).astype(np.uint8)
        
    print(all_images.shape)
    print(all_masks.shape)
    print(all_labels[all_labels == 1].shape)
    print(all_labels[all_labels == 0].shape)
    
    #teste
    #visualize_augmentation(all_images[:10], all_images[156:166], 10)

    print("Shape das imagens aumentadas:", all_images.shape)
    
    assert len(all_images) == len(all_masks) == len(all_labels)
    
    return all_images, all_masks, all_labels  

#ajustada para aumentar o sick da ufpe
def apply_augmentation_and_expand_ufpe(train, labels, num_augmented_copies, seed=42, resize=False, target_size=0):

    print("Aumentando o dataset com cópias aumentadas...")
    print("Shape original das imagens:", train.shape)
    print("Shape original das máscaras:", labels.shape)

    random.seed(seed)
    train = np.expand_dims(train, axis=-1)
    
    all_images = []
    all_labels = []
    
    # Adiciona as imagens e labels originais
    for image, label in zip(train, labels):
        all_images.append(image)
        all_labels.append(label)
    
    transformations = [random_rotation, random_zoom, random_brightness]

    # Aplica cada transformação separadamente
    for _ in range(num_augmented_copies):
        for image, label in zip(train, labels):
            for transformation in transformations:
                aug_seed = random.randint(0, 10000)
                random.seed(aug_seed)
                augmented_image = apply_transformation(image, transformation)
                
                # 50% de chance de flip
                if random.random() > 0.5:
                    random.seed(aug_seed)
                    augmented_image = apply_transformation(augmented_image, random_flip)
                
                all_images.append(augmented_image)
                all_labels.append(label)
                
                # Se for sick, faz o aumento mais uma vez
                if label == 1:
                    extra_seed = random.randint(0, 10000)
                    random.seed(extra_seed)
                    extra_augmented = apply_transformation(image, transformation)
                    if random.random() > 0.5:
                        random.seed(extra_seed)
                        extra_augmented = apply_transformation(extra_augmented, random_flip)
                    all_images.append(extra_augmented)
                    all_labels.append(label)
    
    all_images = np.array([img for img in all_images])
    all_labels = np.array(all_labels)
    
    if resize:
        all_images = tf.image.resize_with_pad(all_images, target_size, target_size, method="bilinear")
    
    all_images = np.squeeze(all_images, axis=-1)
        
    print(all_images.shape)
    print(all_labels[all_labels == 1].shape)
    print(all_labels[all_labels == 0].shape)
    print("Shape das imagens aumentadas:", all_images.shape)
    
    return all_images, all_labels

"""
Params:
    original_img: dataset original
    aug_img: dataset alterado (aug e resize)
    num_images: imagens visualizadas
    
    Return:
    void -> mas as imagens são salvas
"""
def visualize_augmentation(original_img, aug_img, num_images=5):
    
    plt.figure(figsize=(15, 6))         
    for i in range(num_images):
        # Imagem Original
        plt.subplot(2, num_images, i + 1)
        plt.imshow(original_img[i].astype('float32'))
        plt.title("Original")
        plt.axis("off")
        
        # Imagem Augmentada
        plt.subplot(2, num_images, i + 1 + num_images)
        plt.imshow(aug_img[i].astype("float32"))
        plt.title("Augmentada")
        plt.axis("off")
    plt.savefig("foto_aug/")


def create_aug_dataset(val_aug, input_dir ="", output_dir=""):
    
    angles_list = ["Frontal", "Left45", "Left90", "Right45", "Right90"]
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for position in angles_list:
        imagens_train, labels_train, imagens_valid, labels_valid, imagens_test, labels_test = load_data(position, input_dir)
        imagens_train, labels_train = apply_augmentation_and_expand(imagens_train, labels_train, val_aug, resize=False)
        
        #salvar imagens e labels em arquivos separados
        np.save(os.path.join(output_dir, f"imagens_train_{position}"), imagens_train)
        np.save(os.path.join(output_dir, f"labels_train_{position}"), labels_train)
                
        np.save(os.path.join(output_dir, f"imagens_valid_{position}"), imagens_valid)
        np.save(os.path.join(output_dir, f"labels_valid_{position}"), labels_valid)
        
        np.save(os.path.join(output_dir, f"imagens_test_{position}"), imagens_test)
        np.save(os.path.join(output_dir, f"labels_test_{position}"), labels_test)
        
        #verificação
        print(imagens_train.shape)
        print(labels_train.shape)






def filtrar_imgs_masks(angulo, img_path, mask_path):
    """
    Filtra imagens e máscaras baseado no ângulo especificado.
    Retorna listas de caminhos das imagens e máscaras.
    """
    angulos = {
        "Frontal": "1",
        "Left45": "4",
        "Right45": "2",
        "Left90": "5",
        "Right90": "3"
    }

    idx_angle = angulos.get(angulo)
    if not idx_angle:
        raise ValueError(f"Ângulo inválido: {angulo}")

    pattern = re.compile(f".*{idx_angle}\.S.*")

    imgs_files = sorted([img for img in os.listdir(img_path) if pattern.match(img)])
    masks_files = sorted([mask for mask in os.listdir(mask_path) if pattern.match(mask)])

    assert len(imgs_files) == len(masks_files), "Número de imagens e máscaras não corresponde!"

    data_imgs = [os.path.join(img_path, img) for img in imgs_files]
    data_masks = [os.path.join(mask_path, mask) for mask in masks_files]

    return data_imgs, data_masks


def load_imgs_masks(angulo, img_path, mask_path, augment=False, resize=False, target = 224):
    """
    Carrega imagens e máscaras em formato numpy arrays normalizados (0 a 1).
    Retorna as listas imgs_train, imgs_valid, masks_train, masks_valid.
    """
    data_imgs, data_masks = filtrar_imgs_masks(angulo, img_path, mask_path)

    imagens = [np.array(Image.open(img).convert('L')) / 255.0 for img in data_imgs]

    mascaras = [np.array(Image.open(mask).convert('L')) / 255.0 for mask in data_masks]

    if resize:

        imagens = np.expand_dims(imagens, axis=-1)
        mascaras = np.expand_dims(mascaras, axis=-1)

        # imagens = tf.image.resize_with_pad(imagens, target, target, method="bilinear")
        # mascaras = tf.image.resize_with_pad(mascaras, target, target, method="nearest")

        imagens = tf_letterbox(imagens, target, mode="bilinear")
        mascaras = tf_letterbox_black(mascaras, target, mode="nearest")


        imagens = np.squeeze(imagens, axis=-1)
        mascaras = np.squeeze(mascaras, axis=-1)

    

    imgs_train, imgs_valid, masks_train, masks_valid = train_test_split(
        imagens, mascaras, test_size=0.2, random_state=42
    )

    imgs_train = np.array(imgs_train)
    masks_train = np.array(masks_train)
    imgs_valid = np.array(imgs_valid)
    masks_valid = np.array(masks_valid)

    print("Train shape:", imgs_train.shape)
    print("Valid shape:", imgs_valid.shape)

    if augment:
        imgs_train_aug, masks_train_aug = apply_augmentation_and_expand_seg(
            imgs_train, masks_train, num_augmented_copies = 2, resize=False
        )

        return imgs_train_aug, imgs_valid, masks_train_aug, masks_valid
    
    return imgs_train, imgs_valid, masks_train, masks_valid

def load_imgs_masks_only(angulo, img_path, mask_path):
    """
    Carrega imagens e máscaras em formato numpy arrays normalizados (0 a 1).
    Retorna as listas imagens e masks
    """
    data_imgs, data_masks = filtrar_imgs_masks(angulo, img_path, mask_path)

    imagens = [np.array(Image.open(img).convert('L')) / 255.0 for img in data_imgs]
    mascaras = [np.array(Image.open(mask).convert('L')) / 255.0 for mask in data_masks]

    return np.array(imagens), np.array(mascaras)


def criar_pastas_yolo(Directory = ""):
    """
    Cria a estrutura de diretórios para armazenar as imagens e máscaras do YOLO.
    """
    pastas = [
        f"{Directory}",
        f"{Directory}/images",
        f"{Directory}/images/train",
        f"{Directory}/images/val",
        f"{Directory}/masks",
        f"{Directory}/masks/train",
        f"{Directory}/masks/val",
        f"{Directory}/labels/train",
        f"{Directory}/labels/val"
    ]
    
    for pasta in pastas:
        os.makedirs(pasta, exist_ok=True)


def criar_pastas_yolo_2_classes(Directory = ""):
    """
    Cria a estrutura de diretórios para armazenar as imagens e máscaras do YOLO.
    """
    pastas = [
        f"{Directory}",
        f"{Directory}/images",
        f"{Directory}/images/train",
        f"{Directory}/images/val",
        f"{Directory}/masks_breast",
        f"{Directory}/masks_breast/train",
        f"{Directory}/masks_breast/val",
        f"{Directory}/masks_marker",
        f"{Directory}/masks_marker/train",
        f"{Directory}/masks_marker/val",
        f"{Directory}/labels/train",
        f"{Directory}/labels/val"
    ]
    
    for pasta in pastas:
        os.makedirs(pasta, exist_ok=True)


def mover_arquivos_yolo(imgs_train, imgs_valid, masks_train, masks_valid, Directory = ""):
    """
    Move imagens e máscaras para os diretórios adequados dentro de Yolo_dataset.
    """
    criar_pastas_yolo(Directory)

    for img in imgs_train:
        shutil.copy(img, f"{Directory}/images/train")
    for img in imgs_valid:
        shutil.copy(img, f"{Directory}/images/val")
    for mask in masks_train:
        shutil.copy(mask, f"{Directory}/masks/train")
    for mask in masks_valid:
        shutil.copy(mask, f"{Directory}/masks/val")

def mover_arquivos_yolo_2_classes(imgs_train, imgs_val,
                        mb_train,  mb_val,     # breast masks
                        mm_train,  mm_val,     # marker masks
                        directory=""):
    """
    Move imagens e máscaras para os diretórios adequados dentro de Yolo_dataset.
    """
    criar_pastas_yolo_2_classes(directory)

    for img in imgs_train:
        shutil.copy(img, f"{directory}/images/train")
    for img in imgs_val:
        shutil.copy(img, f"{directory}/images/val")

    for mask in mb_train:
        shutil.copy(mask, f"{directory}/masks_breast/train")
    for mask in mb_val:
        shutil.copy(mask, f"{directory}/masks_breast/val")

    for mask in mm_train:
        shutil.copy(mask, f"{directory}/masks_marker/train")
    for mask in mm_val:
        shutil.copy(mask, f"{directory}/masks_marker/val")


def yolo_data(angulo, img_path, mask_path, outputDir="", augment=False):
    """
    Prepara os dados de imagens e máscaras e move para a estrutura YOLO.
    """
    data_imgs, data_masks = filtrar_imgs_masks(angulo, img_path, mask_path)

    imgs_train, imgs_valid, masks_train, masks_valid = train_test_split(
        data_imgs, data_masks, test_size=0.2, random_state=42
    )

    mover_arquivos_yolo(imgs_train, imgs_valid, masks_train, masks_valid, outputDir)

    if augment:

        augment_and_save(
            imgs_train, masks_train, outputDir, num_augmented_copies=2
        )


    masks_to_polygons(
        input_dir= f"{outputDir}/masks/train",
        output_dir=f"{outputDir}/labels/train"
    )
    masks_to_polygons(
        input_dir=f"{outputDir}/masks/val",
        output_dir=f"{outputDir}/labels/val"
    )

def yolo_data_2_classes(angulo, img_path, mask_breast_path, mask_marker_path, outputDir="", augment=False):
    """
    Prepara os dados de imagens e máscaras e move para a estrutura YOLO.
    """
    data_imgs, data_breast_masks = filtrar_imgs_masks(angulo, img_path, mask_breast_path)
    _, data_marker_masks = filtrar_imgs_masks(angulo, img_path, mask_marker_path)


    idx_train, idx_val = train_test_split(
    range(len(data_imgs)), test_size=0.2, random_state=42, shuffle=True)

    imgs_train = [data_imgs[i]            for i in idx_train]
    imgs_val   = [data_imgs[i]            for i in idx_val]
    mb_train   = [data_breast_masks[i]    for i in idx_train]
    mb_val     = [data_breast_masks[i]    for i in idx_val]
    mm_train   = [data_marker_masks[i]    for i in idx_train]
    mm_val     = [data_marker_masks[i]    for i in idx_val]

    mover_arquivos_yolo_2_classes(imgs_train, imgs_val, mb_train, mb_val, mm_train, mm_val, outputDir)

    if augment:

        augment_and_save_2_classes(
            imgs_train, mb_train, mm_train, outputDir, num_augmented_copies=2 )


    for split in ("train", "val"):
        masks_pair_to_polygons(
            breast_dir = f"{outputDir}/masks_breast/{split}",
            marker_dir = f"{outputDir}/masks_marker/{split}",
            output_dir = f"{outputDir}/labels/{split}",
            cls_breast=0,
            cls_marker=1
        )
    



def masks_to_polygons(input_dir, output_dir):
    
    for j in os.listdir(input_dir):
        image_path = os.path.join(input_dir, j)
        # load the binary mask and get its contours
        mask = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)

        H, W = mask.shape
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # convert the contours to polygons
        polygons = []
        for cnt in contours:
            if cv2.contourArea(cnt) > 200:
                polygon = []
                for point in cnt:
                    x, y = point[0]
                    polygon.append(x / W)
                    polygon.append(y / H)
                polygons.append(polygon)

        
        with open('{}.txt'.format(os.path.join(output_dir, j)[:-4]), 'w') as f:
            for polygon in polygons:
                for p_, p in enumerate(polygon):
                    if p_ == len(polygon) - 1:
                        f.write('{}\n'.format(p))
                    elif p_ == 0:
                        f.write('0 {} '.format(p))
                    else:
                        f.write('{} '.format(p))

            f.close()

def masks_pair_to_polygons(breast_dir, marker_dir, output_dir,
                           cls_breast=0, cls_marker=1,
                           min_area=200, approx_eps=0.002):
    """
    Para cada arquivo presente em `breast_dir` e `marker_dir`
    (mesmo nome base), gera *um* .txt em `output_dir` contendo:

        cls_breast  x1 y1 x2 y2 ...   (polígono do seio)
        cls_marker  x1 y1 x2 y2 ...   (polígono do marcador)

    Coordenadas normalizadas 0-1 conforme YOLOv8-seg.
    """
    os.makedirs(output_dir, exist_ok=True)

    file_names = [n for n in os.listdir(breast_dir)
                  if n.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff"))]

    for f in file_names:
        # caminhos das duas máscaras
        mask_breast = os.path.join(breast_dir,  f)
        mask_marker = os.path.join(marker_dir, f)

        if not os.path.exists(mask_marker):
            continue   

        # Carrega uma das máscaras só para obter H,W 
        m0 = cv2.imread(mask_breast, cv2.IMREAD_GRAYSCALE)
        H, W = m0.shape

        def _contours(mask_path):
            m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            _, m = cv2.threshold(m, 1, 255, cv2.THRESH_BINARY)
            cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
            polys = []
            for c in cnts:
                if cv2.contourArea(c) < min_area:
                    continue
                peri = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, approx_eps * peri, True)
                pts = approx.reshape(-1, 2).astype(np.float32)
                # normaliza
                pts[:, 0] /= W
                pts[:, 1] /= H
                polys.append(pts)
            return polys

        polys_breast  = _contours(mask_breast)
        polys_marker  = _contours(mask_marker)

        
        # escreve .txt
        label_path = '{}.txt'.format(os.path.join(output_dir, f)[:-4])
        with open(label_path, 'w') as txt:
            for poly in polys_breast:
                coords = " ".join([f"{x:.6f} {y:.6f}" for x, y in poly])
                txt.write(f"{cls_breast} {coords}\n")
            for poly in polys_marker:
                coords = " ".join([f"{x:.6f} {y:.6f}" for x, y in poly])
                txt.write(f"{cls_marker} {coords}\n")


#Listar os nomes das imagens que não quero usar para treinar o modelo de classificação
def listar_imgs_nao_usadas(directory, angulo):
    angulos = {
        "Frontal": "1",
        "Left45": "4",
        "Right45": "2",
        "Left90": "5",
        "Right90": "3"
    }

    idx_angle = angulos.get(angulo)
    if not idx_angle:
        raise ValueError(f"Ângulo inválido: {angulo}")
    
    pattern = re.compile(f".*{idx_angle}\.S.*")
    imgs_files = sorted([img for img in os.listdir(directory) if pattern.match(img)])
    nomesexcluidos = set()
    for file in imgs_files:
        # Extrair número do paciente e data do nome da segmentação (Ex: T0337.2.1.S.2019-11-13.00)
        match = re.search(r"T(\d+).*?(\d{4}-\d{2}-\d{2})", file)
        if match:
            paciente, data = match.groups()
            if angulo == "Frontal":
                nome_adequado = f"{int(paciente)}_img_Static-{angulo}_{data}.txt"  
            else:
                nome_adequado = f"{int(paciente)}_img_Static-{angulo}°_{data}.txt"
            nomesexcluidos.add(nome_adequado)
    return nomesexcluidos


def apply_augmentation_and_expand_seg(images, masks, num_augmented_copies, resize=False, target_size=0):

    print("Aumentando o dataset com cópias aumentadas...")
    print("Shape original das imagens:", images.shape)
    print("Shape original das máscaras:", masks.shape)

    #VALUE_SEED = int(time.time() * 1000) % 15000
    VALUE_SEED = 12274
    random.seed(VALUE_SEED)
    print(f"***SEMENTE USADA****: {VALUE_SEED}")
    
    
    images = np.expand_dims(images, axis=-1)
    masks = np.expand_dims(masks, axis=-1)

    all_images = []
    all_masks = []

    
    for img, mask in zip(images, masks):
        all_images.append(img)
        all_masks.append(mask)

    
    spatial_transformations_img = [random_rotation, random_zoom, random_flip]
    spatial_transformations_mask = [random_rotation_mask, random_zoom_mask, random_flip]

    for _ in range(num_augmented_copies):
        for transformation_img, transformation_mask in zip (spatial_transformations_img, spatial_transformations_mask):
            for img, mask in zip(images, masks):    
                # Usa um seed para garantir que a transformação aplicada na imagem e na máscara seja a mesma
                seed = random.randint(0, 10000)
                
                random.seed(seed)
                aug_img = apply_transformation(img, transformation_img,seed)
                
                random.seed(seed)
                aug_mask = apply_transformation(mask, transformation_mask, seed)
                
                all_images.append(aug_img)
                all_masks.append(aug_mask)

    
    all_images = np.array(all_images)
    all_masks = np.array(all_masks)

    
    if resize:
        all_images = tf.image.resize_with_pad(all_images, target_size, target_size, method="bilinear")
        all_masks = tf.image.resize_with_pad(all_masks, target_size, target_size, method="bilinear")

    
    all_images = np.squeeze(all_images, axis=-1)
    all_masks = np.squeeze(all_masks, axis=-1)

    print("Shape das imagens aumentadas:", all_images.shape)
    print("Shape das máscaras aumentadas:", all_masks.shape)
    
    return all_images, all_masks



def augment_and_save(img_paths, mask_paths, outputDir, num_augmented_copies):
    """
    Gera augmentations para cada par imagem/máscara em img_paths/mask_paths,
    salva em outputDir/images/train e outputDir/masks/train preservando
    o nome original e prefixo 'aug_'.
    """

    VALUE_SEED = 12274
    random.seed(VALUE_SEED)
    print(f"***SEMENTE USADA****: {VALUE_SEED}")
    
    spatial_transformations_img = [random_rotation, random_zoom, random_flip]
    spatial_transformations_mask = [random_rotation_mask, random_zoom_mask, random_flip]

    for i, (img_path, mask_path) in enumerate(zip(img_paths, mask_paths)):
        base_name = os.path.basename(img_path)
        # Carrega e normaliza
        img = np.expand_dims(np.array(Image.open(img_path).convert('L')) / 255.0, axis=-1)
        mask = np.expand_dims(np.array(Image.open(mask_path).convert('L')) / 255.0, axis=-1)

        

        # Transformações espaciais (sincronizadas)
        for j in range(num_augmented_copies):
            i = 0
            for trans_img, trans_mask in zip(spatial_transformations_img, spatial_transformations_mask):
                i = i + 1
                seed = random.randint(0, 10000)
                random.seed(seed)
                aug_img = apply_transformation(img, trans_img)
                random.seed(seed)
                aug_mask = apply_transformation(mask, trans_mask)

                fname = f"aug_{j}.{i}_{base_name}"
                img_out = (aug_img.squeeze() * 255).astype(np.uint8)
                mask_out = (aug_mask.squeeze() * 255).astype(np.uint8)

                Image.fromarray(img_out).save(os.path.join(outputDir, 'images/train', fname))
                Image.fromarray(mask_out).save(os.path.join(outputDir, 'masks/train', fname))

def augment_and_save_2_classes(img_paths,
                     mb_paths,               # máscaras Breast
                     mm_paths,               # máscaras Marker
                     outputDir,
                     num_augmented_copies=2):

    VALUE_SEED = 12274
    random.seed(VALUE_SEED)
    print(f"***SEMENTE USADA****: {VALUE_SEED}")

    
    spatial_transformations_img  = [random_rotation, random_zoom, random_flip]
    spatial_transformations_mask = [random_rotation_mask,
                                    random_zoom_mask,
                                    random_flip]          

    for img_path, mb_path, mm_path in zip(img_paths, mb_paths, mm_paths):
        base_name = os.path.basename(img_path)

       
        img = np.expand_dims(np.array(Image.open(img_path).convert('L'))/255.0, axis=-1)
        mb  = np.expand_dims(np.array(Image.open(mb_path).convert('L')) /255.0, axis=-1)
        mm  = np.expand_dims(np.array(Image.open(mm_path).convert('L')) /255.0, axis=-1)

       
        for j in range(num_augmented_copies):
            for k, (t_img, t_mask) in enumerate(zip(spatial_transformations_img,
                                                    spatial_transformations_mask), start=1):
                seed = random.randint(0, 10_000)
                random.seed(seed)
                aug_img = apply_transformation(img, t_img)

                random.seed(seed)
                aug_mb  = apply_transformation(mb, t_mask)

                random.seed(seed)
                aug_mm  = apply_transformation(mm, t_mask)

                fname = f"aug_{j}.{k}_{base_name}"

                
                Image.fromarray((aug_img.squeeze()*255).astype(np.uint8))\
                     .save(os.path.join(outputDir, "images/train", fname))

                Image.fromarray((aug_mb.squeeze()*255).astype(np.uint8))\
                     .save(os.path.join(outputDir, "masks_breast/train", fname))

                Image.fromarray((aug_mm.squeeze()*255).astype(np.uint8))\
                     .save(os.path.join(outputDir, "masks_marker/train", fname))


def filter_dataset_by_id(src_dir: str, dst_dir: str, ids_to_remove):
    """
    Copia recursivamente todo o conteúdo de `src_dir` para `dst_dir`, exceto
    os arquivos cujo nome comece com qualquer um dos IDs em `ids_to_remove`.

    :param src_dir: caminho para o dataset original (ex: "raw_dataset")
    :param dst_dir: caminho para onde será salva a cópia filtrada
    :param ids_to_remove: lista (ou set) de IDs (string ou int) a excluir
    """
    ids_str = {str(i) for i in ids_to_remove}

    for root, dirs, files in os.walk(src_dir):
        rel_path = os.path.relpath(root, src_dir)
        target_root = os.path.join(dst_dir, rel_path)
        os.makedirs(target_root, exist_ok=True)

        for fname in files:
            file_id = fname.split('_', 1)[0]
            if file_id in ids_str:
                print(f"Excluindo {fname} do diretório {root}")
                continue

            src_path = os.path.join(root, fname)
            dst_path = os.path.join(target_root, fname)

            shutil.copy2(src_path, dst_path)


def load_temp_matrix(txt_path: Path) -> np.ndarray:
    """
    Lê o arquivo .txt e devolve uma matriz de temperaturas (float).
    Detecta automaticamente se o separador é ';' ou ' '.
    """

    with open(txt_path, 'r') as f:
        delim = ';' if ';' in f.readline() else ' '
        f.seek(0)
    temp = np.loadtxt(txt_path, delimiter=delim, dtype= np.float32)

    return temp


def _iter_labeled_files(angle_dir, exclude=False, exclude_set=None):
    for label_name, label_val in CLASS_LABELS:
        label_dir = os.path.join(angle_dir, label_name)
        for file in os.listdir(label_dir):
            if exclude and file in exclude_set:
                print(f"Excluindo {file} do diretório {label_dir}")
                continue
            yield label_name, label_val, file, os.path.join(label_dir, file)


def _extract_ufpe_patient_id(file, label_val):
    match = re.search(r'_T(\d+)_(\d+)', file) or re.search(r'_T(\d+) (\(\d+\))', file)
    if not match:
        raise ValueError(f"Não foi possível extrair o ID do arquivo: {file}")

    file_id_aux = match.group(1)
    file_id = match.group(2).replace('(', '').replace(')', '')
    return int(str(label_val + 1) + file_id_aux + file_id)



"""
Carrega o dataaset bruto a partir do caminho fornecido. Exclui os ids do exclude_set e retorna
o conjunto de imagens(em numpy), labels, ids dos pacientes, nomes dos arquivos data do exame e uma string com id e data do exame.
"""
def load_raw_images(angle_dir, exclude=False, exclude_set=None):
    imgs, labels, ids, ids_data = [], [], [], []
    nomes = []

    for _label_name, label_val, file, fpath in _iter_labeled_files(angle_dir, exclude, exclude_set):
        try:
            arr = load_temp_matrix(fpath)
            _id, _data, _id_data = extract_id_data(file)

            imgs.append(arr)
            labels.append(label_val)
            ids.append(_id)
            ids_data.append(_id_data)
            nomes.append(file)
        except Exception as e:
            print(f"Erro ao processar {fpath}: {e}")
            continue

    return np.array(imgs), np.array(labels), np.array(ids, dtype= int), nomes, np.array(ids_data)


def load_raw_images_with_masks(img_angle_dir, mask_angle_dir, exclude=False, exclude_set=None):
    imgs, labels, ids, ids_data, masks = [], [], [], [], []
    nomes = []

    for label_name, label_val, file, fpath_img in _iter_labeled_files(img_angle_dir, exclude, exclude_set):
        fpath_mask = os.path.join(mask_angle_dir, label_name, file)

        try:
            arr_img = load_temp_matrix(fpath_img)
            arr_mask = load_temp_matrix(fpath_mask)
            _id, _data, _id_data = extract_id_data(file)

            imgs.append(arr_img)
            labels.append(label_val)
            ids.append(_id)
            ids_data.append(_id_data)
            nomes.append(file)
            masks.append(arr_mask)
        except Exception as e:
            print(f"Erro ao processar {fpath_img}: {e}")
            continue

    return np.array(imgs), np.array(labels), np.array(ids, dtype= int), nomes, np.array(ids_data), np.array(masks)

def load_raw_images_ufpe(angle_dir, exclude=False, exclude_set=None):
    imgs, labels, ids = [], [], []

    for _label_name, label_val, file, fpath in _iter_labeled_files(angle_dir, exclude, exclude_set):
        try:
            arr = load_temp_matrix(fpath)
            file_id = _extract_ufpe_patient_id(file, label_val)

            imgs.append(arr)
            labels.append(label_val)
            ids.append(file_id)
        except Exception as e:
            print(f"Erro ao processar {fpath}: {e}")
            continue

    return np.array(imgs), np.array(labels), np.array(ids, dtype=int)

SINGLE_RUN_REFERENCE_FOLDS = 5


def _resolve_outer_n_splits(k):
    if k < 1:
        raise ValueError("k deve ser >= 1")
    return SINGLE_RUN_REFERENCE_FOLDS if k == 1 else k


def _make_train_val_indices(imgs, labels, ids, outer_train_val, val_size, seed):
    gss = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=seed)
    train, val = next(gss.split(imgs[outer_train_val],
                                labels[outer_train_val],
                                groups=ids[outer_train_val]))

    train_idx = outer_train_val[train]
    val_idx   = outer_train_val[val]
    return train_idx, val_idx


def make_tvt_splits(imgs, labels, ids, k, val_size, seed):
    outer_k = _resolve_outer_n_splits(k)
    sgkf = StratifiedGroupKFold(n_splits=outer_k, shuffle=True, random_state=seed)
    outer_splits = sgkf.split(imgs, labels, groups=ids)

    if k == 1:
        outer_train_val, test = next(outer_splits)
        train_idx, val_idx = _make_train_val_indices(
            imgs, labels, ids, outer_train_val, val_size, seed
        )
        yield train_idx, val_idx, test
        return

    for outer_train_val, test in outer_splits:
        train_idx, val_idx = _make_train_val_indices(
            imgs, labels, ids, outer_train_val, val_size, seed
        )
        yield train_idx, val_idx, test



def augment_train_fold(x_train, y_train, n_aug=1, seed=42, dataset="uff"):
    """
    Recebe dados de treino de UM fold e concatena n_aug versões aumentadas.
    """
    
    if dataset == "ufpe":
        aug_imgs, aug_labels = apply_augmentation_and_expand_ufpe(    
                                x_train, y_train, n_aug,seed, resize=False)
    else:
        aug_imgs, aug_labels = apply_augmentation_and_expand(    
                                x_train, y_train, n_aug,seed, resize=False)

    return aug_imgs, aug_labels

def augment_train_fold_with_masks(x_train_imgs, X_train_masks, y_train, n_aug=1, seed=42):
    """
    Recebe dados de treino de UM fold e concatena n_aug versões aumentadas.
    """
    aug_imgs, aug_masks, aug_labels = apply_augmentation_and_expand_with_masks(    
                            x_train_imgs, X_train_masks, y_train, n_aug,seed, resize=False)

    return aug_imgs, aug_masks, aug_labels

def normalize(arr, min_v, max_v):
    return (arr - min_v) / (max_v - min_v + 1e-8)


def tf_letterbox(images, target = 224, mode = 'bilinear'):


    TARGET = target          
    PAD_COLOR = 114/255.0  

    #PAD_COLOR = 0.0

    h, w = tf.shape(images)[1], tf.shape(images)[2]
    
    r = tf.cast(tf.minimum(TARGET / tf.cast(h, tf.float32),
                           TARGET / tf.cast(w, tf.float32)), tf.float32)
    
    new_h = tf.cast(tf.round(tf.cast(h, tf.float32) * r), tf.int32)
    new_w = tf.cast(tf.round(tf.cast(w, tf.float32) * r), tf.int32)

    # Resize todas as imagens do batch
    resized = tf.image.resize(images, (new_h, new_w), method=mode)

    # Calcula padding necessário
    pad_h = TARGET - new_h
    pad_w = TARGET - new_w
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left

    paddings = [[0, 0], [top, bottom], [left, right], [0, 0]]
    
    padded = tf.pad(resized, 
				    paddings, 
				    mode='CONSTANT', 
				    constant_values=PAD_COLOR)

    return padded


def tf_letterbox_black(images, target = 224, mode = 'bilinear'):


    TARGET = target          
    #PAD_COLOR = 114/255.0  

    PAD_COLOR = 0.0

    h, w = tf.shape(images)[1], tf.shape(images)[2]
    
    r = tf.cast(tf.minimum(TARGET / tf.cast(h, tf.float32),
                           TARGET / tf.cast(w, tf.float32)), tf.float32)
    
    new_h = tf.cast(tf.round(tf.cast(h, tf.float32) * r), tf.int32)
    new_w = tf.cast(tf.round(tf.cast(w, tf.float32) * r), tf.int32)

    # Resize todas as imagens do batch
    resized = tf.image.resize(images, (new_h, new_w), method=mode)

    # Calcula padding necessário
    pad_h = TARGET - new_h
    pad_w = TARGET - new_w
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left

    paddings = [[0, 0], [top, bottom], [left, right], [0, 0]]
    
    padded = tf.pad(resized, 
				    paddings, 
				    mode='CONSTANT', 
				    constant_values=PAD_COLOR)

    return padded



# Redimensiona as imagens sem adicionar padding
def tf_letterbox_Sem_padding(images, target = 224, mode = 'bilinear'):


    # TARGET = target          
    # PAD_COLOR = 114/255.0  

    # #PAD_COLOR = 0.0

    # h, w = tf.shape(images)[1], tf.shape(images)[2]
    
    # r = tf.cast(tf.minimum(TARGET / tf.cast(h, tf.float32),
    #                        TARGET / tf.cast(w, tf.float32)), tf.float32)
    
    # new_h = tf.cast(tf.round(tf.cast(h, tf.float32) * r), tf.int32)
    # new_w = tf.cast(tf.round(tf.cast(w, tf.float32) * r), tf.int32)

    # Resize todas as imagens do batch
    resized = tf.image.resize(images, (160, 224), method=mode)

    
    return resized

def letterbox_center_crop(images, target=224, mode='bilinear'):
    """
    images: tensor [B, H, W, C]
    Retorna: tensor [B, 224, 224, C] - sem padding.
    """
    h = tf.cast(tf.shape(images)[1], tf.float32)  # altura original
    w = tf.cast(tf.shape(images)[2], tf.float32)  # largura original

    # 1) fator de escala - faz o lado MENOR virar 'target'
    r = tf.maximum(target / h, target / w)        # garante min(h, w) - target

    new_h = tf.cast(tf.round(h * r), tf.int32)
    new_w = tf.cast(tf.round(w * r), tf.int32)

    # 2) redimensiona proporcionalmente
    resized = tf.image.resize(images, (new_h, new_w), method=mode)

    # 3) calcula deslocamentos p/ crop central
    offset_h = (new_h - target) // 2
    offset_w = (new_w - target) // 2

    # 4) recorta [224 × 224]
    cropped = tf.image.crop_to_bounding_box(
        resized, offset_h, offset_w, target, target
    )
    return cropped


def load_imgs_masks_sem_padding(angulo, img_path, mask_path, augment=False, resize=False, target = 224):
    """
    Carrega imagens e máscaras em formato numpy arrays normalizados (0 a 1).
    Retorna as listas imgs_train, imgs_valid, masks_train, masks_valid.
    """
    data_imgs, data_masks = filtrar_imgs_masks(angulo, img_path, mask_path)

    imagens = [np.array(Image.open(img).convert('L')) / 255.0 for img in data_imgs]
    mascaras = [np.array(Image.open(mask).convert('L')) / 255.0 for mask in data_masks]

    if resize:

        imagens = np.expand_dims(imagens, axis=-1)
        mascaras = np.expand_dims(mascaras, axis=-1)

        # imagens = tf.image.resize_with_pad(imagens, target, target, method="bilinear")
        # mascaras = tf.image.resize_with_pad(mascaras, target, target, method="nearest")

        imagens = tf_letterbox_Sem_padding(imagens, target, mode="bilinear")
        mascaras = tf_letterbox_Sem_padding(mascaras, target, mode="nearest")

        imagens = np.squeeze(imagens, axis=-1)
        mascaras = np.squeeze(mascaras, axis=-1)

    

    imgs_train, imgs_valid, masks_train, masks_valid = train_test_split(
        imagens, mascaras, test_size=0.2, random_state=42
    )

    imgs_train = np.array(imgs_train)
    masks_train = np.array(masks_train)
    imgs_valid = np.array(imgs_valid)
    masks_valid = np.array(masks_valid)

    print("Train shape:", imgs_train.shape)
    print("Valid shape:", imgs_valid.shape)

    if augment:
        imgs_train_aug, masks_train_aug = apply_augmentation_and_expand_seg(
            imgs_train, masks_train, num_augmented_copies = 2, resize=False
        )

        return imgs_train_aug, imgs_valid, masks_train_aug, masks_valid
    
    return imgs_train, imgs_valid, masks_train, masks_valid

def load_imgs_masks_recortado(angulo, img_path, mask_path, augment=False, resize=False, target = 224):
    """
    Carrega imagens e máscaras em formato numpy arrays normalizados (0 a 1).
    Retorna as listas imgs_train, imgs_valid, masks_train, masks_valid.
    """
    data_imgs, data_masks = filtrar_imgs_masks(angulo, img_path, mask_path)

    imagens = [np.array(Image.open(img).convert('L')) / 255.0 for img in data_imgs]
    mascaras = [np.array(Image.open(mask).convert('L')) / 255.0 for mask in data_masks]

    if resize:

        imagens = np.expand_dims(imagens, axis=-1)
        mascaras = np.expand_dims(mascaras, axis=-1)

        # imagens = tf.image.resize_with_pad(imagens, target, target, method="bilinear")
        # mascaras = tf.image.resize_with_pad(mascaras, target, target, method="nearest")

        imagens = letterbox_center_crop(imagens, target, mode="bilinear")
        mascaras = letterbox_center_crop(mascaras, target, mode="nearest")

        imagens = np.squeeze(imagens, axis=-1)
        mascaras = np.squeeze(mascaras, axis=-1)

    

    imgs_train, imgs_valid, masks_train, masks_valid = train_test_split(
        imagens, mascaras, test_size=0.2, random_state=42
    )

    imgs_train = np.array(imgs_train)
    masks_train = np.array(masks_train)
    imgs_valid = np.array(imgs_valid)
    masks_valid = np.array(masks_valid)

    print("Train shape:", imgs_train.shape)
    print("Valid shape:", imgs_valid.shape)

    if augment:
        imgs_train_aug, masks_train_aug = apply_augmentation_and_expand_seg(
            imgs_train, masks_train, num_augmented_copies = 2, resize=False
        )

        return imgs_train_aug, imgs_valid, masks_train_aug, masks_valid
    
    return imgs_train, imgs_valid, masks_train, masks_valid



def load_imgs_masks_Black_Padding(angulo, img_path, mask_path, augment=False, resize=False, target = 224):
    """
    Carrega imagens e máscaras em formato numpy arrays normalizados (0 a 1).
    Retorna as listas imgs_train, imgs_valid, masks_train, masks_valid.
    """
    data_imgs, data_masks = filtrar_imgs_masks(angulo, img_path, mask_path)

    # imagens = [np.array(Image.open(img).convert('L')) / 255.0 for img in data_imgs]
    #mascaras = [np.array(Image.open(mask).convert('L')) / 255.0 for mask in data_masks]

    imagens = []
    mascaras = []

    for img, mask in zip(data_imgs, data_masks):
        try:
            arr = load_temp_matrix(img)

            arrMask = np.array(Image.open(mask).convert('L')) / 255.0
            
            imagens.append(arr)
            mascaras.append(arrMask)
    
        except Exception as e:
            print(f"Erro ao processar {img}: {e}")
            continue

    imagens, mascaras = np.array(imagens), np.array(mascaras)

    mn, mx = imagens.min(), imagens.max()

    imagens = normalize(imagens, mn, mx)

    if resize:

        imagens = np.expand_dims(imagens, axis=-1)
        mascaras = np.expand_dims(mascaras, axis=-1)

        # imagens = tf.image.resize_with_pad(imagens, target, target, method="bilinear")
        # mascaras = tf.image.resize_with_pad(mascaras, target, target, method="nearest")

        imagens = tf_letterbox_black(imagens, target, mode="bilinear")
        mascaras = tf_letterbox_black(mascaras, target, mode="nearest")

        imagens = tf.clip_by_value(imagens, 0, 1).numpy().squeeze(axis=-1)
        mascaras = tf.clip_by_value(mascaras, 0, 1).numpy().squeeze(axis=-1)

    

    imgs_train, imgs_valid, masks_train, masks_valid = train_test_split(
        imagens, mascaras, test_size=0.2, random_state=42
    )

    imgs_train = np.array(imgs_train)
    masks_train = np.array(masks_train)
    imgs_valid = np.array(imgs_valid)
    masks_valid = np.array(masks_valid)

    print("Train shape:", imgs_train.shape)
    print("Valid shape:", imgs_valid.shape)

    if augment:
        imgs_train_aug, masks_train_aug = apply_augmentation_and_expand_seg(
            imgs_train, masks_train, num_augmented_copies = 2, resize=False
        )

        return imgs_train_aug, imgs_valid, masks_train_aug, masks_valid
    
    return imgs_train, imgs_valid, masks_train, masks_valid

def load_imgs_masks_distorcidas(angulo, img_path, mask_path, augment=False, resize=False, target = 224):
    """
    Carrega imagens e máscaras em formato numpy arrays normalizados (0 a 1).
    Retorna as listas imgs_train, imgs_valid, masks_train, masks_valid.
    """
    data_imgs, data_masks = filtrar_imgs_masks(angulo, img_path, mask_path)

    imagens = [np.array(Image.open(img).convert('L')) / 255.0 for img in data_imgs]
    mascaras = [np.array(Image.open(mask).convert('L')) / 255.0 for mask in data_masks]

    if resize:

        imagens = np.expand_dims(imagens, axis=-1)
        mascaras = np.expand_dims(mascaras, axis=-1)

        # imagens = tf.image.resize_with_pad(imagens, target, target, method="bilinear")
        # mascaras = tf.image.resize_with_pad(mascaras, target, target, method="nearest")

        imagens = tf.image.resize(imagens, (224, 224), method="bilinear")
        mascaras = tf.image.resize(mascaras, (224, 224), method="nearest")

        imagens = np.squeeze(imagens, axis=-1)
        mascaras = np.squeeze(mascaras, axis=-1)

    

    imgs_train, imgs_valid, masks_train, masks_valid = train_test_split(
        imagens, mascaras, test_size=0.2, random_state=42
    )

    imgs_train = np.array(imgs_train)
    masks_train = np.array(masks_train)
    imgs_valid = np.array(imgs_valid)
    masks_valid = np.array(masks_valid)

    print("Train shape:", imgs_train.shape)
    print("Valid shape:", imgs_valid.shape)

    if augment:
        imgs_train_aug, masks_train_aug = apply_augmentation_and_expand_seg(
            imgs_train, masks_train, num_augmented_copies = 2, resize=False
        )

        return imgs_train_aug, imgs_valid, masks_train_aug, masks_valid
    
    return imgs_train, imgs_valid, masks_train, masks_valid



