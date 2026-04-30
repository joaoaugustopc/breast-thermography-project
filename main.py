import argparse
import random
import time

from src.training.classification import (
    train_model_cv,
)
from src.training.segmentation import train_unet_segmentation
from src.models.Vgg_16 import Vgg_16

# Use o tempo atual em segundos como semente
##VALUE_SEED = int(time.time() * 1000) % 15000
"""
VALUE_SEED = 7758
random.seed(VALUE_SEED)

seed = random.randint(0, 15000)

tf.random.set_seed(seed)

np.random.seed(seed)
"""


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_root", required=True)
    parser.add_argument("--angle", default="Frontal")
    parser.add_argument("--k", type=int, default=5, help="Numero de folds; use 1 para uma unica execucao.")
    parser.add_argument("--resize_to", type=int, default=224)
    parser.add_argument("--n_aug", type=int, default=2)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--message", required=True)
    parser.add_argument("--resize_method", default="BlackPadding")
    parser.add_argument("--segment", default=None)
    parser.add_argument("--segmenter_model", default="none")
    parser.add_argument("--seg_model_path", default="")
    parser.add_argument(
        "--yolo_marker_source",
        choices=["yolo", "manual_masks"],
        default="manual_masks",
        help="Fonte do marcador na segmentacao YOLO: classes da YOLO (yolo) ou mascaras manuais (manual_masks).",
    )

    args = parser.parse_args()

    if args.seed is None:
        VALUE_SEED = int(time.time()*1000) % 15000
        random.seed(VALUE_SEED)
        SEMENTE = random.randint(0,1500000)
    else:
        SEMENTE = args.seed

    if args.segment == None:
        train_model_cv(Vgg_16,
                    raw_root=args.raw_root,
                    angle=args.angle,
                    k=args.k,                 
                    resize_to=args.resize_to,
                    n_aug=args.n_aug,             
                    batch=args.batch,
                    seed= SEMENTE,
                    message=args.message,
                    resize_method=args.resize_method,
                    segmenter= args.segmenter_model,
                    seg_model_path=args.seg_model_path,
                    yolo_marker_source=args.yolo_marker_source)
        
    elif args.segment == "unet":
        train_unet_segmentation(
            message=args.message,
            angle=args.angle,
            resize_to=args.resize_to,
            batch_size=args.batch,
        )
    # elif args.segment == "yolo":
    #     resize_imgs_masks_dataset(
    #     img_dir="Termografia_Dataset_Segmentação_Frontal_jpg/images",
    #     mask_dir="Termografias_Dataset_Segmentação/masks",
    #     output_base="Termografias_Dataset_Segmentação_jpg_224",
    #     target=224,          
    #     resize_method="BlackPadding"
    # )

    #     yolo_data("Frontal", "Termografias_Dataset_Segmentação_jpg_224/images", "Termografias_Dataset_Segmentação_jpg_224/masks", "Yolo_dataset_8_12", True)

        ##Ultimo train37 Então: esse modelo vai ser salvo em train38
        # train_yolo_seg("n", 500, "dataset_yolo_8_12.yaml", seed=SEMENTE)
    
    # elif args.segment == "yolo2"

    
if __name__ == "__main__":
    main()
