import os

os.system(
    "python inference.py --config configs/COCO-detection/faster_rcnn_R_50_FPN_ft_all_jiaonang_30shot.yaml \
     --weights checkpoints/coco/faster_rcnn/faster_rcnn_R_50_FPN_ft_all_30shot/model_final.pth \
        --input datasets/coco/640x640_coco/valid --output output_images/ --batch"
    )