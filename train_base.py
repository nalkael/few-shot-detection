import os

os.system(
    "python -m tools.train_net_origin --num-gpus 1 \
    --config-file configs/COCO-detection/faster_rcnn_R_101_FPN_ortho_base.yaml \
    "
)