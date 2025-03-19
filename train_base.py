import os

os.system(
    "python -m tools.train_net --num-gpus 1 \
    --config-file configs/COCO-detection/faster_rcnn_R_101_FPN_jiaonang_base.yaml \
    "
)