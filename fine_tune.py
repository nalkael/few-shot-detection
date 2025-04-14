import os

# 10-shot
os.system(
   "python3 -m tools.train_net --num-gpus 1 \
    --config-file configs/COCO-detection/faster_rcnn_R_101_FPN_ft_all_ortho_10shot.yaml \
    --opts MODEL.WEIGHTS checkpoints/coco/faster_rcnn/faster_rcnn_R_101_FPN_ortho_all/model_reset_surgery.pth  \
    " 
)

# 20-shot
os.system(
   "python3 -m tools.train_net --num-gpus 1 \
    --config-file configs/COCO-detection/faster_rcnn_R_101_FPN_ft_all_ortho_20shot.yaml \
    --opts MODEL.WEIGHTS checkpoints/coco/faster_rcnn/faster_rcnn_R_101_FPN_ortho_all/model_reset_surgery.pth  \
    " 
)

# 30-shot
os.system(
   "python3 -m tools.train_net --num-gpus 1 \
    --config-file configs/COCO-detection/faster_rcnn_R_101_FPN_ft_all_ortho_30shot.yaml \
    --opts MODEL.WEIGHTS checkpoints/coco/faster_rcnn/faster_rcnn_R_101_FPN_ortho_all/model_reset_surgery.pth  \
    " 
)