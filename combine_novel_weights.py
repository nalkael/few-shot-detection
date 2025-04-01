import os

# to use novel weights, fine-tune a predictor on the novel set
# we reuse the base model trained in the previous stage but retrain the last layer from scratch.
# first remove the last layer from the weights file 
os.system(
    "python3 -m tools.ckpt_surgery \
        --src1 checkpoints/coco/faster_rcnn/faster_rcnn_R_101_FPN_ortho_base/model_final.pth \
        --method remove \
        --save-dir checkpoints/coco/faster_rcnn/faster_rcnn_R_101_FPN_ortho_all \
        --coco")


# next, fine-tune the predictor on the novel set

os.system(
   "python3 -m tools.train_net --num-gpus 1 \
    --config-file configs/COCO-detection/faster_rcnn_R_101_FPN_ft_novel_ortho_50shot.yaml \
    --opts MODEL.WEIGHTS checkpoints/coco/faster_rcnn/faster_rcnn_R_101_FPN_ortho_all/model_reset_remove.pth  \
    " 
)