import os

os.system(
    "python3 -m tools.ckpt_surgery \
        --src1 checkpoints/coco/faster_rcnn/faster_rcnn_R_101_FPN_ortho_base/model_final.pth \
        --src2 checkpoints/coco/faster_rcnn/faster_rcnn_R_101_FPN_ft_novel_ortho_30shot/model_final.pth \
        --method combine \
        --save-dir checkpoints/coco/faster_rcnn/faster_rcnn_R_101_FPN_ortho_all_30shot \
        --coco")

