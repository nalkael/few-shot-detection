import os

"""
print("train 10-shot...")
os.system(
    "python -m tools.train_net --num-gpus 1 \
        --config-file \
        configs/COCO-detection/faster_rcnn_R_101_FPN_ft_all_ortho_10shot.yaml"
    )

print("train 20-shot...")
os.system(
    "python -m tools.train_net --num-gpus 1 \
        --config-file \
        configs/COCO-detection/faster_rcnn_R_101_FPN_ft_all_ortho_20shot.yaml"
    )
"""

print("train 30-shot...")
os.system(
    "python -m tools.train_net --num-gpus 1 \
        --config-file \
        configs/COCO-detection/faster_rcnn_R_101_FPN_ft_all_ortho_30shot.yaml"
    )



# os.system("python -m tools.train_net --num-gpus 1 --config-file configs/COCO-detection/faster_rcnn_R_101_FPN_ft_all_ortho_2shot.yaml")
# os.system("python -m tools.train_net --num-gpus 1 --config-file configs/COCO-detection/faster_rcnn_R_101_FPN_ft_all_ortho_3shot.yaml")
# os.system("python -m tools.train_net --num-gpus 1 --config-file configs/COCO-detection/faster_rcnn_R_101_FPN_ft_all_ortho_10shot.yaml")
# os.system("python -m tools.train_net --num-gpus 1 --config-file configs/COCO-detection/faster_rcnn_R_101_FPN_ft_all_ortho_30shot.yaml")
# os.system("python -m tools.train_net --num-gpus 1 --config-file configs/COCO-detection/faster_rcnn_R_101_FPN_ft_all_ortho_50shot.yaml")