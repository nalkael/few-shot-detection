import os

#print("train base dataset...")

print("train 1-shot...")
# os.system("python -m tools.train_net --num-gpus 1 --config-file configs/COCO-detection/faster_rcnn_R_101_FPN_ft_all_ortho_1shot.yaml")
# os.system("python -m tools.train_net --num-gpus 1 --config-file configs/COCO-detection/faster_rcnn_R_101_FPN_ft_all_ortho_2shot.yaml")
# os.system("python -m tools.train_net --num-gpus 1 --config-file configs/COCO-detection/faster_rcnn_R_101_FPN_ft_all_ortho_3shot.yaml")
# os.system("python -m tools.train_net --num-gpus 1 --config-file configs/COCO-detection/faster_rcnn_R_101_FPN_ft_all_ortho_10shot.yaml")
# os.system("python -m tools.train_net --num-gpus 1 --config-file configs/COCO-detection/faster_rcnn_R_101_FPN_ft_all_ortho_30shot.yaml")
os.system("python -m tools.train_net --num-gpus 1 --config-file configs/COCO-detection/faster_rcnn_R_101_FPN_ft_all_ortho_50shot.yaml")