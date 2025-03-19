import os

#print("train base dataset...")

# print("train 1-shot...")
# os.system("python -m tools.train_net --num-gpus 1 --config-file configs/COCO-detection/faster_rcnn_R_101_FPN_ft_all_ortho_1shot.yaml")
# os.system("python -m tools.train_net --num-gpus 1 --config-file configs/COCO-detection/faster_rcnn_R_101_FPN_ft_all_ortho_2shot.yaml")
# os.system("python -m tools.train_net --num-gpus 1 --config-file configs/COCO-detection/faster_rcnn_R_101_FPN_ft_all_ortho_3shot.yaml")
# os.system("python -m tools.train_net --num-gpus 1 --config-file configs/COCO-detection/faster_rcnn_R_101_FPN_ft_all_ortho_10shot.yaml")
# os.system("python -m tools.train_net --num-gpus 1 --config-file configs/COCO-detection/faster_rcnn_R_101_FPN_ft_all_ortho_30shot.yaml")
# os.system("python -m tools.train_net --num-gpus 1 --config-file configs/COCO-detection/faster_rcnn_R_101_FPN_ft_all_ortho_50shot.yaml")

print("test demo...")

"""
os.system(
   "python -m demo.demo --config-file configs/COCO-detection/faster_rcnn_R_101_FPN_ft_all_30shot.yaml --input demo/1.jpg demo/2.jpg demo/3.jpg demo/4.jpg" 
)
"""

# import os

# Define the folder containing images
image_folder = "/home/rdluhu/Dokumente/few-shot-object-detection/datasets/coco/640x640_coco/valid"

# Get all .jpg files in the folder
image_files = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith(".jpg")]

# Convert list to a space-separated string
input_images = " ".join(image_files)

# Run the command
os.system(
    f"python -m demo.demo --config-file configs/COCO-detection/faster_rcnn_R_101_FPN_ft_all_30shot.yaml --input {input_images}"
)