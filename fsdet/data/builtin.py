"""
This file registers pre-defined datasets at hard-coded paths, and their metadata.
We hard-code metadata for common datasets. This will enable:
1. Consistency check when loading the datasets
2. Use models on these standard datasets directly and run demos,
   without having to download the dataset annotations
We hard-code some paths to the dataset that's assumed to
exist in "./datasets/".

Here we only register the few-shot datasets and complete COCO, PascalVOC and 
LVIS have been handled by the builtin datasets in detectron2. 
"""

import os
import sys
import random
import cv2
from detectron2.data import DatasetCatalog, MetadataCatalog

from detectron2.data.datasets.register_coco import register_coco_instances

sys.path.append(os.path.dirname(os.path.abspath(__file__)))  # Add the script's directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # Add the parent directory

#print("path: ", os.path.dirname(os.path.abspath(__file__)))

from fsdet.data.builtin_meta import _get_builtin_metadata
from fsdet.data.meta_coco import register_meta_coco

from detectron2.utils.visualizer import Visualizer

# ==== Predefined datasets and splits for COCO ==========
root_pth = "/home/rdluhu/Dokumente/few-shot-object-detection/datasets"
_PREDEFINED_SPLITS_COCO = {}
_PREDEFINED_SPLITS_COCO["coco"] = {
    # orthomosaic
    "ortho_train": (
        "dataset_coco/640x640_coco/train",
        "dataset_coco/640x640_coco/train/_annotations.coco.json",
    ),
}


def register_all_coco(root=root_pth):
    # for dataset_name, splits_per_dataset in _PREDEFINED_SPLITS_COCO.items():
    #     for key, (image_root, json_file) in splits_per_dataset.items():
    #         # Assume pre-defined datasets live in `./datasets`.
    #         register_coco_instances(
    #             key,
    #             _get_builtin_metadata(dataset_name),
    #             os.path.join(root, json_file)
    #             if "://" not in json_file
    #             else json_file,
    #             os.path.join(root, image_root),
    #         )

    # register meta datasets
    METASPLITS = [
        (
            "ortho_train_all",
            "dataset_coco/640x640_coco/train",
            "dataset_coco/640x640_coco/train/_annotations.coco.json",
        ),
        (
            "ortho_train_base",
            "dataset_coco/640x640_coco/train",
            "dataset_coco/640x640_coco/train/_annotations.coco.json",
        ),
        ("ortho_test_all", "dataset_coco/640x640_coco/valid", "dataset_coco/640x640_coco/valid/_annotations.coco.json"),
        ("ortho_test_base", "dataset_coco/640x640_coco/valid", "dataset_coco/640x640_coco/valid/_annotations.coco.json"),
        ("ortho_test_novel", "dataset_coco/640x640_coco/valid", "dataset_coco/640x640_coco/valid/_annotations.coco.json"),
    ]

    # register small meta datasets for fine-tuning stage
    for prefix in ["all", "novel"]:
        # for shot in [1, 2, 3, 5, 10, 30]:
        for shot in [1, 2, 3, 5, 10, 30, 50]: # change shot
            for seed in range(10):
                seed = "" if seed == 0 else "_seed{}".format(seed)
                name = "coco_trainval_{}_{}shot{}".format(prefix, shot, seed)
                METASPLITS.append((name, "dataset_coco/640x640_coco/train", ""))

    for name, imgdir, annofile in METASPLITS:
        register_meta_coco(
            name,
            # return _get_coco_fewshot_instances_meta()
            _get_builtin_metadata("coco_fewshot"), 
            os.path.join(root, imgdir),
            os.path.join(root, annofile),
        )

# Register them all under "./datasets"
register_all_coco()

DATASET_NAME = "ortho_test_all"

# Load dataset & metadata
dataset_dicts = DatasetCatalog.get(DATASET_NAME)
metadata = MetadataCatalog.get(DATASET_NAME)

for i in range(20):
    # Select a random sample from the dataset
    sample = random.choice(dataset_dicts)
    img_path = sample["file_name"]
    img = cv2.imread(img_path)  # Read image
    if img is None:
        print(f"Error: Unable to read image {img_path}")
        exit()

    # Create a visualizer and draw bounding boxes
    visualizer = Visualizer(img[:, :, ::-1], metadata=metadata, scale=1.0)  # Convert BGR to RGB
    vis_img = visualizer.draw_dataset_dict(sample).get_image()

    # Convert back to OpenCV format (RGB to BGR)
    vis_img = cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR)

    # Display the image using OpenCV
    cv2.imshow("Dataset Visualization", vis_img)

    # Wait for a key press and close window
    cv2.waitKey(0)
    cv2.destroyAllWindows()

