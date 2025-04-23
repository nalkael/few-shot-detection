import argparse
import glob
import multiprocessing as mp
import os
import time
import sys

import cv2
import tqdm
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger

from demo.predictor import VisualizationDemo
from detectron2.config import get_cfg

from detectron2.engine import DefaultPredictor
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer

from detectron2.data.datasets import register_coco_instances

import supervision as sv
from supervision.metrics import F1Score, Precision, Recall, MeanAveragePrecision, MeanAverageRecall


register_coco_instances(
    "my_dataset_train",
    {}, 
    "datasets/merged_ortho_coco/train/_annotations.coco.json",
    "datasets/merged_ortho_coco/train"
    )


# constants
WINDOW_NAME = "COCO detections"

custom_class_names = [
    "gas valve",
    "manhole",
    "storm drain",
    "under hydrant",
    "utility shaft",
    "water valve",

    "aircraft",
    "oiltank",
    "overpass",
    "playground",
]

metadata = MetadataCatalog.get("my_dataset_train")

custom_metadata = {"thing_classes": custom_class_names}

def setup_cfg():
    # load config from file and command-line arguments
    cfg = get_cfg()
    cfg.merge_from_file("mainprocess/faster_rcnn_R_101_FPN_ft_all_ortho_10shot.yaml") # config file
    cfg.MODEL.WEIGHTS = "checkpoints/coco/faster_rcnn/faster_rcnn_R_101_FPN_ft_all_10shot/model_final.pth" # weights
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.55
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.55
    cfg.freeze()
    return cfg


def inference_image(cfg, image_path="default_image.jpg"):
    predictor = DefaultPredictor(cfg)
    image = cv2.imread(image_path)
    outputs = predictor(image)
    print(f"Inference on image: {image_path}")

    visualizer = Visualizer(image[:, :, ::-1], metadata=custom_metadata, scale=2.0)
    output_image = visualizer.draw_instance_predictions(outputs["instances"].to("cpu")).get_image()  # Draw predictions
    
    # Display the results
    cv2.imshow("Inference Result", output_image)  # Display the image with predictions
    cv2.waitKey(0)  # Wait for a key press to close the image window
    cv2.destroyAllWindows()  # Close the image window
    return outputs



if __name__ == '__main__':
    cfg = setup_cfg()
    predictor = DefaultPredictor(cfg)
    image_path = sys.argv[1] if len(sys.argv) > 1 else "datasets/merged_ortho_coco/valid/20230808_FR_18_3_png.rf.78f257d7f2291739f2d05334cab05cb6.jpg"
    inference_image(cfg, image_path)