import fiftyone as fo
import fiftyone.zoo as foz

# Load COCO dataset
dataset = foz.load_zoo_dataset(
    "coco-2017",
    splits=("train", "validation", "test"),
    label_types=("detections"),
    classes=[],
    max_samples=50,
    shuffle=True,

)

dataset.export(
    export_dir='dataset/coco_sample',
    dataset_type=fo.types.COCODetectionDataset
)