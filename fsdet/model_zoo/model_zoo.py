import os

import pkg_resources
import torch
from detectron2.checkpoint import DetectionCheckpointer
from fsdet.config import get_cfg

from fsdet.modeling import build_model


class _ModelZooUrls(object):
    """
    Mapping from names to our pre-trained models.
    """

    URL_PREFIX = "http://dl.yf.io/fs-det/models/"

    # format: {config_path.yaml} -> model_id/model_final.pth
    CONFIG_PATH_TO_URL_SUFFIX = {
        ### PASCAL VOC Detection ###
        ### COCO Detection ###
        # Base Model
        "COCO-detection/faster_rcnn_R_101_FPN_base.yaml": "coco/base_model/model_final.pth",
        # FRCN+ft-full
        "COCO-detection/faster_rcnn_R_101_FPN_ft_all_1shot_unfreeze.yaml": "coco/FRCN+ft-full_1shot/model_final.pth",
        "COCO-detection/faster_rcnn_R_101_FPN_ft_all_2shot_unfreeze.yaml": "coco/FRCN+ft-full_2shot/model_final.pth",
        "COCO-detection/faster_rcnn_R_101_FPN_ft_all_3shot_unfreeze.yaml": "coco/FRCN+ft-full_3shot/model_final.pth",
        "COCO-detection/faster_rcnn_R_101_FPN_ft_all_5shot_unfreeze.yaml": "coco/FRCN+ft-full_5shot/model_final.pth",
        "COCO-detection/faster_rcnn_R_101_FPN_ft_all_10shot_unfreeze.yaml": "coco/FRCN+ft-full_10shot/model_final.pth",
        "COCO-detection/faster_rcnn_R_101_FPN_ft_all_30shot_unfreeze.yaml": "coco/FRCN+ft-full_30shot/model_final.pth",
        # TFA w/ cos
        "COCO-detection/faster_rcnn_R_101_FPN_ft_all_1shot.yaml": "coco/tfa_cos_1shot/model_final.pth",
        "COCO-detection/faster_rcnn_R_101_FPN_ft_all_2shot.yaml": "coco/tfa_cos_2shot/model_final.pth",
        "COCO-detection/faster_rcnn_R_101_FPN_ft_all_3shot.yaml": "coco/tfa_cos_3shot/model_final.pth",
        "COCO-detection/faster_rcnn_R_101_FPN_ft_all_5shot.yaml": "coco/tfa_cos_5shot/model_final.pth",
        "COCO-detection/faster_rcnn_R_101_FPN_ft_all_10shot.yaml": "coco/tfa_cos_10shot/model_final.pth",
        "COCO-detection/faster_rcnn_R_101_FPN_ft_all_30shot.yaml": "coco/tfa_cos_30shot/model_final.pth",
        # TFA w/ fc
        "COCO-detection/faster_rcnn_R_101_FPN_ft_fc_all_1shot.yaml": "coco/tfa_fc_1shot/model_final.pth",
        "COCO-detection/faster_rcnn_R_101_FPN_ft_fc_all_2shot.yaml": "coco/tfa_fc_2shot/model_final.pth",
        "COCO-detection/faster_rcnn_R_101_FPN_ft_fc_all_3shot.yaml": "coco/tfa_fc_3shot/model_final.pth",
        "COCO-detection/faster_rcnn_R_101_FPN_ft_fc_all_5shot.yaml": "coco/tfa_fc_5shot/model_final.pth",
        "COCO-detection/faster_rcnn_R_101_FPN_ft_fc_all_10shot.yaml": "coco/tfa_fc_10shot/model_final.pth",
        "COCO-detection/faster_rcnn_R_101_FPN_ft_fc_all_30shot.yaml": "coco/tfa_fc_30shot/model_final.pth",
        ### LVIS Detection ###
    }


def get_checkpoint_url(config_path):
    """
    Returns the URL to the model trained using the given config
    Args:
        config_path (str): config file name relative to FsDet's "configs/"
            directory, e.g., "COCO-detection/faster_rcnn_R_101_FPN_ft_all_1shot.yaml"
    Returns:
        str: a URL to the model
    """
    if config_path in _ModelZooUrls.CONFIG_PATH_TO_URL_SUFFIX:
        suffix = _ModelZooUrls.CONFIG_PATH_TO_URL_SUFFIX[config_path]
        return _ModelZooUrls.URL_PREFIX + suffix
    raise RuntimeError("{} not available in Model Zoo!".format(config_path))


def get_config_file(config_path):
    """
    Returns path to a builtin config file.
    Args:
        config_path (str): config file name relative to FsDet's "configs/"
            directory, e.g., "COCO-detection/faster_rcnn_R_101_FPN_ft_all_1shot.yaml"
    Returns:
        str: the real path to the config file.
    """
    cfg_file = pkg_resources.resource_filename(
        "fsdet", os.path.join("..", "configs", config_path)
    )
    if not os.path.exists(cfg_file):
        raise RuntimeError(
            "{} not available in Model Zoo!".format(config_path)
        )
    return cfg_file


def get(config_path, trained: bool = False):
    """
    Get a model specified by relative path under FsDet's official ``configs/`` directory.
    Args:
        config_path (str): config file name relative to FsDet's "configs/"
            directory, e.g., "COCO-detection/faster_rcnn_R_101_FPN_ft_all_1shot.yaml"
        trained (bool): If True, will initialize the model with the trained model zoo weights.
            If False, the checkpoint specified in the config file's ``MODEL.WEIGHTS`` is used
            instead; this will typically (though not always) initialize a subset of weights using
            an ImageNet pre-trained model, while randomly initializing the other weights.
    Example:
    .. code-block:: python
        from fsdet import model_zoo
        model = model_zoo.get("COCO-detection/faster_rcnn_R_101_FPN_ft_all_1shot.yaml", trained=True)
    """
    cfg_file = get_config_file(config_path)

    cfg = get_cfg()
    cfg.merge_from_file(cfg_file)
    if trained:
        cfg.MODEL.WEIGHTS = get_checkpoint_url(config_path)
    if not torch.cuda.is_available():
        cfg.MODEL.DEVICE = "cpu"

    model = build_model(cfg)
    DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
    return model
