"""
This module is used to get the mask of the image.
A few methods are provided: auto, prompt.
"""

import cv2
import torch
import numpy as np
from segment_anything import sam_model_registry, SamPredictor


def mask_from_prompt(
    image: np.ndarray, sam_checkpoint, model_type, num_seeds=400, batch_size=200, device = None
):
    """
    Get mask from prompt.
    This function is used to get the mask of the image from random points on the image.
    :param
        image: np.ndarray, image to be processed.
        sam_checkpoint: str, path to sam checkpoint.
        model_type: str, model type.
        num_seeds: int, number of seeds.
        batch_size: int, batch size. SAM uses a lot of GPU memory, so we need to split the seeds into batches.
    :return: torch.Tensor, torch.Tensor, mask and score.
    """

    print("Image size:", image.shape)

    _device=device

    if _device is None:
        _device = "cpu"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=_device)

    predictor = SamPredictor(sam)

    predictor.set_image(image)

    seeds = torch.rand(num_seeds, 1, 2, device=predictor.device)
    seeds[:, :, 0] *= image.shape[0]
    seeds[:, :, 1] *= image.shape[1]

    transformed_coords = predictor.transform.apply_coords_torch(seeds, image.shape[:2])

    masks = []
    scores = []
    coords_batch = torch.split(transformed_coords, batch_size)
    for coords in coords_batch:
        _masks, _scores, _ = predictor.predict_torch(
            point_coords=coords,
            point_labels=torch.ones(batch_size, 1),
            boxes=None,
            multimask_output=False,
        )
        torch.cuda.empty_cache()
        masks.append(_masks)
        scores.append(_scores)
    return torch.cat(masks).squeeze(), torch.cat(scores).squeeze()
