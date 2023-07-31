"""
This module provides functions to select the masks based on the scores or areas.
"""
import numpy as np
import torch
from torch import nn
import cv2


def make_score_map(masks, scores, score_cut=0.85, area_cut=250):
    """
    Make score map.
    This function is used to make a pixel map with the max mask score as the value.
    :param
        masks: torch.Tensor, masks.
        scores: torch.Tensor, scores.
        cut: float, cut on the score.
    :return: score map, contour list.
    """
    selected_idx = torch.logical_and(
        scores > score_cut, torch.sum(masks, dim=(1, 2)) > area_cut
    )  # do a area cut and score cut.

    scored_mask = torch.multiply(
        masks[selected_idx], scores[selected_idx][:, None, None]
    )  # create multilayer masks with scores as non-zero value
    max_score_map = torch.amax(scored_mask, dim=0)

    pixel_count_by_score = max_score_map.unique(return_counts=True)

    # do a second area cut to remove small areas after amax
    for i in range(pixel_count_by_score[0].shape[0]):
        if pixel_count_by_score[1][i] < area_cut and pixel_count_by_score[0][i] != 0:
            max_score_map[max_score_map == pixel_count_by_score[0][i]] = 0

    # now count the pixels of each score
    reduced_pixel_count_by_score = max_score_map.unique(return_counts=True)

    # masks generated by SAM are not always connected, so we need to remove small areas
    # make an empty score map [final_map, the same shape as max_score_map] and remake a score map with small areas removed
    # iterate through the scores and do the following if the score is not 0:
    # 1. use cv2.findContours to find the contours of the mask
    # 2. find the contour with the largest area, select it as the mask
    # 3. fill the contour with the score in final_map
    final_map = torch.zeros_like(max_score_map).cpu().numpy()
    contour_list = []
    for i in range(reduced_pixel_count_by_score[0].shape[0]):
        if reduced_pixel_count_by_score[0][i] != 0:
            im_bw = (max_score_map == reduced_pixel_count_by_score[0][i]).cpu().numpy()
            (cnts, _) = cv2.findContours(
                np.uint8(im_bw), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
            )
            c = max(cnts, key=cv2.contourArea)
            contour_list.append(c)

            cv2.drawContours(
                image=final_map,
                contours=[c],
                contourIdx=-1,
                color=float(reduced_pixel_count_by_score[0][i].cpu().numpy()),
                thickness=cv2.FILLED,
            )
    print("number of contours in score map:", len(contour_list))
    return final_map, contour_list


def make_hole_map(fmap, window_size=7, area_cut=50):
    # make a hole map
    # fmap: a numpy array of the score map
    # window_size should be an odd number
    # return: hole map, contour list
    m = nn.MaxPool2d(window_size, stride=1, padding=window_size // 2)
    holes = np.uint8(
        (m(torch.from_numpy(fmap[None, :, :]))).cpu().numpy()[0] == 0
    )  # if the max value in the window is 0, then the pixel is a hole

    # use cv2.findContours to find the contours of the holes
    (cnts, _) = cv2.findContours(holes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    hole_map = np.zeros_like(fmap)
    hole_contour_list = []
    for c in cnts:
        if cv2.contourArea(c) > area_cut:
            cv2.drawContours(
                image=hole_map,
                contours=[c],
                contourIdx=-1,
                color=np.random.random() * 0.15 + 0.85,  # random color above 0.5
                thickness=cv2.FILLED,
            )
            hole_contour_list.append(c)
    print("number of contours in hole map:", len(hole_contour_list))
    return hole_map, hole_contour_list
