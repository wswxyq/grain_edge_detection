import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
from segment_anything import sam_model_registry, SamPredictor
import torch.nn as nn


imagepath = "IMAGES/AP1O4/000001_RT_01_SemAdcDef1_Dedicated_BSD_RID_000001.tif"
image = cv2.imread(imagepath)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


sam_checkpoint = "model/sam_vit_h_4b8939.pth"
model_type = "vit_h"

device = "cuda:5"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

predictor = SamPredictor(sam)

predictor.set_image(image)


num_seeds = 800
seeds = torch.rand(
    num_seeds, 1, 2, device=predictor.device
)
seeds[:, :, 0] *= image.shape[0]
seeds[:, :, 1] *= image.shape[1]


transformed_coords = predictor.transform.apply_coords_torch(seeds, image.shape[:2])
masks, scores, _ = predictor.predict_torch(
    point_coords=transformed_coords,
    point_labels=torch.ones(num_seeds, 1),
    boxes=None,
    multimask_output=False,
)


cut = 0.85  # cut on the score
area_cut = 2500  # cut on area (unit: pixels)
scores_w_cut = (scores > cut).clone().long() * scores  # set score below cut to be zero


scored_mask = torch.multiply(
    masks.squeeze(), scores_w_cut.squeeze()[:, None, None]
)  # create multilayer masks with scores as non-zero value


max_score_map = torch.amax(scored_mask, dim=0)

pixel_count_by_score = max_score_map.unique(return_counts=True)


for i in range(pixel_count_by_score[0].shape[0]):
    if pixel_count_by_score[1][i] < area_cut and pixel_count_by_score[0][i] != 0:
        max_score_map[max_score_map == pixel_count_by_score[0][i]] = 0


reduced_pixel_count_by_score = max_score_map.unique(return_counts=True)


# now fill the holes with 0

window_size = 25  # should be an odd number
m = nn.MaxPool2d(window_size, stride=1, padding=window_size // 2)
holes = m(max_score_map[None, :, :])[0]

edge_img = (max_score_map.cpu().numpy() != 0).astype(int) + (
    holes.cpu().numpy() == 0
).astype(int)


# why is some boundaries missing? Because we only plot the masks without marking their boundaries.
# We need to plot the boundary of these masks.
# The boundary points: are non-zero score points && has other values as neighbor points


x = []
y = []
for i in range(len(reduced_pixel_count_by_score[0])):
    if reduced_pixel_count_by_score[0][i] != 0:
        single_score_img = (
            (max_score_map == reduced_pixel_count_by_score[0][i]).cpu().numpy()
        )
        image_cv = np.uint8(single_score_img)
        contours, _ = cv2.findContours(
            image_cv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        x += list(contours[0][:, 0, 0])
        y += list(contours[0][:, 0, 1])


close_grain_edge_enhance_map = torch.tensor(edge_img, dtype=float)
imax = close_grain_edge_enhance_map.shape[1] - 1
jmax = close_grain_edge_enhance_map.shape[0] - 1
linewd = 8
for i, j in zip(x, y):
    close_grain_edge_enhance_map[
        max(j - linewd // 2, 0) : min(j + linewd // 2, jmax),
        max(i - linewd // 2, 0) : min(i + linewd // 2, imax),
    ] = 0.0
_window_size = 3  # should be an odd number
plt.figure(figsize=(16, 8))

plt.subplot(121)
plt.imshow(image)
plt.axis("off")
plt.axis("equal")
plt.title("Original Image"), plt.xticks([]), plt.yticks([])

plt.subplot(122)
plt.axis("off")
plt.axis("equal")
plt.imshow(close_grain_edge_enhance_map, cmap="gray")
ax = plt.gca()
plt.title("Enhanced Edge plot"), plt.xticks([]), plt.yticks([])
# white->1, black->0
plt.savefig("asout/testplot.png")
