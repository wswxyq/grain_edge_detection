# %%
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
from segment_anything import sam_model_registry, SamPredictor
import torch.nn as nn
import src.SAM.get_mask as get_mask
import src.SAM.map as map
import src.plot.colors as colors


imagepath = "./sem3.png"
image = cv2.imread(imagepath)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# %%
sam_checkpoint = "sam_vit_h_4b8939.pth"
model_type = "vit_h"
num_seeds = 1200
batch_size = 200

# %%
masks, scores = get_mask.mask_from_prompt(
    image,
    sam_checkpoint,
    model_type,
    num_seeds=num_seeds,
    batch_size=batch_size,
    device="cuda",
)
print("from SAM:", masks.shape, scores.shape)

# %%
import src.SAM.map as map

cut = 0.8  # cut on the score
area_cut = 250  # cut on area (unit: pixels)
fmap, cl = map.make_score_map(masks, scores, score_cut=cut, area_cut=area_cut)

# %%
hmap, hcl = map.make_hole_map(fmap, window_size=7)

# %%
allmap = fmap + hmap
rgbmap = colors.floatIMG2RGB(allmap)

# %%
plt.figure(figsize=(16, 8))

plt.subplot(121)
plt.imshow(image)
plt.axis("off")
plt.axis("equal")
plt.title("Original Image"), plt.xticks([]), plt.yticks([])

plt.subplot(122)
plt.axis("off")
plt.axis("equal")
plt.imshow(rgbmap, cmap="gray")
ax = plt.gca()
plt.title("Enhanced Edge plot"), plt.xticks([]), plt.yticks([])

plt.savefig("testplot.png")

# %%
