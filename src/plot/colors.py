import random
import numpy as np

def RGBgen(length, lower=100, upper=255):
    # Generate a list of RGB colors, each value is different
    # length: number of colors to generate
    # return: list of RGB colors
    # RGB values are in range [0, 255]
    rgb_set = set()
    numRGB = 0
    while numRGB < length:
        new_rgb = (
            random.randint(lower, upper),
            random.randint(lower, upper),
            random.randint(lower, upper),
        )
        if new_rgb not in rgb_set:
            rgb_set.add(new_rgb)
            numRGB += 1
    return list(rgb_set)


def float2RGB(floatlist: list):
    # Convert a map of float numbers to RGB colors
    # floatlist: list of float numbers
    # return: list of RGB colors
    # 0 is always mapped to black
    rgbmap = {}
    rgblength = len(set(floatlist))
    rgblist = RGBgen(rgblength)
    for i, f in enumerate(set(floatlist)):
        rgbmap[f] = rgblist[i]
    rgbmap[0] = (0, 0, 0)
    return rgbmap

def floatIMG2RGB(floatimg: np.ndarray):
    # Convert a map of float numbers to RGB colors
    # floatimg: a numpy array of float numbers, shape: (height, width)
    # return: a numpy array of RGB colors, shape: (height, width, 3)
    # 0 is always mapped to black
    rgbmap = float2RGB(np.unique(floatimg))
    RGBimg = np.zeros((floatimg.shape[0], floatimg.shape[1], 3), dtype=np.uint8)
    for i in range(floatimg.shape[0]):
        for j in range(floatimg.shape[1]):
            RGBimg[i, j, :] = rgbmap[floatimg[i, j]]
    return RGBimg