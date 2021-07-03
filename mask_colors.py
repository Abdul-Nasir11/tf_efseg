from PIL import Image
import numpy as np



BDD_palette =[0,0,0,217,83,79,91,192,222]


def BDD_color_out(mask):
    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(BDD_palette)

    return new_mask