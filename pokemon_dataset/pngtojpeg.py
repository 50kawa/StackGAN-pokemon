# coding: utf-8
import codecs
import sys
import pickle
import os
from PIL import Image

ls_file_name = os.listdir("pokemon-images-and-types/images")

for fname in ls_file_name:
    if fname.split(".")[1] == "jpg":
        continue
    jpg_name = fname.rstrip("png") + "jpg"
    image = Image.open(os.path.join("pokemon-images-and-types/images", fname)).convert('RGBA')
    background = Image.new("RGBA", image.size, (255, 255, 255))
    alpha_composite = Image.alpha_composite(background, image)
    alpha_composite = alpha_composite.convert("RGB")
    alpha_composite.save(os.path.join("pokemon-images-and-types/jpg_images", jpg_name),"JPEG",quality=95)