from PIL import Image
import numpy as np
from file_util import (getImagePath)
'''
TODO:
# Data preparation
1. Create png image of 0-7 numbers in 8bit scale
2. Save them as 10*10

'''

#Done: Convert images to blackwhite images
imageFolder = "Images/v1"
bwFolder = "Images/BlackWhite/v1"
for i in range(8):
    imageName = f"{i}.png"
    imagePath = getImagePath(imageFolder, imageName)
    image = Image.open(imagePath)
    imageBW = image.convert("1")
    bwName = f"{i}_bw.png"
    bwPath = getImagePath(bwFolder,bwName)
    imageBW.save(bwPath)
exit()