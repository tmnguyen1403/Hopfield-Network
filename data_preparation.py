from PIL import Image
import numpy as np
import os

'''
TODO:
# Data preparation
1. Create png image of 0-7 numbers in 8bit scale
2. Save them as 10*10

'''
#Convert to black and white
# image = Image.open("0.png")
# image_bw = image.convert("1")
# image_bw.save("0_bw.png")
#image.show()

#Show image pixel:
def getImagePath(rootFolder,imageName):
    return os.path.join(rootFolder,imageName)

#Done: Convert images to blackwhite images
imageFolder = "Images"
for i in range(8):

    imageName = f"{i}.png"
    imagePath = getImagePath(imageFolder, imageName)
    image = Image.open(imagePath)
    imageBW = image.convert("1")
    bwName = f"{i}_bw.png"
    bwPath = getImagePath(imageFolder,bwName)
    imageBW.save(bwPath)
exit()