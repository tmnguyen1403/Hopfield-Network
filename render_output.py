import numpy as np
import matplotlib.pyplot as plt
from file_util import (getImagePath,bipolarToImage)
import os
#Done: Convert bw image values to bipolar values   
outputFolder = "Output/v1"
maxNumber = 2
divisor = 4 
for selected_number in range(maxNumber):
    numberOutput = []
    maxIteration = 2
    for i in range(maxIteration):
        for j in range(0,100,divisor):
            outputName = f"{selected_number}_{i}_{j}.txt"
            imagePath = getImagePath(outputFolder, outputName)
            if not os.path.exists(imagePath):
                print(f"Error: File {imagePath} does not exist")
                break
            imageMatrix = np.loadtxt(imagePath)
            numberOutput.append(imageMatrix)
    #print(numberOutput)
    bwImages = [np.vectorize(bipolarToImage)(matrix) for matrix in numberOutput]

    outputImageFolder = "OutputImages/v1"
    for image in bwImages:
        for i in range(maxIteration):
            for j in range(0,100,divisor):
                imageName = f"{selected_number}_{i}_{j}.png"
                imagePath = getImagePath(outputImageFolder,imageName)
                plt.imsave(imagePath,image,cmap='gray')