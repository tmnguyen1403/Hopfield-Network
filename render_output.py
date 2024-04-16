import numpy as np
import matplotlib.pyplot as plt
from file_util import (getImagePath,bipolarToImage)

#Done: Convert bw image values to bipolar values   
outputFolder = f"Output"
numberOutput = []
selected_number = 7
maxIteration = 20
for i in range(1,maxIteration+1):
    outputName = f"{selected_number}_{i}.txt"
    imagePath = getImagePath(outputFolder, outputName)
    imageMatrix = np.loadtxt(imagePath)
    numberOutput.append(imageMatrix)
#print(numberOutput)
bwImages = [np.vectorize(bipolarToImage)(matrix) for matrix in numberOutput]

outputImageFolder = "OutputImages"
for i,image in enumerate(bwImages):
    imageName = f"{selected_number}_{i}.png"
    imagePath = getImagePath(outputImageFolder,imageName)
    plt.imsave(imagePath,image,cmap='gray')