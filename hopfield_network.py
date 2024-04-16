from PIL import Image
import os
import numpy as np
from file_util import (getImagePath,imageToBiPolar,saveOutput,randomFlip,hardLimit)

#Done: Convert bw image values to bipolar values   
imageFolder = f"Images{os.path.sep}BlackWhite"
numberImages = [] 
for i in range(8):
    imageName = f"{i}_bw.png"
    imagePath = getImagePath(imageFolder, imageName)
    imageBW = Image.open(imagePath)
    imageArray = np.array(imageBW)
    imageArray = imageArray.astype(int)
    numberImages.append(imageArray.copy())

bipolarArrays = [np.vectorize(imageToBiPolar)(array) for array in numberImages]

maxRow = 10
maxCol = 10
#Debug values
for i,bipolarImage in enumerate(bipolarArrays):
    print(f"Number {i}")
    if bipolarImage.shape != (maxRow,maxCol):
        print(f"Image does not have the correct dimension. Expect: {(maxRow,maxCol)}. Image: {(bipolarImage.shape)}")
        exit(1)
    print(bipolarImage)

# Training
'''
TODO:
1. Initialize weight with the 8 examplars
2. Produce noisy input by randomly flip the bit of an examplar with the probability of 0.25
3. Train the Hopfield Network
'''


# TODO: 1. Initialize weight with the 8 examplars
identityMatrix = np.eye(100)
weight = np.zeros((maxRow*maxRow, maxCol*maxCol))
for i,imageArray in enumerate(bipolarArrays):
    weight += np.outer(imageArray,imageArray)
    weight -= identityMatrix
print("weight:\n", weight)
#exit(0)
#verify that the left diagonal is all zero
if not np.all(np.diag(weight) == 0):
    print("Error: Left diagonal values are not all zero")
    exit(1)

# Train to recognize 1
number = 7
inputImage = bipolarArrays[number].copy()

inputFlip = inputImage#np.vectorize(randomFlip)(inputImage)
#DEBUG: Save the flip input for retrain
inputFolder = "Input"
inputName = f"{number}_flip.txt"
inputPath = getImagePath(inputFolder,inputName)
np.savetxt(inputPath, inputFlip,fmt='%.0f')

#Dimension: 10x10 --> 1 x 100
inputFlatten = inputFlip.flatten()
print(f"flattenshape: {inputFlatten}")
prevInput = np.zeros(len(inputFlatten))
loopCounter = 0
originalFlatten = inputImage.flatten()
maxRun = 20
outputFolder = "Output"
while not np.all(inputFlatten == originalFlatten) and (loopCounter < maxRun):
    loopCounter += 1
    print(f"Epoch: {loopCounter}")
    # Asynchronous update
    for i in range(len(inputFlatten)):
        #dot product
        dotvalue = np.dot(weight[i],inputFlatten)
        print(f"dotvalue: {dotvalue} - threshold: {hardLimit(dotvalue)}")
        inputFlatten[i] = hardLimit(dotvalue)
        # if np.all(inputFlatten == originalFlatten):
        #     print(f"Converge. #loops {loopCounter}")
        #     break
        # else:
        #     prevInput = inputFlatten.copy()
    outputName = f"{number}_{loopCounter}.txt"
    outputData = inputFlatten.reshape(maxRow,maxCol)
    saveOutput(outputFolder,outputName,outputData)