from PIL import Image
import os
import numpy as np
from file_util import (getImagePath,imageToBiPolar,saveOutput,randomFlip,hardLimit)

#Done: Convert bw image values to bipolar values   
imageFolder = f"Images{os.path.sep}BlackWhite/v1"
numberImages = []
examplarNumbers = [3,5]
for number in examplarNumbers:
    imageName = f"{number}_bw.png"
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
weight = np.zeros((maxRow*maxRow, maxCol*maxCol))
for i,imageArray in enumerate(bipolarArrays):
    weight += np.outer(imageArray,imageArray)
print("weight:\n", weight)
#Zero out the weight:
for i in range(len(weight)):
    weight[i][i] = 0
print("zero weight:\n", weight)
#exit(0)
#verify that the left diagonal is all zero
if not np.all(np.diag(weight) == 0):
    print("Error: Left diagonal values are not all zero")
    exit(1)

# Train to recognize number
for index,number in enumerate(examplarNumbers):
    inputImage = bipolarArrays[index].copy()

    inputFlip = np.vectorize(randomFlip)(inputImage)
    #DEBUG: Save the flip input for retrain
    inputFolder = "Input/v1"
    inputName = f"{number}_flip.txt"
    inputPath = getImagePath(inputFolder,inputName)
    np.savetxt(inputPath, inputFlip,fmt='%.0f')

    #Dimension: 10x10 --> 1 x 100
    inputFlatten = inputFlip.flatten()
    print(f"flattenshape: {inputFlatten}")
    prevInput = np.zeros(len(inputFlatten))
    loopCounter = 0
    originalFlatten = inputImage.flatten()
    maxRun = 2
    outputFolder = "Output/v1"
    # iTest = np.dot(weight, inputFlatten)
    # print("Itest\n",iTest,iTest.shape)
    # iHard = np.vectorize(hardLimit)(iTest)
    # print("Ihard\n",iHard,iHard.shape)
    # print("Ihard\n",iHard.reshape(maxRow,maxCol))
    # np.savetxt(inputPath, iHard.reshape(maxRow,maxCol),fmt='%.0f')
    divisor = 100
    while not np.all(inputFlatten == originalFlatten) and (loopCounter < maxRun):
        loopCounter += 1
        print(f"Epoch: {loopCounter}")
        # Asynchronous update
        for i in range(len(inputFlatten)):
            #dot product
            print(f"weight{i}", weight[i])
            dotvalue = np.dot(weight[i],inputFlatten)
            print(f"dotvalue: {dotvalue} - threshold: {hardLimit(dotvalue)}")
            inputFlatten[i] = hardLimit(dotvalue)
            #print(f"{i}:\n",inputFlatten)
            # if np.all(inputFlatten == originalFlatten):
            #     print(f"Converge. #loops {loopCounter}")
            #     break
            # else:
            #     prevInput = inputFlatten.copy()
            if i % divisor == 0:
                outputName = f"{number}_{loopCounter}_{i}.txt"
                outputData = inputFlatten.reshape(maxRow,maxCol)
                saveOutput(outputFolder,outputName,outputData)