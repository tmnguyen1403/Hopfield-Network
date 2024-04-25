from PIL import Image
import os
import ast 
import numpy as np
import random
from file_util import (getImagePath,imageToBiPolar,saveOutput,randomFlip,hardLimit,compare_image)


maxRow = 10
maxCol = 10

# Obtain exemplars from manually generatated image files.
# Parameters: Folder, list of exemplers to train eg: [3,5]
def exemplarFromImageFiles(folderpath, listNum):
    numberImages = []
    exemplarNumbers = listNum
    for number in exemplarNumbers:
        imageName = f"{number}_bw.png"
        imagePath = getImagePath(folderpath, imageName)
        imageBW = Image.open(imagePath)
        imageArray = np.array(imageBW)
        imageArray = imageArray.astype(int)
        numberImages.append(imageArray.copy())
    return numberImages


# Read exemplars from text file with exemplars matrices.
def exemplarFromMatrix(folder_path):
    exemplars = []
    for i in range(8):
        filename = f"{i}.txt"
        file_path = os.path.join(folder_path,filename)
        with open(file_path, "r") as f:
            data = ast.literal_eval(f.read())
            exemplars.append(np.array(data))
    return np.array(exemplars)


# Two options to read exemplars:
#OPTION 1:
#   bipolarArrays that was produced from Minh's images.
'''exemplarNumbers = [3,5]
imageFolder = f"Images{os.path.sep}BlackWhite/v1"
numberImages = exemplarFromImageFiles(imageFolder, exemplarNumbers)
bipolarArrays = [np.vectorize(imageToBiPolar)(array) for array in numberImages]
'''

#OPTION 2:
#   bipolarArrays that was produced using Sootatt's drawing pad.
exemplarNumbers = [i for i in range(0,8)]
rawdataFolder = f"{os.getcwd()}{os.path.sep}input{os.path.sep}new1"
bipolarArrays = exemplarFromMatrix(rawdataFolder)



#Debug values
for i,bipolarImage in enumerate(bipolarArrays):
    #print(f"Number {i}")
    if bipolarImage.shape != (maxRow,maxCol):
        print(f"Image does not have the correct dimension. Expect: {(maxRow,maxCol)}. Image: {(bipolarImage.shape)}")
        exit(1)
    #print(bipolarImage)

# Training
'''
TODO:
1. Initialize weight with the 8 exemplars
2. Produce noisy input by randomly flip the bit of an exemplar with the probability of 0.25
3. Train the Hopfield Network
'''

# TODO: 1. Initialize weight with the 8 exemplars
weight = np.zeros((maxRow*maxRow, maxCol*maxCol))
for i,imageArray in enumerate(bipolarArrays):
    weight += np.outer(imageArray,imageArray)

#print("weight:\n", weight)
#Zero out the weight:
for i in range(len(weight)):
    weight[i][i] = 0
#print("zero weight:\n", weight)
#verify that the left diagonal is all zero
if not np.all(np.diag(weight) == 0):
    print("Error: Left diagonal values are not all zero")
    exit(1)

def recallNumber(exemplars, noisyInput, w, expect_examplar):
    global maxRow, maxCol
    max_iteration = 100
    statecp = np.copy(noisyInput)
    # Asynchronous update:
    for j in range(max_iteration):
        state =np.copy(statecp)
        # Randomize the order on which neuron to update first.
        update_order = np.random.permutation(maxRow * maxCol)
        for i in update_order:
            # if i % 50 == 0:
            #   print("iteration:", j, "; Updating neuron:",i)
            activation = np.dot(w[i],state) + state[i]
            if activation > 0:
                state[i] = 1
            else:
                state[i] = -1
            # Check for convergence:
            if np.array_equal(expect_examplar,state):
                print(f"\nConverge at iteration: {j} - update: {i}\n")
                return state
    return state


# --------------------------------------------------------------------------------------------------------------
flattened_bipolarArrays = []
for array in bipolarArrays:
    flattened_bipolarArrays.append(array.flatten())

#Creating noisy input of number 'target':
target = 2
chosen_img = flattened_bipolarArrays[target]
noisyInput = np.copy(chosen_img)
# Randomly select 5-15 indices to flip: 
flip_index = random.sample(range(0, 100), random.randint(2, 4))
for i in flip_index:
    noisyInput[i] = noisyInput[i] * -1


result = recallNumber(flattened_bipolarArrays,noisyInput,weight,chosen_img)
if len(result) > 0:
    similar_score = compare_image(np.reshape(chosen_img,(10,10)),np.reshape(result,(10,10)))
    print(f"similar_score: {similar_score}")    

noisyInput = np.array(['X' if x==1 else ' ' for x in noisyInput])
noisyInput = noisyInput.reshape(10,10)
print("noisyInput:\n",noisyInput)


original = np.array(['X' if x==1 else ' ' for x in flattened_bipolarArrays[target]])
original = original.reshape(10,10)
print("original:\n",original)


if result is not None:
    result = np.array(['X' if x==1 else ' ' for x in result])
    result = result.reshape(10,10)
    print("result:\n", result)

print("-------------------------------------------")

# --------------------------------------------------------------------------------------------------------------
'''

def noisyExemplarFromMatrix(filepath,flipList):
    exemplars = []
    for i in flipList:
        filename = f"f{i}.txt"
        with open(filepath + "\\" + filename, "r") as f:
            data = ast.literal_eval(f.read())
            exemplars.append(np.array(data))
    return np.array(exemplars)


flipList = [0,1,2,7]
rawFlippedFolder = os.getcwd()+"\\input\\flipped"
noisyInput = noisyExemplarFromMatrix(rawFlippedFolder,flipList)[1].flatten()

target = 1
result = recallNumber(flattened_bipolarArrays,noisyInput,weight)
noisyInput = np.array(['X' if x==1 else ' ' for x in noisyInput])
noisyInput = noisyInput.reshape(10,10)
print("noisyInput:\n",noisyInput)

original = np.array(['X' if x==1 else ' ' for x in flattened_bipolarArrays[target]])
original = original.reshape(10,10)
print("original:\n",original)

result = np.array(['X' if x==1 else ' ' for x in result])
result = result.reshape(10,10)
print("result:\n", result)
'''

'''
    # Train to recognize number
    for index,number in enumerate(exemplarNumbers):
        inputImage = bipolarArrays[index].copy()
        inputFlip = np.vectorize(randomFlip)(inputImage)
        #print(inputImage)
        #print(inputFlip)
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
                    saveOutput(outputFolder,outputName,outputData)'''