from PIL import Image
import numpy as np
import os

#Show image pixel:
def getImagePath(rootFolder,imageName):
    return os.path.join(rootFolder,imageName)

# 0(black) --> 1(black)
# 1(white) --> -1(white) is black, 1 is white
def imageToBiPolar(value):
    #Black is 1
    if value == 0:
        return 1
    #White is -1 
    elif value == 1:
        return -1
    else:
        print("The function receive unknow value ")
        return 0

#Done: Convert bw image values to bipolar values   
imageFolder = f"Images{os.path.sep}BlackWhite"
numberImages = [] 
for i in range(8):
    imageName = f"{i}_bw.png"
    imagePath = getImagePath(imageFolder, imageName)
    imageBW = Image.open(imagePath)
    imageArray = np.array(imageBW)
    imageArray = imageArray.astype(int)
    numberImages.append(imageArray)

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

#verify that the left diagonal is all zero
print(f"Left diagonal values are all zero: {np.all(np.diag(weight) == 0)}")

# Train to recognize 1
number = 2
inputImage = bipolarArrays[number]
#Random flip input with 0.25 probability
def randomFlip(x,prob=0.25):
    if np.random.rand() >= prob:
        return -1.0*x
    return x
inputFlip = np.vectorize(randomFlip)(inputImage)
#Save the flip input for retrain
inputFolder = "Input"
inputName = f"{number}_flip.txt"
inputPath = getImagePath(inputFolder,inputName)
np.savetxt(inputPath, inputFlip,fmt='%.0f')

#Asynchronous update
#Dimension: 10x10 --> 1 x 100
def threshold(x):
    if x > 0:
        return 1
    else:
        return -1

inputFlatten = inputFlip.flatten()
print(f"flattenshape: {inputFlatten}")
prevInput = np.zeros(len(inputFlatten))
loopCounter = 0
originalFlatten = inputImage.flatten()
while not np.all(inputFlatten == prevInput):
    for i in range(len(inputFlatten)):
        loopCounter += 1
        print(f"Epoch: {loopCounter}")
        #dot product
        dotvalue = np.dot(weight[i],inputFlatten)
        print(f"dotvalue: {dotvalue} - threshold: {threshold(dotvalue)}")
        inputFlatten[i] = threshold(dotvalue)
        if np.all(inputFlatten == prevInput):
            print(f"Converge. #loops {loopCounter}")
            break
        else:
            prevInput = inputFlatten.copy()

# Output inspect
outputFolder = "Output"
outputName = inputName
outputPath = getImagePath(outputFolder,outputName)
outputImage = inputFlatten.reshape(maxRow,maxCol)
np.savetxt(outputPath,outputImage,fmt="%.0f")

