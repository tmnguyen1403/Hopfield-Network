from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt

#Show image pixel:
def getImagePath(rootFolder,outputName):
    return os.path.join(rootFolder,outputName)

# 0(black) --> 1(black)
# 1(white) --> -1(white) is black, 1 is white
def bipolarToImage(value):
    #Black is 0
    if value == 1:
        return 1
    #White is 1 
    elif value == -1:
        return 0
    else:
        print("The function receive unknow value ")
        return 0

#Done: Convert bw image values to bipolar values   
outputFolder = f"Output"
numberOutput = [] 
for i in range(1,3):
    outputName = f"{i}_flip.txt"
    imagePath = getImagePath(outputFolder, outputName)
    imageMatrix = np.loadtxt(imagePath)
    # #imageBW = Image.open(imagePath)
    # imageArray = np.array(imageBW)
    # imageArray = imageArray.astype(int)
    numberOutput.append(imageMatrix)
print(numberOutput)
bwImages = [np.vectorize(bipolarToImage)(matrix) for matrix in numberOutput]

for i,image in enumerate(bwImages):
    plt.imsave(f"{i}_output.png",image,cmap='gray')