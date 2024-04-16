#Show image pixel:
import os
import numpy as np

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

def saveOutput(outputFolder,outputName,outputData):
    # Output inspect
    outputPath = getImagePath(outputFolder,outputName)
    np.savetxt(outputPath,outputData,fmt="%.0f")

#Random flip input with 0.25 probability
def randomFlip(x,prob=0.25):
    if np.random.rand() >= prob:
        return -1.0*x
    return x

def hardLimit(x):
    if x >= 0:
        return 1
    else:
        return -1