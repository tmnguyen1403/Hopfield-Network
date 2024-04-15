from PIL import Image
import numpy as np

'''
TODO:
# Data preparation
1. Create png image of 0-7 numbers in 8bit scale
2. Save them as 10*10
3. Read and convert those images to numpy array
4. Convert the images to black and white
5. Convert the value to bipolar
--> Those are the 8 examplars that we need

# Training
1. Initialize weight with the 8 examplars
2. Produce noisy input by randomly flip the bit of an examplar with the probability of 0.25
3. Train the Hopfield Network

'''
#Convert to black and white
# image = Image.open("0.png")
# image_bw = image.convert("1")
# image_bw.save("0_bw.png")
#image.show()

#Show image pixel:
image_bw = Image.open("0_bw.png")
image_array = np.array(image_bw)
image_array = image_array.astype(int)
print("Image shape: ", image_array.shape)
print("pixel: ", image_array)

# 0 is black, 1 is white
# convert to 1 is black, and -1 is white
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
image_bipolar = np.vectorize(imageToBiPolar)(image_array)
print("vector: ", vector)
# image_array = np.array(image)
# print("Image shape: ", image_array.shape)
# print("pixel: ", image_array)