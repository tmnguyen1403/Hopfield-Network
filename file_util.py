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
    

def compare_image(im1,im2,positive=1):
    # compare pixel by pixel
    # then calculate the score
    r1,c1 = len(im1),len(im1[0])
    r2,c2 = len(im2),len(im2[0])
    if r1 != r2 or c1 != c2:
        print("Two images have different size")
        return 0
    similar_count = 0.0
    total_positive = 0.0
    for i in range(r1):
        for j in range(c1):
            if im1[i][j] == positive:
                if im1[i][j] == im2[i][j]:
                    similar_count += 1.0
                total_positive += 1.0
            elif im2[i][j] == positive:
                #penalty the similar count
                similar_count -= 1.0
            
    similar_score = similar_count / total_positive
    return similar_score

def compare_state(expect_state,result_state,positive=1):
    # compare pixel by pixel
    # then calculate the score
    n1 = len(expect_state)
    n2 = len(result_state)
    if n1 != n2:
        print("Two states have different size")
        return 0
    similar_count = 0.0
    total_positive = 0.0
    for i,value in enumerate(expect_state):
        if value == positive:
            if value == result_state[i]:
                similar_count += 1.0
            total_positive += 1.0
        elif result_state[i] == positive:
            #penalty the similar count
            similar_count -= 1.0
    similar_score = similar_count / total_positive
    return similar_score

def test_f(input,assert_func,func,msg = ""):
    print(msg)
    result = func(*input)
    assert_func(result)

def assert_img_score(expect,result):
    assert(round(expect,2)==round(result,2))

if __name__ == "__main__":
    a = [[1,1,1,1],[1,1,1,1],[1,1,1,1]]
    test_f((a,a),lambda x: assert_img_score(1.0,x),compare_image,"Test score of the same image")
    b = np.copy(a)
    b[0][0] = -1
    b[0][1] = -1

    test_f((a,b),lambda x: assert_img_score(0.83,x),compare_image,"Test score of the same image")
