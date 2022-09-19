import os
import random
import cv2 as cv
import numpy as np
path= r""# image path
file_list=os.listdir(path)

target1=np.array([248,248,247])
target2=np.array([249,161,228])
target3=np.array([109,121,218])
target4=np.array([242,242,171])
target5=np.array([146,133,249])
target6=np.array([60,60,60])
target7=np.array([183,240,236])
target8=np.array([43,153,200])
target9=np.array([69,207,150])
target10=np.array([111,234,246])
target11=np.array([99,209,239])
target12=np.array([64,64,64])
target13=np.array([189,158,241])
target14=np.array([65,65,65])
target15=np.array([144,154,218])
target16=np.array([67,67,67])
target17=np.array([254,254,254])
j = 0
for i in file_list:
    print(i)
    img=os.path.join(path,i)
    src=cv.imread(img)
    h,w,c=src.shape
    mask=np.zeros((h,w,3))
    #查看都有什么类别
    for i in range(h):
        for o in range(w):
            if (src[i][o] == target1).all():
                mask[i][o] = np.array([1, 1, 1])
            elif (src[i][o] == target2).all():
                 mask[i][o] = np.array([2, 2, 2])
            elif (src[i][o] == target3).all():
                 mask[i][o] = np.array([3, 3, 3])
            elif (src[i][o] == target4).all():
                 mask[i][o] = np.array([4, 4, 4])
            elif (src[i][o] == target5).all():
                 mask[i][o] = np.array([5, 5, 5])
            elif (src[i][o] == target6).all():
                 mask[i][o] = np.array([6, 6, 6])
            elif (src[i][o] == target7).all():
                mask[i][o] = np.array([7, 7, 7])
            elif (src[i][o] == target8).all():
                mask[i][o] = np.array([8, 8, 8])
            elif (src[i][o] == target9).all():
                mask[i][o] = np.array([9, 9, 9])
            elif (src[i][o] == target10).all():
                mask[i][o] = np.array([10, 10, 10])
            elif (src[i][o] == target11).all():
                mask[i][o] = np.array([11, 11, 11])
            elif (src[i][o] == target12).all():
                mask[i][o] = np.array([12, 12, 12])
            elif (src[i][o] == target13).all():
                mask[i][o] = np.array([13, 13, 13])
            elif (src[i][o] == target14).all():
                mask[i][o] = np.array([14, 14, 14])
            elif (src[i][o] == target15).all():
                mask[i][o] = np.array([15, 15, 15])
            elif (src[i][o] == target16).all():
                mask[i][o] = np.array([16, 16, 16])
            elif (src[i][o] == target17).all():
                mask[i][o] = np.array([17, 17, 17])
            else:
                print(src[i][o])
                j += 1
    cv.imwrite(img, mask)
    print(j)