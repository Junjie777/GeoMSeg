import os
import random
import cv2.cv2 as cv
import numpy as np

path = r""  # image path
save_path = r""
file_list = os.listdir(path)

target1 = np.array([105, 84, 245])
target2 = np.array([47, 156, 215])
target3 = np.array([224, 251, 105])
target4 = np.array([31, 60, 243])
target5 = np.array([197, 213, 222])
target6 = np.array([113, 241, 234])
target7 = np.array([135, 204, 229])
target8 = np.array([48, 203, 228])
target9 = np.array([119, 129, 237])
target10 = np.array([233, 117, 248])
target11 = np.array([172, 253, 144])
target12 = np.array([130, 120, 245])
target13 = np.array([151, 253, 250])
target14 = np.array([150, 233, 186])
target15 = np.array([95, 221, 140])
j = 0
for i in file_list:
    print(i)
    img = os.path.join(path, i)
    save_img = os.path.join(save_path, i)
    src = cv.imread(img)
    h, w, c = src.shape
    mask = np.zeros((h, w, 3))
    # 查看都有什么类别
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
            else:
                print(src[i][o])
                j += 1
    cv.imwrite(save_img, mask)
    print(j)
