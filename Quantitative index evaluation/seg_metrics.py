# -*- coding: utf-8 -*-


import numpy as np
import cv2
import os

""" 
混淆矩阵
P\L     P    N 
P      TP    FP 
N      FN    TN 
"""


def color_dict(labelFolder, classNum):
    colorDict = []
    ImageNameList = os.listdir(labelFolder)
    for i in range(len(ImageNameList)):
        ImagePath = labelFolder + "/" + ImageNameList[i]
        img = cv2.imread(ImagePath).astype(np.uint32)
        if (len(img.shape) == 2):
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB).astype(np.uint32)
        img_new = img[:, :, 0] * 1000000 + img[:, :, 1] * 1000 + img[:, :, 2]
        unique = np.unique(img_new)
        for j in range(unique.shape[0]):
            colorDict.append(unique[j])
        colorDict = sorted(set(colorDict))
        if (len(colorDict) == classNum):
            break
    colorDict_BGR = []
    for k in range(len(colorDict)):
        color = str(colorDict[k]).rjust(9, '0')
        color_BGR = [int(color[0: 3]), int(color[3: 6]), int(color[6: 9])]
        colorDict_BGR.append(color_BGR)
    colorDict_BGR = np.array(colorDict_BGR)
    colorDict_GRAY = colorDict_BGR.reshape((colorDict_BGR.shape[0], 1, colorDict_BGR.shape[1])).astype(np.uint8)
    colorDict_GRAY = cv2.cvtColor(colorDict_GRAY, cv2.COLOR_BGR2GRAY)
    return colorDict_BGR, colorDict_GRAY


def ConfusionMatrix(numClass, imgPredict, Label):
    mask = (Label >= 0) & (Label < numClass)
    label = numClass * Label[mask] + imgPredict[mask]
    count = np.bincount(label, minlength=numClass ** 2)
    confusionMatrix = count.reshape(numClass, numClass)
    return confusionMatrix


def OverallAccuracy(confusionMatrix):
    # acc = (TP + TN) / (TP + TN + FP + TN)
    OA = np.diag(confusionMatrix).sum() / confusionMatrix.sum()
    return OA


def Precision(confusionMatrix):
    precision = np.diag(confusionMatrix) / confusionMatrix.sum(axis=1)
    return precision


def Recall(confusionMatrix):
    recall = np.diag(confusionMatrix) / confusionMatrix.sum(axis=0)
    return recall


def F1Score(confusionMatrix):
    precision = np.diag(confusionMatrix) / confusionMatrix.sum(axis=1)
    recall = np.diag(confusionMatrix) / confusionMatrix.sum(axis=0)
    f1score = 2 * precision * recall / (precision + recall)
    return f1score


def IntersectionOverUnion(confusionMatrix):
    intersection = np.diag(confusionMatrix)
    union = np.sum(confusionMatrix, axis=1) + np.sum(confusionMatrix, axis=0) - np.diag(confusionMatrix)
    IoU = intersection / union
    return IoU


def MeanIntersectionOverUnion(confusionMatrix):
    intersection = np.diag(confusionMatrix)
    union = np.sum(confusionMatrix, axis=1) + np.sum(confusionMatrix, axis=0) - np.diag(confusionMatrix)
    IoU = intersection / union
    mIoU = np.nanmean(IoU)
    return mIoU


def Frequency_Weighted_Intersection_over_Union(confusionMatrix):
    freq = np.sum(confusionMatrix, axis=1) / np.sum(confusionMatrix)
    iu = np.diag(confusionMatrix) / (
            np.sum(confusionMatrix, axis=1) +
            np.sum(confusionMatrix, axis=0) -
            np.diag(confusionMatrix))
    FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
    return FWIoU


def dice(confusionMatrix):
    intersection = 2 * np.diag(confusionMatrix)
    union = np.sum(confusionMatrix, axis=1) + np.sum(confusionMatrix, axis=0)
    dice = intersection / union
    dice = np.nanmean(dice)
    return dice


#################################################################
LabelPath = r"miou_pr_dir copy"
PredictPath = r"miou_pr_dir"
classNum = 15
#################################################################

colorDict_BGR, colorDict_GRAY = color_dict(LabelPath, classNum)

labelList = os.listdir(LabelPath)
PredictList = os.listdir(PredictPath)

Label0 = cv2.imread(LabelPath + "//" + labelList[0], 0)

label_num = len(labelList)

label_all = np.zeros((label_num,) + Label0.shape, np.uint8)
predict_all = np.zeros((label_num,) + Label0.shape, np.uint8)
for i in range(label_num):
    Label = cv2.imread(LabelPath + "//" + labelList[i])
    Label = cv2.cvtColor(Label, cv2.COLOR_BGR2GRAY)
    label_all[i] = Label
    Predict = cv2.imread(PredictPath + "//" + PredictList[i])
    Predict = cv2.cvtColor(Predict, cv2.COLOR_BGR2GRAY)
    predict_all[i] = Predict

for i in range(colorDict_GRAY.shape[0]):
    label_all[label_all == colorDict_GRAY[i][0]] = i
    predict_all[predict_all == colorDict_GRAY[i][0]] = i

label_all = label_all.flatten()
predict_all = predict_all.flatten()

confusionMatrix = ConfusionMatrix(classNum, predict_all, label_all)
precision = Precision(confusionMatrix)
recall = Recall(confusionMatrix)
OA = OverallAccuracy(confusionMatrix)
IoU = IntersectionOverUnion(confusionMatrix)
FWIOU = Frequency_Weighted_Intersection_over_Union(confusionMatrix)
mIOU = MeanIntersectionOverUnion(confusionMatrix)
f1ccore = F1Score(confusionMatrix)
dice = dice(confusionMatrix)

for i in range(colorDict_BGR.shape[0]):
    try:
        import webcolors

        rgb = colorDict_BGR[i]
        rgb[0], rgb[2] = rgb[2], rgb[0]
        print(webcolors.rgb_to_name(rgb), end="  ")
    except:
        print(colorDict_GRAY[i][0], end="  ")
print("")
print("混淆矩阵:")
print(confusionMatrix)
print("精确度:")
print(precision)
print("召回率:")
print(recall)
print("F1-Score:")
print(f1ccore)
print("整体精度:")
print(OA)
print("IoU:")
print(IoU)
print("mIoU:")
print(mIOU)
print("FWIoU:")
print(FWIOU)
print("dice:")
print(dice)
