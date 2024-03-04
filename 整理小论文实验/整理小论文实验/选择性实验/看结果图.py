# from zhengSnn import *
import time

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from scipy import io
from sklearn.metrics import precision_score, recall_score, f1_score

from SNN import *

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def main():
    # img = cv.imread("216041.jpg")
    # img = cv.imread("噪声图/noisy_img1.png")
    img = cv.imread("re.jpeg")
    f = 41004
    # img = cv.imread("../../BSDS500/data/images/train/"+str(f)+".jpg")
    img = np.array(img)
    img1 = np.zeros(img.shape)
    gray_img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    img[:, :, 0] = cv.GaussianBlur(gray_img, (3, 3), 1)

    data = io.loadmat("../最简化版本/BSDS500/data/groundTruth/train/" + str(f) + ".mat")
    y_true = data['groundTruth'][0][0][0][0][1]
    # print(data['groundTruth'][0][0][0][0][1])
    gt = np.zeros(y_true.shape)
    for i in range(0, y_true.shape[0]):
        for j in range(0, y_true.shape[1]):
            if y_true[i][j] == 1:
                gt[i][j] = 255

    stim = np.zeros(img.shape)
    outsideV = np.zeros(img.shape)
    edge1 = np.zeros(img.shape)
    for i in range(0, 1):
        for j in range(0, int(img.shape[1])):
            my_neuron = Neuron(0 - img[0][j][i])
            # 初始化神经元

            for k in range(0, img.shape[0]):
                signal = my_neuron.sensing(img[k][j][i])
                img1[k][j][i] = img[k][j][i]
                stim[k][j][i] = signal[0]
                outsideV[k][j][i] = signal[1]
                edge1[k][j][i] = signal[2]

    i = 0
    stim1 = np.zeros(img.shape)
    outsideV1 = np.zeros(img.shape)
    edge2 = np.zeros(img.shape)
    for i in range(0, 1):
        for j in range(0, int(img.shape[0])):
            my_neuron = Neuron(0 - img[j][0][i])
            for k in range(0, img.shape[1]):
                signal = my_neuron.sensing(img[j][k][i])
                # print(signal)
                img1[j][k][i] = img[j][k][i]
                stim1[j][k][i] = signal[0]
                outsideV1[j][k][i] = signal[1]
                edge2[j][k][i] = signal[2]

    outsideV1 = outsideV1[:, :, 0]
    edge1 = edge1[:, :, 0]
    edge2 = edge2[:, :, 0]
    edge = cv.Canny((img[:, :, 0]).astype(np.uint8), 100, 100)
    end_time = time.time()  # 程序结束时间
    out = np.zeros(edge1.shape)
    y_pre = np.zeros(edge1.shape)
    for i in range(edge1.shape[0]):
        for j in range(edge1.shape[1]):
            # print(mask_OTSU[i][j],mask_OTSU1[i][j])
            if edge1[i][j] == 1 or edge2[i][j] == 1:
                out[i][j] = 255
                y_pre[i][j] = 1

    o = np.zeros(edge.shape)
    for j in range(0, edge.shape[0]):
        for k in range(0, edge.shape[1]):
            if edge[j][k] == 255:
                o[j][k] = 1

    plt.figure()
    plt.subplot(131)
    plt.imshow(outsideV1, cmap='gray')
    plt.axis('off')  # 去掉坐标轴

    plt.subplot(132)
    plt.imshow(out, cmap='gray')
    plt.axis('off')  # 去掉坐标轴
    plt.subplot(133)
    plt.imshow(edge, cmap='gray')
    plt.axis('off')  # 去掉坐标轴
    plt.show()

    y_pre1 = np.reshape(o, [-1])
    y_true = np.reshape(y_true, [-1])
    y_pred = np.reshape(y_pre, [-1])

    p = precision_score(y_true, y_pred, average='binary')
    r = recall_score(y_true, y_pred, average='binary')
    f1score = f1_score(y_true, y_pred, average='binary')

    print("our", p, r, f1score)

    p = precision_score(y_true, y_pre1, average='binary')
    r = recall_score(y_true, y_pre1, average='binary')
    f1score = f1_score(y_true, y_pre1, average='binary')
    print("canny", p, r, f1score)


main()
