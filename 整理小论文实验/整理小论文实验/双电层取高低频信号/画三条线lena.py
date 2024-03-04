# from zhengSnn import *
from SNN import *
import cmath
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import random
import matplotlib.pylab as mp
from sklearn.metrics import mean_squared_error

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def main():
    img = cv.imread("lena.jpg")
    img = np.array(img, int)
    print(img.shape)

    i = 0
    stim = np.zeros(img.shape)
    s = np.zeros(img.shape)
    outsideV = np.zeros(img.shape)
    for i in range(0, 1):
        for j in range(0, int(img.shape[0])):
            my_neuron = Neuron(0 - img[j][0][i])
            # 初始化神经元
            for k in range(0, img.shape[1]):
                signal = my_neuron.sensing(img[j][k][i])
                # print(signal)
                stim[j][k][i] = signal[0]
                outsideV[j][k][i] = signal[1]

    stim = stim
    stim1 = np.abs(stim)
    outsideV = outsideV
    # stim = cv.cvtColor(stim.astype(np.uint8), cv.COLOR_RGB2GRAY)
    # gray_img = cv.cvtColor(stim1.astype(np.uint8), cv.COLOR_RGB2GRAY)
    # gray_img1 = cv.cvtColor(img.astype(np.uint8), cv.COLOR_RGB2GRAY)
    ret1, mask_OTSU1 = cv.threshold(img[:, :, 0].astype(np.uint8), 200, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    ret2, mask_OTSU = cv.threshold(stim1[:, :, 0].astype(np.uint8), 200, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    a = []
    b = []
    c = []
    for i in range(1, 481):
        a.append(img[5, i, 0] - img[5, i - 1, 0])
        b.append(img[105, i, 0] - img[105, i - 1, 0])
        c.append(img[205, i, 0] - img[205, i - 1, 0])

    plt.figure()
    plt.subplot(131)
    plt.imshow(img[:, :, 0], cmap='gray')
    plt.axis('off')  # 去掉坐标轴
    plt.subplot(132)
    plt.imshow(stim[:, :, 0], cmap='gray')
    plt.axis('off')  # 去掉坐标轴
    plt.subplot(133)
    plt.imshow(outsideV[:, :, 0], cmap='gray')
    plt.axis('off')  # 去掉坐标轴

    plt.figure()
    plt.subplot(131)
    plt.plot(img[5, 1:481, 0], 'r')
    plt.plot(outsideV[5, 1:481, 0], 'black')

    plt.subplot(132)
    plt.plot(img[105, 1:481, 0], 'r')
    plt.plot(outsideV[105, 1:481, 0], 'black')

    plt.subplot(133)
    plt.plot(img[205, 1:481, 0], 'r')
    plt.plot(outsideV[205, 1:481, 0], 'black')

    plt.figure()
    plt.subplot(131)
    # outsideV[0, 0, 0] = img[0, 0, 0]
    # plt.plot(s[5, 0:512, 0])

    plt.plot(a, 'black')
    plt.plot(stim[5, 1:512, 0], 'r')

    plt.subplot(132)
    # outsideV[256, 0, 0] = img[256, 0, 0]
    # plt.plot(s[205, 0:512, 0])

    plt.plot(b, 'black')
    plt.plot(stim[105, 1:481, 0], 'r')

    plt.subplot(133)
    # outsideV[407, 0, 0] = img[407, 0, 0]
    # plt.plot(s[405, 0:512, 0])

    plt.plot(c, 'black')
    plt.plot(stim[205, 1:481, 0], 'r')
    plt.show()


main()
