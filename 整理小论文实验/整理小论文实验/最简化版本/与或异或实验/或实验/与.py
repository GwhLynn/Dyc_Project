# from zhengSnn import *
from SNN import *
import numpy as np
import matplotlib.pyplot as plt
import random

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def main():
    # my_neuron = Neuron()
    stimulus = np.zeros(100)
    stimulus1 = np.ones(100)

    s = np.concatenate((stimulus, stimulus1, stimulus1, stimulus), axis=0)
    s1 = np.concatenate((stimulus, stimulus, stimulus1, stimulus1), axis=0)

    # stimulus1 = np.ones(100)

    w = np.loadtxt(open("1.csv", "rb"))
    print(w)
    #    plt.figure(figsize=(6,6))
    #    plt.plot(stimulus)
    #    plt.show()
    neuron1 = Neuron(-w[0] * s[0] - (1 - w[0]) * s1[0])
    a = stimulus.size
    stim = []
    u = []
    i = 0
    re = []
    state = []
    flag = 0
    while i < 400:
        signal1 = neuron1.sensing(w[i] * s[i] + (1 - w[i]) * s1[i])
        i += 1
        stim.append(signal1[0])
        u.append(signal1[1])
        state.append(signal1[2])
        if signal1[2] == 1:
            flag = 1
    np.savetxt("or.csv", state)
    print(np.var(stim))
    if flag == 1:
        print("是异")
    else:
        print("是同")
    plt.figure()
    plt.subplot(131)
    plt.plot(stimulus, 'black')
    plt.plot(stimulus1, 'red')
    plt.ylim(-0.5, 1.5)
    plt.subplot(132)
    plt.plot(u)
    plt.subplot(133)
    plt.plot(state)
    plt.figure()
    plt.plot(s + 6)
    plt.plot(s1 + 4)
    plt.plot(np.array(u) + 2)
    plt.plot(state)
    plt.xticks([])
    plt.yticks([])
    plt.show()


main()
