import numpy as np
from SNN import *

num_neurons = 3  # 神经元数量
num_inputs_per_neuron = 5  # 每个神经元的输入数量
olfactory_net = NeuronNetwork(num_neurons, num_inputs_per_neuron)

# 模拟嗅觉刺激输入
olfactory_input = np.array([0.5, 0.3, 0.2, 0.7, 0.9])

# 激活嗅觉神经网络
activations = olfactory_net.activate_network(olfactory_input)
print("神经元激活值:", activations)
