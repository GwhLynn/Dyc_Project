class Neuron:
    # -------------------------------------------------------
    def __init__(self, current_status):
        # -----------------------------------------
        self.OutsideTConstant = 5  # 膜外时间常数
        self.outside_Potential = current_status  #
        self.state = 0

    # -------------------------------------
    def sensing(self, stimulus):
        stim_amp = (stimulus + self.outside_Potential)  # 输入信号和膜外电势值的和
        # OutsideTConstant就是RC时间常数，U/R就是I，I/C代表的是电压瞬间变化率。
        self.outside_Potential -= stim_amp / self.OutsideTConstant
        # if self.outside_Potential >
        return stim_amp, -self.outside_Potential


class NeuronNetwork:
    def __int__(self, num_neurons, num_inputs_per_neuron):
        self.neurons = []
        for _ in range(num_neurons):
            neuron = NeuronNetwork(num_inputs_per_neuron)
            self.neurons.append(neuron)

    def activate_Network(self, stimulus):
        activations = []
        for neuron in self.neurons:
            activation = neuron.neurons
            activations.append(activation)
        return activations
