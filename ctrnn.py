import random
import math
import itertools
from GeneticAlgorithm import *

#A few helper functions representing a common activation function and its derivative
def sigmoid(x):
    return 1/(1 + math.exp(-x))

def sigmoid_derivative(s):
    return s * (1.0 - s)

"""#Takes the rate of 'loss' of a line, it's current value, and its input and finds an approx of the slope
def leaky_integrator(A, x, C):
    return (-A) * x + C

#Returns an approximation of the value of a lone after time "time_step"
def forward_euler_method(time_step, slope, value):
    return value + time_step * slope"""

class ContinuousNeuron(object):
    def __init__(self, weights=[], bias=0, tau=1, index=0):
        self.weights = weights
        self.bias = bias
        self.tau = tau
        self.input = 0
        self.state = 0
        self.output = 0.0
        self.index = index
        
    @property
    def tau(self):
        return self.__tau
    
    @tau.setter
    def tau(self, x):
        self.__tau = x
        self.leak_rate = 1 / self.__tau
        
    def get_output(self, neuron_outputs, step_size):
        #input from previous neurons
        input_sum = self.input
        for i in range(len(self.weights)):
            input_sum += neuron_outputs[i] * self.weights[i]
        ###+ sum([neuron_outputs[i] * self.weights[i] for i in range(len(self.weights))])
        
        x = step_size * self.leak_rate
        self.state += x * (input_sum - self.state)
        self.output = sigmoid(self.state + self.bias)
        return self.output

class FullyConnectedCTRNN(object):
    def __init__(self, weights, biases, taus):
        self.neurons = [ContinuousNeuron(weights[i], biases[i], taus[i], i) for i in range(len(weights))]
        
    def euler_step(self, inputs, time_step):
        if inputs != None:
            for i in range(len(inputs)):
                self.neurons[i].input = inputs[i]
            
        #get output of all neurons for previous time step
        #this even contains the output of the neuron that it's being fed into
        prev_outputs = [neuron.output for neuron in self.neurons]
        #use previous layer outputs to compute the current layers
        outputs = [neuron.get_output(prev_outputs, time_step) for neuron in self.neurons]
        return outputs
    
    """def randomize_state(self, minimum, maximum):
        #Set the state of neuron
        for neuron in self.neurons:
            neuron.state = random.uniform(minimum, maximum) 
            neuron.output = (neuron.state + neuron.bias)"""
            
    @staticmethod
    def create_network(genome):
        # 1 element -> neuron_count or N
        # N elements -> neuron biases
        # N elements -> neuron taus
        # N^2 elements -> neuron weights
        neuron_count = genome[0]
        
        biases_offset = 1
        neuron_biases = genome[biases_offset: biases_offset + neuron_count]
        
        tau_offset = biases_offset + neuron_count
        neuron_taus = genome[tau_offset: tau_offset + neuron_count]
        
        weights_offset = tau_offset + neuron_count
        neuron_weights = []
        for i in range(neuron_count):
            weight_start = weights_offset + (i * neuron_count)
            weights = genome[weight_start: weight_start + neuron_count]
            #for j in range(neuron_count):
            #    k = i * neuron_count + j
            #    weights.append(genome[weights_offset + k])
            neuron_weights.append(weights)
        
        #Build the network then setup the neuron params
        network = FullyConnectedCTRNN(neuron_weights, neuron_biases, neuron_taus)
        return network