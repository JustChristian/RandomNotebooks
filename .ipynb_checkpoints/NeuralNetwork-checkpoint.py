import random
import math

#A few helper functions representing a common activation function and its derivative
def sigmoid(x):
    return 1/(1 + math.exp(-x))

def sigmoid_derivative(s):
    return s * (1.0 - s)

def squared_error(output, target):
    return (output - target) ** 2

def squared_error_derivative(output, target):
    return 2 * (output - target)

#Neuron and Neuron layer representations
class Neuron(object):
    def __init__(self, input_count):
        self.bias = random.random()
        self.weights = [random.random() for i in range(input_count)]
        self.input_count = input_count
    
    #Takes the weighted sum of inputs then applies a activation function 
    #to narmalize the sum
    def getOutput(self, inputs):
        ###assert(len(inputs) == self.input_count)
        weighted_input_sum = self.bias + sum([(self.weights[i] * inputs[i]) for i in range(len(self.weights))])
        self.output = sigmoid(weighted_input_sum)
        return self.output
    
    def computeOutputError(self, target):
        self.error = squared_error(self.output, target)
        return self.error
    
class NeuronLayer(object):
    def __init__(self, input_count, neuron_count):
        self.neurons = [Neuron(input_count) for i in range(neuron_count)]

    def getOutput(self, inputs):
        return [neuron.getOutput(inputs) for neuron in self.neurons]
    
    def computeOutputError(self, target):
        return [self.neurons[i].computeOutputError(target[i]) for i in range(len(self.neurons))]
    
#The actual neural network class
class NeuralNetwork(object):
    def __init__(self, n_inputs, n_hiddens, n_outputs, n_hidden_layers=1, learning_rate=0.2):
        self.learning_rate = learning_rate
        
        #Build network
        self.layers = []
        self.layers.append(NeuronLayer(n_inputs, n_hiddens))
        for i in range(n_hidden_layers - 1):
            self.layers.append(NeuronLayer(n_hiddens, n_hiddens))
        self.layers.append(NeuronLayer(n_hiddens, n_outputs))
        
    #Use the output of each layer as the input of the next, 
    #and the final output is the output of the system
    def predict(self, inputs):
        layer_output = inputs
        for layer in self.layers:
            layer_output = layer.getOutput(layer_output)
        return layer_output
     
    def computeError(self, target):
        return self.layers[-1].computeOutputError(target)
        
    def back_propagate(self, target):
        #1. Compute errors and deltas for the output layer
        output_layer = self.layers[-1]
        for i in range(len(output_layer.neurons)):
            neuron = output_layer.neurons[i]
            neuron.error = squared_error_derivative(neuron.output, target[i])
            neuron.delta = neuron.error * sigmoid_derivative(neuron.output)

        #2. Apply derivative of sigmoid to next layer from each layer (output layer error in the case of last hidden layer)  ... dot-product of weights of each neuron by delta of outputs???
        for i in reversed(range(len(self.layers) - 1)):
            layer = self.layers[i]
            next_layer = self.layers[i+1]
            
            for neuron_i in range(len(layer.neurons)):
                neuron = layer.neurons[neuron_i]
                neuron.error = 0.0
                for next_neuron_i in range(len(next_layer.neurons)):
                    next_neuron = next_layer.neurons[next_neuron_i]
                    neuron.error += next_neuron.delta * next_neuron.weights[neuron_i]
                neuron.delta = neuron.error * sigmoid_derivative(neuron.output)

        
    def update_weights(self, inputs):
        for i in range(len(self.layers)):
            curr_layer = self.layers[i]
            
            #outputs of previous layer are the input of this layer
            if i != 0:
                inputs = [neuron.output for neuron in self.layers[i-1].neurons]
                
            for neuron in curr_layer.neurons:
                for j in range(len(inputs)):
                    neuron.weights[j] -= self.learning_rate * neuron.delta * inputs[j]
                neuron.bias -= self.learning_rate * neuron.delta
                
        
    
    def train(self, rows, n_epoch=1, print_epoch=False):
        for epoch in range(n_epoch):
            sum_error = 0.0
            
            for row in rows:
                inputs = row[:-1]
                target = row[-1]
                
                output = self.predict(inputs)
                sum_error += abs(sum(self.computeError(target)))
                self.back_propagate(target)
                self.update_weights(inputs) #Check that this should be inputs passed in
            
            if print_epoch:
                print(">epoch %d: %.3f" % (epoch, sum_error))