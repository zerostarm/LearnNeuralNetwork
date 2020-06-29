'''
Created on Jun 28, 2020

@author: Stephen
'''

import numpy as np
import matplotlib.pyplot as plt
from functionLibrary import *

class Network:
    def __init__(self):
        
        self.layer1 = Layer_Dense(2, 100)
        self.layer2 = Layer_Dense(100,10)
        self.layer3 = Layer_Dense(10, 3)
        
        self.layerA = Activation_ReLU()
        self.layerA2 = Activation_ReLU()
        self.layerA3 = Activation_HardMax()
    
    def forward(self, X):
        self.layers = [self.layer1, self.layer2, self.layer3]
        #print(X.shape)
        self.layer1.forward(X)
        #print(layer1.output.shape)
        self.layerA.forward(self.layer1.output)
        #print(layerA.output.shape)
        self.layer2.forward(self.layerA.output)
        self.layerA2.forward(self.layer2.output)
        #print(layerA2.output.shape)
        self.layer3.forward(self.layerA2.output)
        #print(layer3.output)
        self.layerA3.forward(self.layer3.output)
        
        self.output = self.layerA3.output
        self.classes = []
    
        for row in self.output:
            a = np.argmax(row)
            if a == 0:
                self.classes.append(0)
            elif a == 1:
                self.classes.append(1)
            else:
                self.classes.append(2)
    def error(self, actual):
        self.error = np.abs(actual-self.classes)
        self.errorscore = self.error/(2*len(actual))


network1 = Network()

inpu, actual = spiral_data(100, 3)
network1.forward(inpu)
network1.error(actual)

print(network1.errorscore)
#print(network1.errorscore.shape)
#print(network1.layer1.weights.shape, network1.layer1.biases.shape)
#print(network1.layer2.weights.shape, network1.layer2.biases.shape)
#print(network1.layer3.weights.shape, network1.layer3.biases.shape)

gradient = inpu.T.dot(network1.error)/ inpu.shape[0]
print(gradient)

num_iterations = 100
dw = np.zeros(num_iterations)
step_size = .9
incFactor = 1.1
step_size_max = 1.0
decFactor = 0.9
step_size_min = 0.0
for i in range(num_iterations):
    dw[i] = inpu.T.dot(network1.error)/ inpu.shape[0]
    
    if dw[i] * dw[i - 1] > 0:
        step_size = min(step_size * incFactor, step_size_max)
    elif dw[i] * dw[i - 1] < 0:
        step_size = max(step_size * decFactor, step_size_min)
    
    w[i] = w[i - 1] - np.sign(dw[i]) * step_size