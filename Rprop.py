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
        self.layerA3 = Activation_Partial()
        
        self.Optimalw1 = [self.layer1.weights]
        self.Optimalw2 = [self.layer2.weights]
        self.Optimalw3 = [self.layer3.weights]
        
        self.Optimalb1 = [self.layer1.biases]
        self.Optimalb2 = [self.layer2.biases]
        self.Optimalb3 = [self.layer3.biases]
        
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
    def Error(self, actual):
        self.error = np.abs(actual-self.classes)
        self.trueError = actual-self.classes
        self.errorscore = np.sum(np.abs(self.trueError))/(5/3*len(actual))
        '''
        out = 1000*np.ones(actual.shape)
        np.divide(5/3*len(actual), self.trueError, out = out, where=self.trueError!=0.0) 
        print(out)
        self.error = out'''
    
    def updateWeights(self, dw, step_size):
        self.layer1.weights = self.layer1.weights - np.sign(dw) * step_size
        self.layer2.weights = self.layer2.weights - np.sign(dw) * step_size
        self.layer3.weights = self.layer3.weights - np.sign(dw) * step_size
    def updateBiases(self, dt, step_size_2):
        self.layer1.biases = self.layer1.biases - np.sign(dt) * step_size_2
        self.layer2.biases = self.layer2.biases - np.sign(dt) * step_size_2
        self.layer3.biases = self.layer3.biases - np.sign(dt) * step_size_2
    
    def updateWeightsRandom(self):
        self.layer1.weights = 1.0*np.random.normal(0,.5,(2, 100))
        self.layer2.weights = 1.0*np.random.normal(0,.5,(100,10))
        self.layer3.weights = 1.0*np.random.normal(0,.5,(10, 3))
    def updateBiasesRandom(self):
        self.layer1.biases = 1.0*np.random.normal(0,.5,(1, 100))
        self.layer2.biases = 1.0*np.random.normal(0,.5,(1,10))
        self.layer3.biases = 1.0*np.random.normal(0,.5,(1, 3))
    
    def updateWeightsCross(self, weights1, weights2, weights3):
        w1 = self.layer1.weights
        w2 = self.layer2.weights
        w3 = self.layer3.weights
        
        size = w1.shape[0]*w1.shape[1]
        print("size is ", size)
        print(int(size/2))
        n = np.random.randint(1, high=int(size/2))
        g = gencoordinates(0, w1.shape[0], 0, w1.shape[1])
        positions = []
        print("n is ", n)
        for i in range(n):
            positions.append(next(g))
        print(positions)
        
        for i in range(len(positions)):
            x = positions[i,0]
            y = positions[i,1]
            
        
        
network1 = Network()
'''
inpu, actual = spiral_data(200, 3)
network1.forward(inpu)
network1.Error(actual)

#print(network1.errorscore)
#print(network1.errorscore.shape)
#print(network1.layer1.weights.shape, network1.layer1.biases.shape)
#print(network1.layer2.weights.shape, network1.layer2.biases.shape)
#print(network1.layer3.weights.shape, network1.layer3.biases.shape)

gradient = inpu.T.dot(network1.error)/ inpu.shape[0]
#print(gradient)
'''
num_iterations = 1
dw = []
dt = []
step_size = .5
step_size_2 = 1.2
incFactor = 1.3
step_size_max = 50.
decFactor = 0.9
step_size_min = -1.0

inpu, actual = spiral_data_with_cloudinessi(100, 3, 1)
loss = []
errorscorelist = []
calltypes = []

for i in range(num_iterations):
    inpu, actual = spiral_data_with_cloudinessi(100, 3, 1)
    network1.forward(inpu)
    network1.Error(actual)
    errorscorelist.append(network1.errorscore)
        
    #print(inpu.T.dot(network1.error)/ inpu.shape[0])
    tup = inpu.T.dot(network1.error)/inpu.shape[0]
    print(tup)
    
    '''
    if network1.errorscore <= np.amin(errorscorelist):
        #network1.updateWeightsCross(network1.Optimalw1[-1], network1.Optimalw2[-1], network1.Optimalw3[-1])
        
        network1.Optimalw1.append(network1.layer1.weights)
        network1.Optimalw2.append(network1.layer2.weights)
        network1.Optimalw3.append(network1.layer3.weights)
        
        network1.Optimalb1.append(network1.layer1.biases)
        network1.Optimalb2.append(network1.layer2.biases)
        network1.Optimalb3.append(network1.layer3.biases)
    '''
    
    dw.append(tup[0])
    dt.append(tup[1])
    
    if dw[i] * dw[i - 1] > 0:
        step_size = min(step_size * incFactor, step_size_max)
    #elif dw[i] * dw[i - 1] < 0:
    else:
        step_size = max(step_size * decFactor, step_size_min)
    
    if dt[i] * dt[i - 1]  > 0:
        step_size_2 = min(step_size_2 * incFactor, step_size_max)
    #elif dt[i] * dt[i - 1] < 0:
    else:
        step_size_2 = max(step_size_2 * decFactor, step_size_min)
    loss.append(tup)
    
    
    calls = []
    failsw = 0
    failsb = 0
    if i in [0,1,2,3,4]:
        network1.updateWeights(dw[i], step_size)
        network1.updateBiases(dt[i], step_size_2)
        calls.append(0)
        calls.append(0)
    else:
        if np.isclose(dw[i], dw[i-2], rtol= 1e-3) or np.isclose(dw[i], dw[i-3], rtol= 1e-3) or np.isclose(dw[i], dw[i-4], rtol= 1e-3): #add more cases for this and other one
            failsw +=1
        else:
            failsw = failsw
        if failsw == 2:
            network1.updateWeightsRandom()
            calls.append(1)
            failsw = 0
        else: 
            network1.updateWeights(dw[i], step_size)
            calls.append(0)
        
        if np.isclose(dt[i], dt[i-2], rtol= 1e-2) or np.isclose(dt[i], dt[i-3], rtol= 1e-2) or np.isclose(dt[i], dt[i-4], rtol= 1e-2):
            failsb += 1
        if failsb == 2:
            network1.updateBiasesRandom()
            calls.append(2)
            failsb = 0
        else:
            network1.updateBiases(dt[i], step_size_2)
            calls.append(0)
    calltypes.append(np.asarray(calls))
    
    
plt.scatter(inpu[:,0], inpu[:,1], c = network1.classes, cmap = "brg")
plt.show()

loss = np.asarray(loss)
calltypes = np.asarray(calltypes)
#print(calltypes)
print(calltypes.shape)
plt.plot(range(num_iterations), loss[:,0], label="Loss 1")
plt.plot(range(num_iterations), loss[:,1], label="Loss 2")
plt.plot(range(num_iterations), errorscorelist, label="Error Score")
#plt.scatter(range(num_iterations), calltypes, label="Call types")
#print(errorscorelist)
plt.legend()
plt.show()