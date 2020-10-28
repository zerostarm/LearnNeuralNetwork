'''
Created on Jun 28, 2020

@author: Stephen
'''


import numpy as np
import matplotlib.pyplot as plt

from functionLibrary import *
'''
def __init__(self, n_inputs, n_nuerons):
        self.genetic_code = 1.0*np.random.normal(0,.5, (1,n_inputs*n_nuerons + n_nuerons))
        temp = np.reshape(self.genetic_code, (n_inputs + 1, n_nuerons))
        self.weights = temp[0:2,:]
        print(self.weights.shape)
        self.biases = np.zeros((1, n_nuerons))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases
'''


np.random.seed(0)

layer1 = Layer_Dense(2, 100)
layer2 = Layer_Dense(100,10)
layer3 = Layer_Dense(10, 3)

layerA = Activation_ReLU()
layerA2 = Activation_ReLU()
layerA3 = Activation_SoftMax()

'''
X, y = spiral_data(100, 3)
#print(y)
#print(X)
#plt.scatter(X[:,0], X[:,1], c=y,  cmap = "brg")
#plt.show()
'''

learning_rate = 0.001

Xlist = []
ylist = []
clist = []
errorlist = [100]

cloudiness = .1 #np.linspace(1, 2.5, 100)

for i in range(1000):
    
    X, y = spiral_data_with_cloudinessi(100, 3, cloudiness)
    Xlist.append(X)
    ylist.append(y)
    
    layers = [layer1, layer2, layer3]
    
    #print(X.shape)
    layer1.forward(X)
    #print(layer1.output.shape)
    layerA.forward(layer1.output)
    #print(layerA.output.shape)
    layer2.forward(layerA.output)
    layerA2.forward(layer2.output)
    #print(layerA2.output.shape)
    layer3.forward(layerA2.output)
    #print(layer3.output)
    layerA3.forward(layer3.output)
    
    output = layerA3.output
    #print(output)
    
    classes = []
    
    for row in output:
        a = np.argmax(row)
        if a == 0:
            classes.append(0)
        elif a == 1:
            classes.append(1)
        else:
            classes.append(2)
    clist.append(classes)
    #print(classes)
    loss = binary_cross_entropy(y, classes)
    #print(loss)
    
    error = RMSerror(y, classes)#np.abs(y - classes)
    #print(error)
    gradient = X.T.dot(error)/ X.shape[0]
    #print(gradient.shape)
    
    errorscore = np.sum(error)/(2*len(error)) 
    print(errorscore)
    errorlist.append(errorscore)
    
    if errorscore <= np.amin(errorlist):
        goodlayerslist = [layer1.weights, layer1.biases,
                          layer2.weights, layer2.biases,
                          layer3.weights, layer3.biases]
    
       
    for layer in layers:
        #print(layer.weights)
        for i in range(len(layer.biases)):
            layer.weights[i] += -learning_rate * gradient[1]#**layer.weights[i]
            layer.biases[i] += -learning_rate * gradient[0]#**layer.weights[i]
    learning_rate *= 0.9

Xlist = np.asarray(Xlist)
ylist = np.asarray(ylist)
clist = np.asarray(clist)

plt.scatter(Xlist[-1,:,0], Xlist[-1,:,1], c=clist[-1],  cmap = "brg")
plt.show()
'''
for i in range(len(ylist)):
    plt.scatter(Xlist[i,:,0], Xlist[i,:,1], c=ylist[i],  cmap = "brg")
plt.show()'''