'''
Created on Jun 29, 2020

@author: Stephen
'''

import numpy as np
import matplotlib.pyplot as plt
from functionLibrary import *

class Network:
    def __init__(self): 
        '''
        Initializes a new copy of the network 
        '''
        self.layer1 = Layer_Dense(2, 100)
        self.layer2 = Layer_Dense(100,100)
        self.layer3 = Layer_Dense(100, 100)
        self.layer4 = Layer_Dense(100, 10)
        self.layer5 = Layer_Dense(10,3)
        
        self.layerA = Activation_ReLU()
        self.layerA2 = Activation_ReLU()
        self.layerA3 = Activation_IntegerPart()
        self.layerA4 = Activation_ReLU()
        self.layerA5 = Activation_SoftMax()
        
        self.Optimalw1 = [self.layer1.weights]
        self.Optimalw2 = [self.layer2.weights]
        self.Optimalw3 = [self.layer3.weights]
        
        self.Optimalb1 = [self.layer1.biases]
        self.Optimalb2 = [self.layer2.biases]
        self.Optimalb3 = [self.layer3.biases]
        
        self.layers = [self.layer1, self.layer2, self.layer3, self.layer4, self.layer5]
        self.activations = [self.layerA, self.layerA2, self.layerA3, self.layerA4, self.layerA5]
        
        self.errorscore = 1e100
        
        self.mutation_chance = 25
        
    def forward(self, X):
        '''
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
        '''
        for i in range(len(self.layers)):
            if i == 0: 
                self.layers[0].forward(X)
            else:
                self.layers[i].forward(self.activations[i-1].output)
            self.activations[i].forward(self.layers[i].output)
        #Make all in forward above into a for loop based on self.layers.
        
        self.output = self.activations[-1].output
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
    
    def updateWeightsCross(self, network2):
        crossedGencode = []
        for i in range(len(self.layers)):
            w = np.copy(self.layers[i].genetic_code)
            w2 = np.copy(network2.layers[i].genetic_code)
            cross_over_point = np.random.randint(len(w))
            for i in range(cross_over_point):
                temp = w[i]
                w[i] = w2[i]
                w2[i] = temp
            w = self.mutate(w)
            crossedGencode.append(w)
        newNetwork = Network()
        for i in range(len(newNetwork.layers)):
            newNetwork.layers[i].makeWeights(crossedGencode[i])
        return newNetwork
    
    def mutate(self, genecode):
        if np.random.randint(100) < self.mutation_chance:
            for i in range(np.random.randint(len(genecode))):
                n = np.random.randint(len(genecode))
                genecode[n] = 1.0*np.random.normal(0,10.0)
        return genecode

number_of_generations = 500
number_of_individuals = 1000


individuals = []

data = []
goodclasses = []
for i in range(number_of_generations):
    inpu, actual = spiral_data_with_cloudinessi(90, 3, 1)
    data.append(inpu)
    goodclasses.append(actual)
#print(data)

best_individual = Network()
best_individual2 = Network()
best_individual.forward(data[0])
best_individual2.forward(data[0])
best_individual.Error(goodclasses[0])
best_individual2.Error(goodclasses[0])

for i in range(number_of_generations):
    best_individual_of_generation = Network()
    best_2_of_generation = Network()
    best_individual_of_generation.forward(data[0])
    best_2_of_generation.forward(data[0])
    best_individual_of_generation.Error(goodclasses[0])
    best_2_of_generation.Error(goodclasses[0])

    for j in range(number_of_individuals):
        child = best_individual.updateWeightsCross(best_individual2)
        individuals.append(child)
        individuals[j].forward(data[i])
        individuals[j].Error(goodclasses[i])
        if individuals[j].errorscore <= best_individual_of_generation.errorscore:
            best_2_of_generation = best_individual_of_generation
            best_individual_of_generation = individuals[j]
        #plt.scatter(j, best_individual_of_generation.errorscore)
        #plt.scatter(j, best_2_of_generation.errorscore)
    #plt.show()
    if best_individual_of_generation.errorscore <= best_individual.errorscore:
        best_individual2 = best_individual
        best_individual = best_individual_of_generation
    if best_2_of_generation.errorscore <= best_individual2.errorscore:
        best_individual2 = best_2_of_generation
    individuals = []           
    plt.scatter(i, best_individual.errorscore)
    print("Generation", i, "Done!")
plt.show()
data = np.asarray(data)
goodclasses = np.asarray(goodclasses)
#print(data.shape)
best_individual.forward(data[-1])
best_individual.Error(goodclasses[-1])
#plt.scatter(data[-1,:,0], data[-1,:,1], c = goodclasses[-1, :], cmap = "brg")
#plt.show()
#print(best_individual.classes)
plt.scatter(data[-1,:,0], data[-1,:,1], c = best_individual.classes, cmap = "brg")
plt.show()