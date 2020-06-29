'''
Created on Jun 28, 2020

@author: Stephen
'''
import numpy as np
np.random.seed(0)


def spiral_data(points, classes):
    X = np.zeros((points*classes, 2))
    y = np.zeros(points*classes, dtype='uint8')
    for class_number in range(classes):
        ix = range(points*class_number, points*(class_number+1))
        r = np.linspace(0.0, 1, points)
        t = np.linspace(class_number*4, (class_number+1)*4, points) + np.random.randn(points)*0.2
        X[ix] = np.c_[r*np.sin(t*2), r*np.cos(t*2)]
        y[ix] = class_number
    return X, y

def spiral_data_with_cloudinessi(points, classes, cloudiness=2.5):
    X = np.zeros((points*classes, 2))
    y = np.zeros(points*classes, dtype='uint8')
    for class_number in range(classes):
        ix = range(points*class_number, points*(class_number+1))
        r = np.linspace(0.0, 1, points)
        t = np.linspace(class_number*4, (class_number+1)*4, points) + np.random.randn(points)*0.2
        X[ix] = np.c_[r*np.sin(t*cloudiness), r*np.cos(t*cloudiness)]
        y[ix] = class_number
    return X, y

def spiral_data_with_vcloudiness(points, classes, cloudiness = [2.5,2.5]):
    X = np.zeros((points*classes, 2))
    y = np.zeros(points*classes, dtype='uint8')
    for class_number in range(classes):
        ix = range(points*class_number, points*(class_number+1))
        r = np.linspace(0.0, 1, points)
        t = np.linspace(class_number*4, (class_number+1)*4, points) + np.random.randn(points)*0.2
        X[ix] = np.c_[r*np.sin(t*cloudiness[0]), r*np.cos(t*cloudiness[1])]
        y[ix] = class_number
    return X, y


class Layer_Dense:
    def __init__(self, n_inputs, n_nuerons):
        self.weights = 0.9*np.random.randn(n_inputs, n_nuerons)
        self.biases = np.zeros((1, n_nuerons))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

class Layer_Hysteresis:
    """
        output function:
        X(t) = X0*sin(w*t)
        Y(t) = Y0*sin(w*t - p)
    
        Y(t) = chi(i)*X(t) + integ( psi(tau)*C(t-tau) d(tau))
        chi(i) is the instantaneous response and psi(tau)  is the impulse response to an impulse that occurred tau time units in the past.

    """
    def __init__(self, n_inputs, n_nuerons):
        self.weights = 0.10*np.random.randn(n_inputs, n_nuerons)
        self.weightsX = 0.10*np.random.randn(n_inputs, n_nuerons)
        self.biases = np.zeros((1, n_nuerons))
        self.previous_weights = np.asarray([np.zeros((n_inputs, n_nuerons))])
    def X(self, t):
        t = np.asarray(t)
        #print(self.weightsX.shape)
        #print(t.shape)
        #print(np.dot(self.weightsX.T,t.T).shape)
        return np.dot(self.weightsX,np.sin(np.dot(self.weights.T,t.T)))
    def X2(self, t):
        sin = np.sin(np.dot(self.weights,t.T))
        return np.dot(self.weightsX.T,sin)
    def forward(self, inputs):
        #print(inputs.shape)
        #print(self.previous_weights.shape)
        #print(np.sum(self.previous_weights, 0).shape)
        X = self.X(inputs)
        X2 = self.X2(np.dot(inputs,np.sum(self.previous_weights, 0))) 
        right = .99*X
        left = np.exp(-len(self.previous_weights))*X2/len(self.previous_weights)    
        self.output =  np.dot(right,left.T)
        

class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

class Activation_SoftMax:
    def forward(self, inputs):
        expo = np.exp(inputs)
        expo_sum = np.sum(expo)
        self.output = expo/expo_sum
        
class Activation_UnitStep:
    def forward(self, inputs):
        self.output = np.heaviside(inputs, 0.0)

class Activation_IntegerPart:
    def forward(self, inputs):
        self.output = np.ceil(inputs)

class Activation_HardMax:
    def forward(self, inputs):
        self.output = inputs/np.amax(inputs)

def softmax(inputs):
    expo = np.exp(inputs)
    expo_sum = np.sum(expo)
    return expo/expo_sum

def binary_cross_entropy(actual, predicted):
    sum_score = 0.0
    for i in range(len(actual)):
        sum_score += actual[i] * np.log(1e-15 + predicted[i])
    mean_sum_score = 1.0 / len(actual) * sum_score
    return -mean_sum_score

def cross_entropy(actual, predicted):
    return -actual * np.log(np.add(1e-15,predicted))

def RMSerror(actual, predicted):
    return np.sqrt(1/2 * (np.power(actual,2) + np.power(predicted,2)))

def MinusRMSerror(actual, predicted):
    return np.sqrt(1/2 * (np.abs(np.power(actual,2) - np.power(predicted,2))))

def HarmonicMean(actual, predicted):
    return 1/(np.divide(1,actual) + np.divide(1, predicted))
