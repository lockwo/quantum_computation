import numpy as np

# 1D parameters

def ParameterShift(circuit, weights, s=np.pi/2):
    gradients = np.zeros_like(weights)
    weight_copy = np.copy(weights)
    for i in range(len(weights)):
        weight_copy[i] += s
        plus = circuit(weight_copy)
        weight_copy[i] -= (2 * s)
        minus = circuit(weight_copy)
        gradients[i] = (plus - minus)/(2 * np.sin(s)) 
    return gradients


