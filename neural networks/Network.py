import numpy as np
from Neuron import Neuron

class Network:
    def __init__(self): 
        weights = np.array([0, 1])
        bias = 0
        
        self.h0 = Neuron(weights, bias)
        self.h1 = Neuron(weights, bias)
        self.o0 = Neuron(weights, bias)
        
    def feedforward(self, x) -> float:
        o0_x0 = self.h0.feedforward(x)
        o0_x1 = self.h1.feedforward(x)

        o0_x = np.array([o0_x0, o0_x1])
        
        return self.o0.feedforward(o0_x)
        
def main():
    x = np.array([2, 3])
    network = Network()
    print(network.feedforward(x))
    
if __name__ == "__main__":
    main()