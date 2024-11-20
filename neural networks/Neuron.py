import numpy as np
    
class Neuron:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias
    
    def sigmoid(self, x):
        """
        Activation function
        """
        return 1/(1+np.exp(-x))

    def feedforward(self, inputs) -> float:
        total = np.dot(self.weights, inputs) + self.bias
        return self.sigmoid(total)

def main():
    weights = np.array([0, 1])
    bias = 0
    n = Neuron(weights, bias)

    x = np.array([2, 3])
    print(n.feedforward(x))
    
if __name__ == '__main__':
    main()