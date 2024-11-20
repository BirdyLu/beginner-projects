from Neuron import Neuron
from Network import Network
import numpy as np

def mse_loss(y_true, y_pred):
    return ((y_true - y_pred) ** 2).mean()

def main():
    y_true = np.array([1, 0, 0, 1])
    y_pred = np.array([0, 0, 0, 0])
    print(mse_loss(y_true, y_pred))
    
if __name__ == "__main__":
    main()