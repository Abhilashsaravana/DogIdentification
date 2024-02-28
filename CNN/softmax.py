
import numpy as np


class softmax:
    
    def __init__(self):
        
        self.y_pred = None
        self.y_true = None

    def forward(self,input):
        exp_a = np.exp(input-np.max(input)) #find the expon
        sum_exp_a = np.sum(exp_a)
        self.y_pred = exp_a / sum_exp_a

        return self.y_pred

    def backward(self,y_true):
        self.y_true = y_true
        dinput = [a - b for a, b in zip(self.y_pred, self.y_true)]

        return dinput