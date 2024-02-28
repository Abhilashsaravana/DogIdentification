import numpy as np

class fully_connected_layer:
    def __init__(self, outputsize, inputsize): # constructor takes in the desired input and output sizes
        self.outputSize = outputsize # set the class outputSize variable to the desired output size
        self.weights = np.random.randn(inputsize, outputsize) # initialize the weights to random values
        self.orgweights = self.weights # keep track of the original randomized weights
        self.biases = np.zeros(outputsize) # intialize all biases to 0

    # This function performs a forward propagation pass of this fully connected layer
    # input is a list
    def forward(self, input): 
        self.input = np.array(input) # turn the input list into a numpy array
        output = np.dot(self.input, self.weights) + self.biases # the output is the dotproduct of the input and the weights, plus the biases at the end
        self.lastOutput = output/self.outputSize # record the last output calculated
        return self.lastOutput # return the calulated output
    
    # This function performs a backward propagation pass of this fully connected layer.
    # learningRate is a scalar that controls how much the weights are adjusted in the gradient direction
    # passedDownPartialDerivatives is a list of partial derivatives of the cost with respect to the output nodes. In our model, these derivatives should come from the softmax layer
    def backward(self, learningRate, passedDownPartialDerivatives): 
        passedDownPartialDerivatives = np.array(passedDownPartialDerivatives)  # convert to numpy array
        grad_weights = np.outer(self.input, passedDownPartialDerivatives)  # outer product to get gradient of weights
        grad_bias = passedDownPartialDerivatives  # gradient for bias
        self.weights = self.weights - learningRate * grad_weights # update the weights
        self.biases = self.biases - learningRate * grad_bias  # update biases

        self.out = np.dot(passedDownPartialDerivatives, self.weights.T)  # Calculate gradients for the previous layer
        return self.out # return the partial derivatives of the cost with respect to the input nodes for the previous layer
