import numpy as np
class relu:
    def __init__(self):
        self.height = 0
        self.width = 0

    def forward(self,image):
        self.image = image
        self.height = image.shape[0]
        self.width = image.shape[1]
        # Get the shape of the input dataset and create an output array with the same shape
        relu_out = np.zeros((self.height, self.width))
        # For each layer in the input dataset, perform relu for each pixel of the image
        for i in np.arange(0, self.height):
           for j in np.arange(0, self.width):
                # ReLU = max(x, 0)
                relu_out[i, j] = np.max([self.image[i, j], 0])
        return relu_out
    
    def backward(self, dL_dout):
        dz = np.array(self.image)
        dz[self.image < 0] = 0
        dz[self.image >= 0]  = 1
        dz *= dL_dout
        return dz