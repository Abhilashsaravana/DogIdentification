import numpy as np

class max_pool:
    #Initialize a filter size of 2x2
    def __init__(self):
        self.size = 2
    
    #This function returns the maxpool of regions from the input image with a stride of 2
    def region(self,image):
        height = int(len(image) / self.size)
        width = int(len(image[0]) / self.size)
        
        self.image = image
        
        for i in range(height):
            for j in range(width):
                patch = image[(i*self.size) : (i*self.size + self.size),(j*self.size):(j*self.size+self.size)]
                yield patch,i,j
    
    #Performs a forward pass of the maxpooling layer and returns as an output with half the initial dimensions 
    def forward(self,input):
        ht, wt = input.shape
        self.image = input
        out = np.zeros((ht//self.size, wt//self.size))
        
        for patch,i,j in self.region(input):
            out[i,j] = np.amax(patch,axis=(0,1))
            
        return out
    
    #Performs a backward pass of the maxpooling layer and returns the gradients of the loss function w.r.t the outputs of the maxpool
    #from the forward pass in the place of the max value positions
    def backward(self, input):
        
        #initialize other non max values to 0
        out = np.zeros(self.image.shape)
        
        for patch,i,j in self.region(self.image):
            ht , wt = patch.shape
            
            old_max = np.amax(patch,axis=(0,1))
            
            for i2 in range(ht):
                for j2 in range(wt):
                    if patch[i2, j2] == old_max:
                        #assigns the gradient
                        out[i * 2 + i2, j * 2 + j2] = input[i, j]
         
        return out 
    