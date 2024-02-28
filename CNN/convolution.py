import preprocessing
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

# 2D cross correlation layer on 1 channel
class convolution_layer:
    def __init__(self):
        self.filter = np.random.randn(5, 5)
        self.lastinput = [0]*4

    def forward(self,image,index):
        #pad the image so the output size and input size matches
        self.lastinput[index] = image
        self.image = image
        #get the shape of the image and filter
        image_shape = self.image.shape[0]
        filter_shape = self.filter.shape[0]
        #caculate the output shape, this should be equal to the original input shape
        output_shape = image_shape - filter_shape + 1
        #create a empty numpy array to store the output of cross correlation
        output = np.zeros((output_shape, output_shape))

        # When in the range of the output array's shape perform this action
        for i in range(output_shape):
            for j in range(output_shape):
            # The output pixel would be equal to the sum of certain area of the original image times the filters
            # The parts of the original image: the top left most of the parts is the output pixel's location, the shape is equal to the filter shape
                output[i,j] = np.sum(self.image[i:i+filter_shape, j:j+filter_shape] * self.filter)
        min_val = output.min()
        max_val = output.max()
        output = np.round(256 * (output - min_val) / (max_val - min_val)).astype(int)

        return output

    def backward(self, dL_dout, step_size, index):
        '''
        dL_dout: The loss gradient for the output from the previous function
        filter: 3D numpy matrix of [w, w, n], n stands for number of layers
        image: 2D numpy matrix of [i, j]
        step_size: determine the step size for updating filter weights, to be used in future model
        '''
        im_shape = self.lastinput[index].shape[0]
        filter_shape = self.filter.shape[-1]
        output_shape = im_shape - filter_shape + 1
        dL_dfilter = np.zeros(self.filter.shape)
        dL_d_input = np.zeros((im_shape, im_shape))
        for i in range(output_shape):
            for j in range(output_shape): 
                #In th range of the last input image run the following code
                dL_dfilter += dL_dout[i,j] * self.lastinput[index][i:(i+filter_shape), j:(j+filter_shape)]
                dL_d_input[i:(i+filter_shape), j:(j+filter_shape)] += dL_dout[i, j] * self.filter
        #update filter
        filter = self.filter - step_size * dL_dfilter
        self.filter = filter
        return dL_d_input
    
def conv():
    conv_filter1 = np.zeros((2,3,3))
    conv_filter1[0,:,:] = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]) #sobel_y, horizontal edges
    conv_filter1[1,:,:] = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]) #sobel_x, vertical edges

    input_image = mpimg.imread("Ragdoll.jpg")
    gray_img = preprocessing.grayscale(input_image)
    downscale_image = preprocessing.resize_img(gray_img, 256)
    conv = convolution_layer(downscale_image, conv_filter1)

    out = conv.forward()

    plt.figure(1)
    plt.axis(False)
    plt.title("First convolutional layer")
    plt.subplot(1,2,1)
    plt.imshow(out[:,:,0],cmap="gray")
    plt.axis(False)
    plt.subplot(1,2,2)    
    plt.imshow(out[:,:,1],cmap="gray")
    plt.axis(False)
    plt.show()