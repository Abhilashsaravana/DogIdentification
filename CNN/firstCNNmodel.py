import matplotlib
from matplotlib.image import imread
import matplotlib.pyplot as plt
import numpy as np
import math
import preprocessing
import convolution
import relu
import maxpool
import fullyConnectedLayer
import softmax
import crossEntropy
import os



# FORWARD PROPOGATION
def initialize():
    global conv1_filter1
    global conv1_filter2
    global conv1_filter3
    global conv1_filter4
    global relu1_1
    global relu1_2
    global relu1_3
    global relu1_4
    global maxpool1_1
    global maxpool1_2
    global maxpool1_3 
    global maxpool1_4
    # Layer 2
    global conv2_filter1
    global conv2_filter2
    global conv2_filter3
    global conv2_filter4
    global relu2_1 
    global relu2_2 
    global relu2_3 
    global relu2_4 
    global maxpool2_1
    global maxpool2_2
    global maxpool2_3 
    global maxpool2_4 
    # FCL
    global fcl_layer
    # Softmax
    global soft_max


    conv1_filter1 = convolution.convolution_layer()
    conv1_filter2 = convolution.convolution_layer()
    conv1_filter3 = convolution.convolution_layer()
    conv1_filter4 = convolution.convolution_layer()

    relu1_1 = relu.relu()
    relu1_2 = relu.relu()
    relu1_3 = relu.relu()
    relu1_4 = relu.relu()

    maxpool1_1 = maxpool.max_pool()
    maxpool1_2 = maxpool.max_pool()
    maxpool1_3 = maxpool.max_pool()
    maxpool1_4 = maxpool.max_pool()
    
    conv2_filter1 = convolution.convolution_layer()
    conv2_filter2 = convolution.convolution_layer()
    conv2_filter3 = convolution.convolution_layer()
    conv2_filter4 = convolution.convolution_layer()
    

    #   Relu 2
    relu2_1 = relu.relu()
    relu2_2 = relu.relu()
    relu2_3 = relu.relu()
    relu2_4 = relu.relu()
    #   Maxpool 2
    maxpool2_1 = maxpool.max_pool()
    maxpool2_2 = maxpool.max_pool()
    maxpool2_3 = maxpool.max_pool()
    maxpool2_4 = maxpool.max_pool()
    
    #   fullyconnectedLayer
    #   Column concadenation
    
    output_size = 3
    input_size = 676
    fcl_layer = fullyConnectedLayer.fully_connected_layer(output_size, input_size)

    #   Softmax
    soft_max = softmax.softmax()
    


    
def forward(img, y_true):
    # global Variable declaration
    # Layer 1
    global conv1_filter1
    global conv1_filter2
    global conv1_filter3
    global conv1_filter4
    global relu1_1
    global relu1_2
    global relu1_3
    global relu1_4
    global maxpool1_1
    global maxpool1_2
    global maxpool1_3 
    global maxpool1_4
    # Layer 2
    global conv2_out1_filter1
    global conv2_out1_filter2
    global conv2_out1_filter3
    global conv2_out1_filter4
    global conv2_out2_filter1
    global conv2_out2_filter2
    global conv2_out2_filter3
    global conv2_out2_filter4
    global conv2_out3_filter1
    global conv2_out3_filter2
    global conv2_out3_filter3
    global conv2_out3_filter4
    global conv2_out4_filter1
    global conv2_out4_filter2
    global conv2_out4_filter3
    global conv2_out4_filter4
    global relu2_1 
    global relu2_2 
    global relu2_3 
    global relu2_4 
    global maxpool2_1
    global maxpool2_2
    global maxpool2_3 
    global maxpool2_4 
    # FCL
    global fcl_layer
    # Softmax
    global soft_max
    
    input = plt.imread(img)

    # Preprocess
    grayscaled_img = preprocessing.grayscale(input)
    resized_img = preprocessing.resize_img(grayscaled_img,64)
    
    # First Layer
    # Conv 1


    conv1_out1 = conv1_filter1.forward(resized_img,0)
    conv1_out2 = conv1_filter2.forward(resized_img,0)
    conv1_out3 = conv1_filter3.forward(resized_img,0)
    conv1_out4 = conv1_filter4.forward(resized_img,0)

    #  Relu 1
 
    relu1_out1 = relu1_1.forward(conv1_out1)
    relu1_out2 = relu1_2.forward(conv1_out2)
    relu1_out3 = relu1_3.forward(conv1_out3)
    relu1_out4 = relu1_4.forward(conv1_out4)
    

    # Maxpool 1

    maxpool1_out1 = maxpool1_1.forward(relu1_out1)
    maxpool1_out2 = maxpool1_2.forward(relu1_out2)
    maxpool1_out3 = maxpool1_3.forward(relu1_out3)
    maxpool1_out4 = maxpool1_4.forward(relu1_out4)
    
   
    #   Second Layer
    #   Conv 2
    # input image: 1st maxpool output from convolutional layer 1

    conv2_out1_1 = conv2_filter1.forward(maxpool1_out1,0)
    conv2_out1_2 = conv2_filter1.forward(maxpool1_out2,1)
    conv2_out1_3 = conv2_filter1.forward(maxpool1_out3,2)
    conv2_out1_4 = conv2_filter1.forward(maxpool1_out4,3)
    # input image: 2nd maxpool output from convolutional layer 1

    conv2_out2_1 = conv2_filter2.forward(maxpool1_out1,0)
    conv2_out2_2 = conv2_filter2.forward(maxpool1_out2,1)
    conv2_out2_3 = conv2_filter2.forward(maxpool1_out3,2)
    conv2_out2_4 = conv2_filter2.forward(maxpool1_out4,3)
    # input image: 3rd maxpool output from convolutional layer 1

    conv2_out3_1 = conv2_filter3.forward(maxpool1_out1,0)
    conv2_out3_2 = conv2_filter3.forward(maxpool1_out2,1)
    conv2_out3_3 = conv2_filter3.forward(maxpool1_out3,2)
    conv2_out3_4 = conv2_filter3.forward(maxpool1_out4,3)
    # input image: 4th maxpool output from convolutional layer 1

    conv2_out4_1 = conv2_filter4.forward(maxpool1_out1,0)
    conv2_out4_2 = conv2_filter4.forward(maxpool1_out2,1)
    conv2_out4_3 = conv2_filter4.forward(maxpool1_out3,2)
    conv2_out4_4 = conv2_filter4.forward(maxpool1_out4,3)
    
    conv2_out1_a = np.add(conv2_out1_1, conv2_out1_2) 
    conv2_out1_b = np.add(conv2_out1_3, conv2_out1_4)
    conv2_out1 = np.add(conv2_out1_a,conv2_out1_b)
   
    conv2_out2_a = np.add(conv2_out2_1, conv2_out2_2) 
    conv2_out2_b = np.add(conv2_out2_3, conv2_out2_4)
    conv2_out2 = np.add(conv2_out2_a,conv2_out2_b)
    
    conv2_out3_a = np.add(conv2_out3_1, conv2_out3_2) 
    conv2_out3_b = np.add(conv2_out3_3, conv2_out3_4)
    conv2_out3 = np.add(conv2_out3_a, conv2_out3_b)
    
    conv2_out4_a = np.add(conv2_out4_1, conv2_out4_2) 
    conv2_out4_b = np.add(conv2_out4_3, conv2_out4_4)
    conv2_out4 = np.add(conv2_out4_a,conv2_out4_b)
    
    
    #   Relu 2
    # relu2_1 = relu.relu(conv2_out1)
    # relu2_2 = relu.relu(conv2_out2)
    # relu2_3 = relu.relu(conv2_out3)
    # relu2_4 = relu.relu(conv2_out4)

    relu2_out1 = relu2_1.forward(conv2_out1)
    relu2_out2 = relu2_2.forward(conv2_out2)
    relu2_out3 = relu2_3.forward(conv2_out3)
    relu2_out4 = relu2_4.forward(conv2_out4)
    
    # #   Maxpool 2
    # maxpool2_1 = maxpool.max_pool()
    # maxpool2_2 = maxpool.max_pool()
    # maxpool2_3 = maxpool.max_pool()
    # maxpool2_4 = maxpool.max_pool()

    maxpool2_out1 = maxpool2_1.forward(relu2_out1)
    maxpool2_out2 = maxpool2_2.forward(relu2_out2)
    maxpool2_out3 = maxpool2_3.forward(relu2_out3)
    maxpool2_out4 = maxpool2_4.forward(relu2_out4)
    
    #   fullyconnectedLayer
    #   Column concadenation
    maxpool_ls_1 = maxpool2_out1.flatten()
    maxpool_ls_1 = maxpool_ls_1.tolist()
    maxpool_ls_2 = maxpool2_out2.flatten()
    maxpool_ls_2 = maxpool_ls_2.tolist()
    maxpool_ls_3 = maxpool2_out3.flatten()
    maxpool_ls_3 = maxpool_ls_3.tolist()
    maxpool_ls_4 = maxpool2_out4.flatten()
    maxpool_ls_4 = maxpool_ls_4.tolist()

    fcl_input = maxpool_ls_1 + maxpool_ls_2 + maxpool_ls_3 + maxpool_ls_4

    fcl_output = fcl_layer.forward(fcl_input)

    #   Softmax
    # soft_max = softmax.softmax()
    soft_output = soft_max.forward(fcl_output)
    
    # Cross Entropy
    loss = crossEntropy.cross_entropy_loss(y_true,soft_output)

    print(soft_output)


    return loss, soft_output






# BACK PROPOGATION
def backward(y_true, learning_rate):
    
    #   Softmax
    softmax_back = soft_max.backward(y_true)

    #   Fully Connected Layer
    fcl_back = fcl_layer.backward(learning_rate, softmax_back)
    
    #   Maxpool 2
    maxpool2_back_1 = maxpool2_1.backward(fcl_back[0:169].reshape(13,13))
    maxpool2_back_2 = maxpool2_1.backward(fcl_back[169:338].reshape(13,13))
    maxpool2_back_3 = maxpool2_1.backward(fcl_back[338:507].reshape(13,13))
    maxpool2_back_4 = maxpool2_1.backward(fcl_back[507:676].reshape(13,13))

    
    #   Relu 2
    relu2_back1 = relu2_1.backward(maxpool2_back_1)
    relu2_back2 = relu2_2.backward(maxpool2_back_2)
    relu2_back3 = relu2_3.backward(maxpool2_back_3)
    relu2_back4 = relu2_4.backward(maxpool2_back_4)

    #   Conv 2
    conv2_out1_back1 = conv2_filter1.backward(relu2_back1,learning_rate,0)
    conv2_out1_back2 = conv2_filter1.backward(relu2_back1,learning_rate,1)
    conv2_out1_back3 = conv2_filter1.backward(relu2_back1,learning_rate,2)
    conv2_out1_back4 = conv2_filter1.backward(relu2_back1,learning_rate,3)
    conv2_out1_back = conv2_out1_back1 + conv2_out1_back2 + conv2_out1_back3 + conv2_out1_back4
    
    conv2_out2_back1 = conv2_filter2.backward(relu2_back2,learning_rate,0)
    conv2_out2_back2 = conv2_filter2.backward(relu2_back2,learning_rate,1)
    conv2_out2_back3 = conv2_filter2.backward(relu2_back2,learning_rate,2)
    conv2_out2_back4 = conv2_filter2.backward(relu2_back2,learning_rate,3)
    conv2_out2_back = conv2_out2_back1 + conv2_out2_back2 + conv2_out2_back3 + conv2_out2_back4
    
    conv2_out3_back1 = conv2_filter3.backward(relu2_back3,learning_rate,0)
    conv2_out3_back2 = conv2_filter3.backward(relu2_back3,learning_rate,1)
    conv2_out3_back3 = conv2_filter3.backward(relu2_back3,learning_rate,2)
    conv2_out3_back4 = conv2_filter3.backward(relu2_back3,learning_rate,3)
    conv2_out3_back = conv2_out3_back1 + conv2_out3_back2 + conv2_out3_back3 + conv2_out3_back4
    
    conv2_out4_back1 = conv2_filter4.backward(relu2_back4,learning_rate,0)
    conv2_out4_back2 = conv2_filter4.backward(relu2_back4,learning_rate,1)
    conv2_out4_back3 = conv2_filter4.backward(relu2_back4,learning_rate,2)
    conv2_out4_back4 = conv2_filter4.backward(relu2_back4,learning_rate,3)
    conv2_out4_back = conv2_out4_back1 + conv2_out4_back2 + conv2_out4_back3 + conv2_out4_back4
    
    #   Maxpool 1
    maxpool1_back_1 = maxpool1_1.backward(conv2_out1_back)
    maxpool1_back_2 = maxpool1_2.backward(conv2_out2_back)
    maxpool1_back_3 = maxpool1_3.backward(conv2_out3_back)
    maxpool1_back_4 = maxpool1_4.backward(conv2_out4_back)
    
    
    #   Relu 1
    relu1_back1 = relu1_1.backward(maxpool1_back_1)
    relu1_back2 = relu1_2.backward(maxpool1_back_2)
    relu1_back3 = relu1_3.backward(maxpool1_back_3)
    relu1_back4 = relu1_4.backward(maxpool1_back_4)
    

    #   Conv 1
    conv1_out1 = conv1_filter1.backward(relu1_back1,learning_rate,0)
    conv1_out2 = conv1_filter2.backward(relu1_back2,learning_rate,0)
    conv1_out3 = conv1_filter3.backward(relu1_back3,learning_rate,0)
    conv1_out4 = conv1_filter4.backward(relu1_back4,learning_rate,0)


totalAccurate = 0
# TEST
# images should be a list of images, in other words, a list of 2D arrays
def test(images: list, y_true: list, check):
    totalLoss = 0
    global totalAccurate
    #totalAccurate = 0
    breeds = ["Husky", " Beagle", " Golden Retriver"]
    print("Starting test for ",breeds[check])
    for each in images:
        try:
            print("heyy")
            loss, out = forward(each, y_true)
            totalLoss += loss
            predictedClass = np.argmax(out)
            print(breeds[predictedClass])
            if(predictedClass == check):
                print("CORRECT!!!")
                totalAccurate+=1
                #print(totalAccurate)
        except Exception as e:
            print(e)
        
    print("The loss average is: ", totalLoss/len(images))
    print(f"The accuracy is: {totalAccurate}/{len(images)} = {totalAccurate/len(images)*100}%")
        





# TRAIN
def train(images: list, y_true: list, learning_rate: float):
    totalLoss = 0
    totalAccurate = 0
    for each in images:
        try:
            forward(each, y_true)
            backward(y_true, learning_rate)
        except:
            continue

    





images_husky = [("../ILSVRC/Data/CLS-LOC/train/n02109961/" + f) for f in os.listdir("../ILSVRC/Data/CLS-LOC/train/n02109961")]
images_beagle = [("../ILSVRC/Data/CLS-LOC/train/n02088364/" + f) for f in os.listdir("../ILSVRC/Data/CLS-LOC/train/n02088364")] 
images_golden = [("../ILSVRC/Data/CLS-LOC/train/n02099601/" + f) for f in os.listdir("../ILSVRC/Data/CLS-LOC/train/n02099601")]

initialize()
# train(images_husky[1:10],[1,0,0],0.01)
# train(images_beagle[1:10],[0,1,0],0.01)
# train(images_golden[1:10],[0,0,1],0.01)
train(images_beagle[11:13],[0,1,0],0.01)
train(images_golden[11:12],[0,0,1],0.01)

train(images_husky[51:75],[1,0,0],0.01)
train(images_golden[51:75],[0,0,1],0.01)
train(images_beagle[51:75],[0,1,0],0.01)

train(images_golden[101:125],[0,0,1],0.01)
train(images_husky[101:125],[1,0,0],0.01)
train(images_beagle[101:125],[0,1,0],0.01)

# train(images_golden[151:175],[0,0,1],0.01)
# train(images_beagle[151:175],[0,1,0],0.01)
# train(images_husky[151:175],[1,0,0],0.01)


# test(images_husky[1000],[1,0,0],0)
test([images_beagle[2]],[0,1,0],1)
# test(images_golden[1000:1050],[0,0,1],2)




