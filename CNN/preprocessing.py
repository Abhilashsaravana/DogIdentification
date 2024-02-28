import numpy as np
def grayscale(img:np.array):
   #isolate the RGB channel of the image 
    orgR = img[:, :, 0]
    orgG = img[:, :, 1]
    orgB = img[:, :, 2]

    #create greyscale image using BT.709 standard
    grayImage = 0.2126 * orgR + 0.7152 * orgG + 0.0722*orgB

    return grayImage

def resize_img(image:np.array, new_size: int):
    # get the shape of the input image
    height, width = image.shape
    if height > width:  # Portrait Orientation
        new_height = new_size
        new_width = int(new_size * width / height)
    elif height < width:    # Landscpae Orientation
        new_width = new_size
        new_height = int(new_size * height / width)
    else:   # Sqaure image
        new_width = new_size
        new_height = new_size
    # create an empty image with given new size
    new_img = 255 * np.zeros((new_size, new_size), dtype=np.uint8)
    # calculate the pixel needed for on the x-y dimension to create the new image
    x_pad = (new_size - new_width) // 2
    y_pad = (new_size - new_height) // 2

    # Perform the resizing without changing the ratio of the original image
    for y in range(new_height):
        for x in range(new_width):
            orig_x = int(x * width / new_width)
            orig_y = int(y *height / new_height)
            new_img[y + y_pad, x + x_pad] = image[orig_y, orig_x]

    return new_img