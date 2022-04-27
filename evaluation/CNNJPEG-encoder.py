# -*- coding: utf-8 -*-
"""
Created on Thu Sep 16 14:16:00 2021

@author: Diego Wildenstein

This is the serial implementation of the CNN-JPEG encoder. it uses a 3-layer 
CNN to reduce high frequency image components for better compression.
"""

import tensorflow as tf
import keras
from keras import layers
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

# Set the input Directory. The CNN will compress all images in this directory,
# assuming they are valid images.
inputDir = "./inputs_val/"

# Set the output directory where we write our compressed images.
outputDir = "./serial_val/"


# First, load the pretrained model weights and biases
#tModel = np.load("../CNNCompression-master/ColorModel/weightsComCNN.npz")
tModel = np.load("./weightsComCNN.npz")

w1 = tModel["w1"]
w2 = tModel["w2"]
w3 = tModel["w3"]

b1 = tModel["b1"]
b2 = tModel["b2"]
b3 = tModel["b3"]


# Now we can start to define the CNN-JPEG encoder model.
model = keras.models.Sequential([
    layers.Conv2D(filters=64, kernel_size=[3,3], strides=(1,1) ,\
                  weights=[w1,b1]),
    
    layers.Conv2D(filters=64, kernel_size=[3,3], strides=(2,2) ,\
                  weights=[w2,b2]),
    
    layers.Conv2D(filters=3, kernel_size=[3,3]  ,\
                  weights=[w3,b3])

    ])

model.compile(loss='MeanSquaredError', optimizer='adam', metrics=['accuracy'])


#set up a timer so we can find out how long the compressions take.
import time
t1 = time.time()


# Loop through the validation dir and compress all images by running a model 
# inference on each image.
count=0;
for file in os.listdir(inputDir):
    
    # When we collect timing data, remember to supress print statements.
    #print("Infile: ", inputDir+file)
    img = cv2.imread(inputDir+file)

    # cv2 is dumb and uses BGR format, so we have to convert it to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Show the test image
    #plt.imshow(img)

    # bIdx is the batch index, x and y are image dims, c is number of channels
    #(bIdx, x, y, c) = imgf.shape
    
    # Now we have to convert the images to floating point tensors.
    imgf = img.reshape(1,img.shape[0],img.shape[1],img.shape[2])
    imgf = imgf / 255
    imgTens = tf.convert_to_tensor(imgf, dtype=tf.float32)


    # This line runs the actual model inference on the image tensor.
    res = model(imgTens)
    
    # To print out the image, we have to remove the batch index dimension and 
    # extract the image from the tensor.
    res1 = res.numpy()
    res1 = res1.reshape((res1.shape[1], res1.shape[2], res1.shape[3]))
    res1 = (res1 / np.max(res1)) * 255
 
    
     #plt.imshow(res1.astype('uint8'))

    # Save the image, dont forget to save in RGB format.    
    res1 = cv2.cvtColor(res1, cv2.COLOR_BGR2RGB)
    cv2.imwrite(outputDir+str(count)+".jpg", res1)

    #print(count)
    count = count+1


# Stop timer, and print the elapsed time.
t2 = time.time()
print("elapsed: ", t2-t1)

#also print the throughput, the frames compressed per second.
throughput = (count+1) / (t2-t1)
print("Throughput = ",throughput)

#save the model for future use
model.save("Encoder-Model.hdf5")



