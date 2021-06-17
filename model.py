# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 15:56:17 2020

@author: Guido
"""

import time
import os
import csv
import matplotlib.pyplot as plt
import cv2
import numpy as np

t0=time.time() #Set initial time

os.chdir("C:/Users/Guido/Desktop/beta_simulator_windows/Take_3") #Setting the directory

## Extracting data from the csv driving_log


lines=[]
with open("driving_log.csv") as csvfile:
    reader=csv.reader(csvfile)
    for line in reader:
        lines.append(line)  
  
print("txt read. %s lines"%(len(lines)))
t1=time.time()

## Importing the images and flipping them
images=[]
measurements=[]

i=0 #counter of lines
for line in lines:
    i+=1
    if i%100==0: print("%s lines"%i) #Counter of lines read
    if i%1000==0: print("Time: %s s"%(time.time()-t0)) #Display time 
    for j in range(3):
        path=line[j].split('data/')[-1]
        image=plt.imread(path)
        flip=cv2.flip(image, 1) #Mirroring the image

        #Setting the correction factor for center and side cameras
        if j==0: correction=0
        elif j==1:correction=0.3
        elif j==2:correction=-0.3
        measurement=float(line[3])
        
        
        #Append the images and measurements values on lists
        images.append(image)
        measurements.append(measurement+correction)
        images.append(flip)
        measurements.append(-1.0*(measurement+correction))

t2=time.time()
delta=t2-t0
print("Time: %f"%delta)
print("All set. Showtime!")       
print("Number of images: %s"%len(measurements))




#################### Time for CNN!  ####################
import keras
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D as conv2d
from keras.layers.pooling import MaxPooling2D

#Creating the numpy arrays for CNN
images= np.array(images)
y_train= np.array(measurements)


#Building the CNN Architecture

#Initializing
model=Sequential()
model.add(Lambda(lambda x: x/255-0.5,input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))

#Convolutional part
model.add(conv2d(24,5,strides=(2,2)))
model.add(conv2d(36,5,strides=(2,2)))
model.add(conv2d(48,5,strides=(2,2)))
model.add(conv2d(64,3))
model.add(conv2d(64,3))

model.add(Flatten())

#Fully connected
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

#Training the CNN
model.compile(optimizer='adam', loss='mse')
model.fit(images,y_train,validation_split=0.2,shuffle=True,nb_epoch=1)
print("Saving...") 
model.save('../model.h5')
