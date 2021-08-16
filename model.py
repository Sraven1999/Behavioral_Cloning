# imports
import csv
import cv2
import math
import sklearn
import numpy as np
from keras.layers import Input, Lambda
import tensorflow as tf
from keras.layers import Flatten, Dense, Dropout, MaxPooling2D, Convolution2D, Cropping2D, Activation
from keras.models import Sequential
from sklearn.utils import shuffle
import os

# reads data from driving log

samples = []
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)
    for line in reader:
        samples.append(line)

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples,test_size=0.2)

# generator code used from lab with adjustments added to account for multiple angles and code added to account for brighness change, the brighness change is added to change the brightness from a random number from a uniform distribution. Then the image is flipped and is added to teh array.
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                #for i in range(3):
                    #name = './data/IMG/'+batch_sample[i].split('/')[-1]
                    #center_image = cv2.imread(name)
                    #center_angle = float(batch_sample[3])
                path = 'data/IMG/'
                center_img = cv2.cvtColor(cv2.imread(path + sample[0].split('/')[-1]), cv2.COLOR_BGR2RGB)
                left_img = cv2.cvtColor(cv2.imread(path + sample[1].split('/')[-1]), cv2.COLOR_BGR2RGB)
                right_img = cv2.cvtColor(cv2.imread(path + sample[2].split('/')[-1]), cv2.COLOR_BGR2RGB)
                angle = float(sample[3])
                    if abs(center_angle) >=0:
                        #correction = 0.2
                        #if i == 1:
                        #   center_angle = center_angle + correction
                        #if i == 2:
                        #    center_angle = center_angle - correction
                        correction = 0.4
                        center_angle = angle
                        left_angle = center_angle + correction
                        right_angle = center_angle - correction
                    
                        images.extend([center_img, left_img, right_img])
                        angles.extend([center_angle, left_angle, right_angle])
                        image_flipped = np.fliplr(center_image)
                        angle_flipped = -center_angle
                        images.append(image_flipped)
                        angles.append(angle_flipped)
                        image_bright1 = brightness_change(center_image)
                        image_bright2 = brightness_change(image_flipped)
                        images.append(image_bright1)
                        angles.append(center_angle)
                        images.append(image_bright2)
                        angles.append(angle_flipped)
            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

def brightness_change(image):
# change the brightness
    image_change = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    random_bright = np.random.uniform(0.2,0.8)
    image_change[:,:,2] = image_change[:,:,2]*random_bright
    image_change = cv2.cvtColor(image1,cv2.COLOR_HSV2RGB)

    return image_change
# Set our batch size
batch_size=32

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)


model = Sequential()
# Preprocess incoming data, centered around zero with small standard deviation 
model.add(Lambda(lambda x: x/127.5 - 1., input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((70,25), (0,0))))
model.add(Convolution2D(24,5,5,border_mode='valid'))
model.add(MaxPooling2D((2,2)))
model.add(Activation('relu'))
model.add(Convolution2D(36,5,5,border_mode='valid'))
model.add(MaxPooling2D((2,2)))
model.add(Activation('relu'))
model.add(Convolution2D(48,3,3,border_mode='valid'))
model.add(MaxPooling2D((2,2)))
model.add(Activation('relu'))
model.add(Convolution2D(64,3,3,border_mode='valid'))
model.add(Flatten())
model.add(Dropout(0.1))
model.add(Activation('relu'))
model.add(Dense(1164))
model.add(Dropout(0.1))
model.add(Activation('relu'))
#model.add(Dense(850))
#model.add(Dropout(0.1))
#model.add(Activation('relu'))
#model.add(Dense(550))
#model.add(Dropout(0.1))
#model.add(Activation('relu'))
#model.add(Dense(200))
#model.add(Dropout(0.1))
#model.add(Activation('relu'))
model.add(Dense(100))  
model.add(Dropout(0.1))
model.add(Activation('relu'))
model.add(Dense(50))
model.add(Dropout(0.1))
model.add(Activation('relu'))
model.add(Dense(10))
model.add(Dropout(0.1))
model.add(Activation('linear'))
model.add(Dense(1))

model.compile(optimizer="adam", loss="mse")
model.fit_generator(train_generator, steps_per_epoch=math.ceil(len(train_samples)/batch_size), validation_data=validation_generator, validation_steps=math.ceil(len(validation_samples)/batch_size), epochs=5, verbose=1)
model.save('model.h5')
