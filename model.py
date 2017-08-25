import csv
import cv2
import numpy as np
import tqdm as t
import matplotlib.pyplot as plt


images = []
measurements = []
data_directory = './data/'

lines = []

with open(data_directory + 'driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
            lines.append(line)
corrections = [0.0,0.25,-0.25]
    
for line in t.tqdm(lines): 
    for i in range(0,3):
        #pass through center,left,right images
        
        source_path = line[i]
        filename = source_path.split('/')[-1]
        current_path = data_directory + 'IMG/' + filename
        
        
        img = cv2.cvtColor(cv2.imread(current_path), cv2.COLOR_BGR2RGB)
        
        #cropped image to save memory, drive.py changed accordingly
        
        cropped_img = img[50:140,:]
        images.append(cropped_img)
        measurement = float(line[3]) + corrections[i]
        measurements.append(measurement)
        
        augimg = cv2.flip(cropped_img,1)
        images.append(augimg)
        measurements.append(measurement*-1)

# store training data

X_train = np.array(images)
y_train = np.array(measurements)

print(len(y_train))

from keras.models import Sequential
from keras.layers import Flatten,Dense, Lambda, Activation, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Cropping2D
# Nvidia

model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5,input_shape=(90,320,3)))
model.add(Convolution2D(24,5,5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(36,5,5, subsample=(2,2), activation='relu'))

model.add(Convolution2D(48,5,5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(64,3,3, activation='relu'))
model.add(Convolution2D(64,3,3, activation='relu'))
model.add(Flatten())
model.add(Dense(100))

model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

    
model.compile(loss='mse', optimizer='adam')
model.fit(X_train,y_train, validation_split=0.2,shuffle=True, epochs=5)

model.save('model.h5')

