import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as npimg
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

## Keras
import keras
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Lambda, Conv2D, Convolution2D, MaxPooling2D, Dropout, Flatten, Dense
#from keras.utils import multi_gpu_model
#import keras.backend.tensorflow_backend as tfback
import tensorflow as tf

import cv2
import pandas as pd
import random
import ntpath

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split


from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint

IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS = 66, 200, 3
INPUT_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)

columns = ["img","steering"]
data = pd.read_csv(r'/home/choyya/oscar_dataset/yild5350/img.csv', names = columns)
datadir = '/home/choyya/oscar_dataset/yild5350/2019-12-13-14-54-28/'


def load_img_steering(datadir, df):
  """Get img and steering data into arrays"""
  image_path = []
  steering = []
  for i in range(len(data)):
    indexed_data = data.iloc[i]
    center = indexed_data[0]
    image_path.append(os.path.join(datadir, center.strip()))
    steering.append(float(indexed_data[1]))
  image_paths = np.asarray(image_path)
  steerings = np.asarray(steering)
  return image_paths, steerings

image_paths, steerings = load_img_steering(datadir , data)

for s in steerings:
  print(s)
#print(steerings)


X_train, X_valid, Y_train, Y_valid = train_test_split(image_paths, steerings, test_size=0.2, random_state=0)


def img_preprocess(img):
  """Take in path of img, returns preprocessed image"""
  img = npimg.imread(img)
  # Cropping the image
  #img = img[60:-25, :, :]
  #img = img[300:,:,:]
  #plt.imshow(img)
  # Resizing the image
  img = cv2.resize(img, (IMAGE_WIDTH, IMAGE_HEIGHT), cv2.INTER_AREA)
  # Converting the image to YUV
  img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    
  return img
  
  
X_train = np.array(list(map(img_preprocess, X_train)))
X_valid = np.array(list(map(img_preprocess, X_valid)))


def nvidia_model():
  model = Sequential()
  model.add(Lambda(lambda x: x, input_shape=INPUT_SHAPE))  ## 127.5-1.0
  
  
  model.add(Conv2D(24,(5,5), strides=(2, 2), activation='relu'))
  model.add(Conv2D(36, (5,5), strides=(2, 2), activation='relu'))
  model.add(Conv2D(48, (5,5), strides=(2, 2), activation='relu'))
  model.add(Conv2D(64, (3,3), activation='relu'))
  model.add(Conv2D(64, (3,3), activation='relu'))
  #model.add(Dropout(0.5))
  model.add(Flatten())
  model.add(Dense(100, activation='relu'))
  model.add(Dense(50, activation='relu'))
  model.add(Dense(10, activation='relu'))
  model.add(Dense(1))
  # model.summary()
  optimizer = Adam(lr=1e-4)
  #model = multi_gpu_model(model, gpus = 2)
  model.compile(loss='mse', optimizer=optimizer,metrics=['mse'])
  # model.compile(loss='mse', optimizer=optimizer, metrics=['mean_squared_error'])
  return model

model = nvidia_model()

print(model.summary())



filepath="hdf5/nvidia_oscar_100.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

history = model.fit(X_train, Y_train, epochs=100, validation_data=(X_valid, Y_valid), batch_size=32, verbose=1, shuffle=1, callbacks=callbacks_list)
# model.save("base_cnn_1e4.h5")
print("model saved")
from matplotlib.pyplot import figure
figure(num=None, figsize=(3, 3), dpi=300, facecolor='w', edgecolor='k')

plt.plot(history.history['loss'], color = 'red')
plt.plot(history.history['val_loss'], color = 'green')

#plt.plot(history.history['loss'])
#plt.plot(history.history['val_loss'])
plt.title('Nvidia Model carla')
plt.ylabel('Mean Squared error')
plt.xlabel('Epoch')
plt.legend(['training_loss', 'validation_loss'], loc='upper right')
plt.savefig('plots/Nvidia_oscar_100.png', dpi=300, bbox_inches='tight')
plt.show()
