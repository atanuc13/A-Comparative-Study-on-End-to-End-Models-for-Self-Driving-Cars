

import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Conv2D, BatchNormalization, LSTM, Dense, Flatten

from keras.layers import Reshape

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as npimg
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

## Keras
import keras
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Lambda, Conv2D, Convolution2D, MaxPooling2D, Dropout, Flatten, Dense
#from keras.utils import multi_gpu_model
#import keras.backend.tensorflow_backend as tfback
import tensorflow as tf

from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint


import cv2
import pandas as pd
import random
import ntpath

## Sklearn
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
import tensorflow as tf
tf.keras.backend.clear_session()



IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS = 66, 200, 3
INPUT_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)

# In[5]:


columns = ["img","steering"]
columns = ["img","steering"]
data = pd.read_csv(r'/home/choyya/dataset1/data1.csv', names = columns)
datadir = '/home/choyya/dataset1/data/'


# In[11]:


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


# In[16]:


X_train, X_valid, Y_train, Y_valid = train_test_split(image_paths, steerings, test_size=0.2, random_state=0)


# In[19]:


def img_preprocess(img):
  """Take in path of img, returns preprocessed image"""
  img = npimg.imread(img)
  # Cropping the image
  #img = img[60:-25, :, :]
  #img = img[200:-5, :, :]
  #img = img[300:,:,:]
  #cv2.imshow("Sheep", img)
  # Resizing the image
  img = cv2.resize(img, (IMAGE_WIDTH, IMAGE_HEIGHT), cv2.INTER_AREA)
  # Converting the image to YUV
  img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
  #cv2.imshow("Sheep", img)
  return img


# In[ ]:


X_train = np.array(list(map(img_preprocess, X_train)))
X_valid = np.array(list(map(img_preprocess, X_valid)))

# In[ ]:


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, BatchNormalization, LSTM, Dense, Flatten, Reshape
from tensorflow.keras.applications.vgg16 import VGG16

# Load VGG16 model
vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze all layers in VGG16
for layer in vgg16.layers:
    layer.trainable = False

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, BatchNormalization, LSTM, Dense, Flatten, Reshape
from tensorflow.keras.applications.vgg16 import VGG16
from keras.layers import TimeDistributed, LSTM, GlobalAveragePooling2D, RepeatVector  
# Load VGG16 model
from keras.applications import VGG19
vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=(66, 200, 3))
timesteps=5
# Freeze all layers in VGG16
for layer in vgg16.layers:
    layer.trainable = False
def my_model():
  model = Sequential()
  model.add(Lambda(lambda x: x, input_shape=INPUT_SHAPE))   ###/127.5-1.0
  model.add(vgg16)
  model.add(GlobalAveragePooling2D())

# Repeat the 2D tensor timesteps times to create a 3D tensor
  model.add(RepeatVector(timesteps))

 #Add the first LSTM layer with 256 units
  model.add(LSTM(100, return_sequences=False))

# Add the second LSTM layer with 128 units
 # model.add(LSTM(128))

 # model.add(Dense(256, activation='relu'))
  model.add(Dense(512, activation='relu'))
  #model.add(Dropout(0.5))
  
  model.add(Dense(256, activation='relu'))
  #model.add(Dropout(0.5))
  
  model.add(Dense(64, activation='relu'))
  #model.add(Dropout(0.5))
  
  model.add(Dense(1))
  # model.summary()
  optimizer = Adam(lr=1e-4)
  #model = multi_gpu_model(model, gpus = 2)
  model.compile(loss='mse', optimizer=optimizer, metrics=["mse"])
  # model.compile(loss='mse', optimizer=optimizer, metrics=['mean_squared_error'])

  return model


# In[ ]:


parallel_model = my_model()
print(parallel_model.summary())


# In[ ]:

filepath="F_n+lstm100_real_100.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

history = parallel_model.fit(X_train, Y_train, epochs=40, validation_data=(X_valid, Y_valid), batch_size=32, verbose=1, shuffle=1,callbacks=callbacks_list)
#parallel_model.save("f1_n+lstm_carla_100.h5")
print("Model saved")

from matplotlib.pyplot import figure
figure(num=None, figsize=(3, 3), dpi=300, facecolor='w', edgecolor='k')

plt.plot(history.history['loss'], color = 'red')
plt.plot(history.history['val_loss'], color = 'green')

#plt.plot(history.history['loss'])
#plt.plot(history.history['val_loss'])
plt.title('MSE vs Epoch')
plt.ylabel('Mean Squared error')
plt.xlabel('Epoch')
plt.legend(['training_loss', 'validation_loss'], loc='upper right')
plt.savefig('F_n+lstm100_real_100.png', dpi=300, bbox_inches='tight')
plt.show()


