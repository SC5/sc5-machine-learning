from keras.preprocessing.image import array_to_img, img_to_array, load_img
from keras.models import Sequential, load_model
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.utils.np_utils import to_categorical
from keras.optimizers import SGD

import numpy as np
import sys
import os
img_width, img_height = 240, 256

np.random.seed(12345)

# -- Data preparation --

X_train = []
Y_train = []
dirs = [x[0] for x in os.walk('data/train/')][1:]

for i, dir in enumerate(dirs):
  for filename in sorted(os.listdir(dir)):
    img = load_img(dir + '/' + filename)
    X_train.append(img_to_array(img))
    Y_train.append(i)

X_train = np.array(X_train)
Y_train = np.array(Y_train)
Y_train = to_categorical(Y_train, nb_classes=len(dirs))

X_test = []
Y_test = []
dirs = [x[0] for x in os.walk('data/test/')][1:]

for i, dir in enumerate(dirs):
  for filename in sorted(os.listdir(dir)):
    img = load_img(dir + '/' + filename)
    X_test.append(img_to_array(img))
    Y_test.append(i)
X_test = np.array(X_test)
Y_test = np.array(Y_test)
Y_test = to_categorical(Y_test, nb_classes=len(dirs))

# -- Neural network architecture --

if len(sys.argv) > 1:
  model_filename = sys.argv[1]
  model = load_model(model_filename)

else:
  model = Sequential()
  model.add(Convolution2D(32, 5, 5, input_shape=(3, img_width, img_height)))
  model.add(Activation('relu'))
  model.add(MaxPooling2D(pool_size=(3, 3)))
  model.add(Dropout(0.25))

  model.add(Convolution2D(32, 3, 3))
  model.add(Activation('relu'))
  model.add(MaxPooling2D(pool_size=(3, 3)))
  model.add(Dropout(0.25))

  model.add(Flatten())
  model.add(Dense(40))
  model.add(Activation('relu'))
  model.add(Dense(20))
  model.add(Activation('relu'))
  model.add(Dense(len(dirs)))
  model.add(Activation('relu'))

sgd = SGD(lr=0.001, clipnorm=1.)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

model.summary()

for i in xrange(0,1000):
  model.fit(
    X_train,
    Y_train,
    batch_size=30,
    nb_epoch=1
  )
  model.save('saved_model.h5')
  (loss_test, acc_test) = model.evaluate(X_test, Y_test, batch_size=32)
  print((loss_test, acc_test))
  if acc_test >= 0.9:
    print("Stopping criterion achieved, saving...")
    model.save('saved_model_high_acc.h5')
