from keras.preprocessing.image import img_to_array, load_img
from keras.models import load_model
from keras.utils.np_utils import to_categorical

import numpy as np
import sys
import os
img_width, img_height = 240, 256


X_test = []
X_test_filename = []
Y_test = []
Y_test_classes = []
class_lookup = []
dirs = [x[0] for x in os.walk('data/test/')][1:]

for i, dir in enumerate(dirs):
  class_lookup.append(dir.split("/")[-1])
  for filename in sorted(os.listdir(dir)):
    X_test_filename.append(filename)
    img = load_img(dir + '/' + filename)
    X_test.append(img_to_array(img))
    Y_test.append(i)
    Y_test_classes.append(i)

X_test = np.array(X_test)
Y_test = np.array(Y_test)
Y_test = to_categorical(Y_test, nb_classes=len(dirs))

model_filename = sys.argv[1]
model = load_model(model_filename)

print("--- Test set predictions ---")

correct_predictions = 0.0
incorrect_predictions = 0.0
for i, example in enumerate(X_test):
  example = example.reshape((1,) + example.shape)
  filename = X_test_filename[i]
  predicted_class = model.predict_classes(example)[0]
  predicted_class_label = class_lookup[predicted_class]
  correct = "CORRECT!" if Y_test_classes[i] == predicted_class else "WRONG =("
  print(
    "image: " + filename +
    ", predicted class: " + str(predicted_class_label) +
    ", correct class: " + class_lookup[Y_test_classes[i]] +
    "..." + correct
  )
  if correct == "CORRECT!":
    correct_predictions += 1.0
  else:
    incorrect_predictions += 1.0
total_accuracy = correct_predictions / (correct_predictions + incorrect_predictions)
print(
  "The neural network got it right" +
  str(int(total_accuracy * 100)) + " per cent of the time!")

