from keras.models import load_model
from keras.utils import load_img, img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.applications.vgg16 import VGG16
from keras.datasets import cifar10
import numpy as np
import random
  
from keras.models import load_model
  
(input_data, target_data), (test_input_data, test_target_data) = cifar10.load_data()

model = load_model('image_recognition.h5')

# image = load_img('images/0000.jpg', target_size=(32, 32))
# img = np.array(image)
# img = img / 255.0
# img = img.reshape(1, 32, 32, 3)
# label = model.predict(img)
# print("Predicted Class (0 - Cars , 1- Planes): ", label[0][0])

max = len(input_data)
data = []
truth_data = []
for i in range(5):
    ind = random.randrange(1, max)
    image = input_data[ind]
    img = np.array(image)
    img = img / 255.0
    img = img.reshape(32, 32, 3)
    data.append(img)

    truth_data.append(target_data[ind])

batch = np.array(data)
truth = np.array(truth_data)
prediction  = model.predict(batch, batch_size=5)
print(prediction)
num_predicts = len(truth)
highest_probs = np.amax(prediction, 1)

for i in range(num_predicts):
    prob = highest_probs[i]
    print(f"Prediction: {np.where(prediction==prob)[1]} \nProbability: {highest_probs[i]} \nTruth: {truth[i]}\n")