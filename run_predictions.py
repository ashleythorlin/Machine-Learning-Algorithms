from keras.models import load_model
from keras.datasets import cifar10
import numpy as np
import random
import matplotlib.pyplot as plt
from keras.models import load_model
  
batch_size = 5

(input_data, target_data), (test_input_data, test_target_data) = cifar10.load_data()

model = load_model('image_recognition.h5')
classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

# Show images
arr = input_data[0]

# image = load_img('images/0000.jpg', target_size=(32, 32))
# img = np.array(image)
# img = img / 255.0
# img = img.reshape(1, 32, 32, 3)
# label = model.predict(img)
# print("Predicted Class (0 - Cars , 1- Planes): ", label[0][0])

max = len(input_data)
data = []
truth_data = []
indices = []
plotimgs = []
for i in range(batch_size):
    # Generate random index
    ind = random.randrange(1, max)
    indices.append(ind)

    image = input_data[ind]
    img = np.array(image)
    #Retrieve image from CIFAR-10
    batchimg = img / 255.0
    batchimg = batchimg.reshape(32, 32, 3)
    data.append(batchimg)

    truth_data.append(target_data[ind])

    # Reshape image for plot
    plotimgs.append(img.reshape(32, 32, 3).astype("uint8"))


batch = np.array(data)
truth = np.array(truth_data)
# print(f"Batch: \n{batch}")
prediction  = model.predict(batch, batch_size=batch_size)
print(f"Prediction: \n{prediction}")
highest_probs = np.amax(prediction, 1)

# Print probabilities
fig, axes = plt.subplots(1, batch_size, figsize=(3, 3))
for i in range(batch_size):
    prob = highest_probs[i]
    axes[i].set_axis_off()
    axes[i].imshow(plotimgs[i])
    classind = np.where(prediction==prob)[1]
    info = f"Prediction: {classind} - {classes[classind[0]]} \nProbability: {highest_probs[i]} \nTruth: {truth[i]} - {classes[truth[i][0]]}\n"
    axes[i].set(title=f"Image #{indices[i]}")
    axes[i].text(16, 42, info, wrap=True, horizontalalignment='center', fontsize=12)

plt.show()