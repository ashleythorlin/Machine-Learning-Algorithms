import numpy
from tensorflow import keras
from keras.utils import np_utils
from keras.utils.vis_utils import plot_model
from keras.datasets import cifar10

seed = 21

### Prep Data

# Load in data
(input_data, target_data), (test_input_data, test_target_data) = cifar10.load_data()

# Normalize inputs from 0-255 to between 0-1 (too wide of a range can negatively impact performance)
input_data = input_data.astype('float32')
test_input_data = test_input_data.astype('float32')
input_data = input_data/255.0
test_input_data = test_input_data/255.0

# One-hot encode outputs
target_data = np_utils.to_categorical(target_data)
test_target_data = np_utils.to_categorical(test_target_data)
# specify number of classes in dataset
class_num = test_target_data.shape[1]

### Design CNN Model

# Define format and add layers

# Convolutional Layer - takes in inputs and runs convolutional filters
model = keras.Sequential()
# 32: number of channels, 3: size of filter is 3x3, relu: activation layer, same: means we aren't changing the image size
model.add(keras.layers.Conv2D(32, 3, input_shape=(5, 32, 32, 3), activation='relu', padding='same'))

# Dropout Layer - prevents overfitting by randomly eliminating some connections between layers
# 0.2: drops 20% of connections
model.add(keras.layers.Dropout(0.2))

# Batch Normalization = normalizes inputs sent to next layer (ensures that NN always creates activations with the distribution we want)
model.add(keras.layers.BatchNormalization())

# Basic block for building CNNs is convolutional layer, activation, dropout, pooling
# Blocks are stacked in pyramid-like pattern, next block containing a conv. layer with a larger filter to find patterns in greater detail/abstract further
# With new convolutional layers, layer count typically increases so the model learns increasingly copmlex representations of patterns (powers of 2 when training on GPU)
model.add(keras.layers.Conv2D(64, 3, activation='relu', padding='same'))
# Important not to have too many pooling layers since they discard data by decreasing input dimensions
model.add(keras.layers.MaxPooling2D(2))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.BatchNormalization())

model.add(keras.layers.Conv2D(64, 3, activation='relu', padding='same'))
model.add(keras.layers.MaxPooling2D(2))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.BatchNormalization())

model.add(keras.layers.Conv2D(128, 3, activation='relu', padding='same'))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.BatchNormalization())

# Flatten data after convolutional layers
model.add(keras.layers.Flatten())
model.add(keras.layers.Dropout(0.2))

# Create a densely connected layer to extract info from feature maps created by conv. layers (info used to classify images)
# Too many densely connected layers can increase the number of neurons/parameters greatly and cause poor performance/accuracy
model.add(keras.layers.Dense(32, activation='relu'))
model.add(keras.layers.Dropout(0.3))
model.add(keras.layers.BatchNormalization())

# Create a densely connected layer with the number of classes for the number of neurons
# Layer returns a neuron vector with length class_num that will store the probability that the image belongs to the class in question
# softmax function will return the neuron with the highest probability, classifying the image
model.add(keras.layers.Dense(class_num, activation='softmax'))

# Compile model
# Optimizer tunes weights in the network to approach the point of lowest loss
# Tracked metrics will be accuracy and validation accuracy to avoid significantly overfitting the CNN (if the network performs better on validation set, then it is overfitting)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# print(model.summary())

# Model Training
# use generated seed as the seed and save model performance to history
numpy.random.seed(seed)
history = model.fit(input_data, target_data, validation_data=(test_input_data, test_target_data), epochs=25, batch_size=64)
scores = model.evaluate(test_input_data, test_target_data, verbose=0)
print(f"Accuracy: {round(scores[1]*100, 2)}%")

# pd.DataFrame(history.history).plot()
# plt.show()

plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
model.save('image_recognition.h5')









