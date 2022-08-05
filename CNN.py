import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import cv2
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D

img = cv2.imread("three.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # cvt to greyscale to match database
resized = cv2.resize(gray, (28, 28), interpolation=cv2.INTER_AREA)  # resize image
normalized = tf.keras.utils.normalize(resized, axis=1)  # normalize
plt.imshow(normalized)
plt.show()

mnist = tf.keras.datasets.mnist
(xtr, ytr), (xte, yte) = mnist.load_data()  # get the handwriting dataset
print(xtr.shape)

# plt.imshow(xtr[0])  # load the first image of the training dataset
# plt.show()
plt.imshow(xtr[0], cmap=plt.cm.binary)  # specify that it should be a binary img
plt.show()

# print(xtr[0]) # print the image values\

# Normalize the data (Preprocessing before training)
# Normalizing data makes computation more efficient by limiting values between 0 and 1
xtr = tf.keras.utils.normalize(xtr, axis=1)
xte = tf.keras.utils.normalize(xte, axis=1)
plt.imshow(xtr[0], cmap=plt.cm.binary)
plt.show()

# Resize image to make it suitable for the convolution operation
IMGSZ = 28
xtrainr = np.array(xtr).reshape(-1, IMGSZ, IMGSZ, 1)  # increase one dimension for the convolutional kernel to scan
# over the image
xtestr = np.array(xtr).reshape(-1, IMGSZ, IMGSZ, 1)
print("Training Sample Dims", xtrainr.shape)
print("Testing Sample Dims", xtestr.shape)

# Init the DNN
model = Sequential()

# First Convolutional Layer (6000 images,
# creating 64 3x3 kernels for the operation
model.add(Conv2D(64, (3, 3), input_shape=xtrainr.shape[1:]))
model.add(Activation("relu"))  # Relu nonlinear activation
model.add(MaxPooling2D(pool_size=(2, 2)))  # Downscales image by running 2x2 mat and taking max value

# 2nd conv layer
model.add(Conv2D(64, (3, 3)))
model.add(Activation("relu"))  # Relu nonlinear activation
model.add(MaxPooling2D(pool_size=(2, 2)))  # Downscales image by running 2x2 mat and taking max value

# 3rd conv layer
model.add(Conv2D(64, (3, 3)))
model.add(Activation("relu"))  # Relu nonlinear activation
model.add(MaxPooling2D(pool_size=(2, 2)))  # Downscales image by running 2x2 mat and taking max value

# Fully Connect DNN
model.add(Flatten())  # Flatten the resulting 2d image pixels after processing from the convolutional layers to 1D in
# order to feed into the DNN
model.add(Dense(64))  # Number of neurons in the DNN (Dense means fully connected)
model.add(Activation("relu"))  # specify the activation method once again

# 2nd layer of the NN
model.add(Dense(32))  # Gradually reduce the number of neurons in the network to the number that should be in the
# output layer, in this case we are checking if a number is one to 10 so there should be 10 neurons at the end
model.add(Activation("relu"))  # specify the activation method once again

# Output layer
model.add(Dense(10))
model.add(Activation("softmax"))  # activation changed to softmax to convert to a probability value

# In terms of binary classification use sigmoid as the activation function

# model.summary() Trains the model while specifiying the cost function being use for back progpogation, the metric
# that is being optimized is accuracy
model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=['accuracy'])
# specifies test data, epochs mean training iterations
model.fit(xtrainr, ytr, epochs=5, validation_split=0.3)  # starts the training process

# Evaluate the model with the test data
# testloss, testacc = model.evaluate(xtestr, yte)
# print("Test Loss on 10000 samples", testloss)
# print("Validation Accuracy on 10000 samples", testacc)

# predictions = model.predict([xtestr])
# print(predictions)

# to get what the output value of the CNN is for a specific image use this:
# print(np.argmax(predictions[0]))
# plt.imshow(xte[0])
# plt.show()


newimg = np.array(normalized).reshape(-1, IMGSZ, IMGSZ, 1)  # for the kernel operation
prediction = model.predict(newimg)

print(np.argmax(prediction))
