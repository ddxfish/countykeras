#Keras single number sequence prediction
#Today is 21-7-26

import random
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras import metrics
import numpy as np

#change these for some simple customizations
seqlength = 10
seqstep = 20
train_items = 1000
training_epochs = 5000

#Returns array of length seqlength, starting at number O, stepping up by seqstep each number
def upbyx(startnum=0, length=seqlength, step=seqstep):
    result = []
    for i in range(length):
        result.append(startnum + i * step)
    return result
print("This is a single sample of upbyx:")
#[0, 20, 40, 60, 80, 100, 120, 140, 160, 180]
print(upbyx())

#Build a set of training data
training_data = []
#Run upbyx to generate 1d array then stack the arrays into a 2d array
for _ in range(train_items):
    #training_data.append(upbyx(0, 10, 10)) #(testing only) quicker than below and results more accurate
    training_data.append(upbyx(0, seqlength, random.randint(0,100))) #randomizes pattern upbyx
train_set = np.array(training_data)
print("This is a single sample of upbyx in an np.array")
#[    0  1164  2328  3492  4656  5820  6984  8148  9312 10476]
print(train_set[0])

#features are X data we ask a prediction on (columns 1-9)
#labels are y values of the prediction (column 10)
#take a slice of the dataset for "answers" aka last number in sequences
train_labels = train_set[:,-1]
#now delete that last number from the "questions" aka first 9 numbers in sequence
train_features = np.delete(train_set, -1, 1)
print("Training data shape (samples, steps, features):", train_features.shape)
#Training data shape (samples, steps, features): (10000, 9)
print("Labels shape aka answers (samples, features):", train_labels.shape)
#Labels shape aka answers (samples, features): (10000,)
print("Sample X value", train_features[0])
#[   0 1164 2328 3492 4656 5820 6984 8148 9312]
print("Sample y value", train_labels[0])
#10476

#Reshape the data for Keras/TF model
#X should be of shape samples,9,1
X = np.reshape(train_features,(train_features.shape[0], train_features.shape[1], 1)).astype(float)
y = np.reshape(train_labels,(train_labels.shape[0], 1, 1)).astype(float)
print("We are feeding this data into the neural network")
print("Shape of X values:", X.shape)
#Shape of X values: (10000, 9, 1)
print("Shape of y values:", y.shape)
#Shape of y values: (10000, 1, 1)
print("Sample X value", X[0])
#[[   0.]
# [1164.]
# [2328.]
# [3492.]
# [4656.]
# [5820.]
# [6984.]
# [8148.]
# [9312.]]
print("Sample y value", y[0])
# [[10476.]]


#Build the model, I built it way more powerful than it needs to be
model = Sequential()
#add input layer, X.shape is (1,9,1), return sequences true until last LSTM
model.add(LSTM(64, input_shape = (X.shape[1], X.shape[2]), return_sequences = True))
model.add(Dropout(0.1))
model.add(LSTM(32, return_sequences = True))
#Dropout layer skipped, add to train more complex models
model.add(Dropout(0.1))
model.add(LSTM(32))
model.add(Dense(1, activation='linear'))
#I used rmsprop and Adam, and mse seemed to make the most sense for loss calcs
model.compile(optimizer='adam',loss='mse')
print(model.summary())
#fit our entire X and y data. split 20% of our data out for validation of training
history = model.fit(X, y, epochs=training_epochs, batch_size=100, verbose=1, validation_split = 0.2)


#Pass a test into the model and make a prediction
#Call upbyx with a step of 10, reshape it and convert to np array with floats
#-1 means to automatically determine
mytest = np.reshape(upbyx(0, 9, 10), (-1,9,1)).astype(float)
print("We are passing the model this shape: ", mytest.shape)
print("We are passing the model this data type: ", type(mytest))
print("We are asking the model to predict on this data:", mytest)
#make sure we pass in np array, 9 in length, np.reshape(1,9,1)
prediction = model.predict(mytest)
print("The model thinks this is the next number: ", prediction)

#Second test, this is the same as above, up by 77
mytest = np.reshape(upbyx(0, 9, 77), (-1,9,1)).astype(float)
print("We are passing the model this shape: ", mytest.shape)
print("We are passing the model this data type: ", type(mytest))
print("We are asking the model to predict on this data:", mytest)
#make sure we pass in np array, 9 in length, np.reshape(1,9,1)
prediction = model.predict(mytest)
print("The model thinks this is the next number: ", prediction)

#Graph the loss during training (delete for headless)
import matplotlib.pyplot as plt
def plot_loss(history, label='loss'):
  plt.plot(history.history['loss'], label=label)
  plt.xlabel('Epoch')
  plt.ylabel('Loss')
  plt.legend()
  plt.grid(True)
  plt.show()
plot_loss(history, "rms")
