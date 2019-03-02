# Using Deep Learning
import numpy as np
import tensorflow as tf
import os

# load training data file
trainImgArray = np.load('training_data.npy')
# reshape the data into 2d
X = trainImgArray.reshape((trainImgArray.shape[0],-1))
# fetch the folder names for our labels
labels = os.listdir('dataset/Training/')
# build a dictionary of labels with key value pair
# {0:'Apple',1:'Orange',2:'Banana'}
labels_dict = {i : labels[i] for i in range(len(labels))}

# Input Layer of neural network
D = X.shape[1]
# Hidden layer neuron size
H = 3
# Number of classes (labels)
C = len(labels)

# fetch all the no of target images per folder
labels_length = []
for root, folder, files in os.walk('dataset/Training/'):
    labels_length.append(len(files))

# make an array to store labels to do label encoding
Y = np.zeros((len(trainImgArray),1))
slice_1 = 0
slice_2 = 0
try:
    for j in range(len(trainlabelslength)):
        slice_1 += trainlabelslength[j]
        slice_2 += trainlabelslength[j+1]
        Y[slice_1:slice_2] = int(j)
except BaseException:
    print("Index out of range")

# now convert labels into one hot encoding
Y = Y.flatten()
N = len(Y)
# make array which will store one hot encoded values
T = np.zeros((N,H))

for i in range(N):
    T[i, int(Y[i])] = 1

# function to initialize weights
def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01, dtype=tf.float64))

# feedforward
def feedForward(X,W1,B1,W2,B2):
    # applying activation function
    # multiple x_data with weights and add bias
    # put them in a sigmoid function
    Z = tf.nn.sigmoid(tf.matmul(X,W1)+ B1)
    # applying matrix multiplication for output layer
    # multiply hidden layer with weight_2 and add bias_2
    return tf.matmul(Z,W2) + B2

# build tensorflow placeholders for x and y
tfX = tf.placeholder(tf.float32, shape=(None, D))
tfY = tf.placeholder(tf.float32, shape=(None, H))

# initialize all weights and bias
W1 = init_weights([X.shape[1],H])
B1 = init_weights([H])
W2 = init_weights([H,C])
B2 = init_weights([C])

# call feedforward now
logits = feedForward(X,W1,B1,W2,B2)
# apply softmax on output layer because of multilabel classification
# softmax will return probabilities that will sum upto 1
# apply cross entropy as a cost function for softmax
cost = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=tfY, logits=logits))

# apply gradient descent and minimize error
trainOutput = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)
# get the index of max probability
# argmax will return index of max probability from logits
predictOutput = tf.argmax(logits,1)

# start tensorflow session
sess = tf.Session()
# globally initialize all variables that built above using tf
init = tf.global_variables_initializer()
# run session and it will initialize variables
sess.run(init)
# set epochs and start a loop
for i in range(1000):
    # run session to train the data and feed X and Y to tensorflow placeholders
    sess.run(trainOutput, feed_dict={tfX: X, tfY: T})
    # get the prediction and check average accuracy
    pred = sess.run(predictOutput, feed_dict={tfX: X, tfY: T})
    print("Accuracy:", np.mean(Y == pred))
