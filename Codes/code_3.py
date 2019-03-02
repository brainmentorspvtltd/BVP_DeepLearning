# Using Deep Learning
import numpy as np
import tensorflow as tf
import os

trainImgArray = np.load('training_data.npy')
# print(trainImgArray.shape)
X = trainImgArray.reshape((trainImgArray.shape[0],-1))
#X.dtype = np.float32
labels = os.listdir('dataset/Training/')
labels_dict = {i : labels[i] for i in range(len(labels))}

# Input Size
D = X.shape[1]
# Hidden layer size
H = 3
# Number of classes
C = 3

def readlabelsLength(path):
    labels_length = []
    for root, folder, files in os.walk(path):
        labels_length.append(len(files))
    return labels_length

trainlabelslength = readlabelsLength(path='dataset/Training/')

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

Y = Y.flatten()
N = len(Y)
T = np.zeros((N,H))

for i in range(N):
    T[i, int(Y[i])] = 1

# print(T)
def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01, dtype=tf.float64))

def feedForward(X,W1,B1,W2,B2):
    Z = tf.nn.sigmoid(tf.matmul(X,W1)+ B1)
    return tf.matmul(Z,W2) + B2

tfX = tf.placeholder(tf.float32, shape=(None, D))
tfY = tf.placeholder(tf.float32, shape=(None, H))

W1 = init_weights([X.shape[1],H])
B1 = init_weights([H])
W2 = init_weights([H,C])
B2 = init_weights([C])

logits = feedForward(X,W1,B1,W2,B2)
cost = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=tfY, logits=logits))

trainOutput = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)
predictOutput = tf.argmax(logits,1)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
for i in range(1000):
    sess.run(trainOutput, feed_dict={tfX: X, tfY: T})
    pred = sess.run(predictOutput, feed_dict={tfX: X, tfY: T})
    print("Accuracy:", np.mean(Y == pred))




