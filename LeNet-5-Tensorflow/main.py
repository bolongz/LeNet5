from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
from lenet5 import Lenet5

mnist = input_data.read_data_sets("MNIST_data", reshape=False, one_hot=True)

X_train, y_train = mnist.train.images, mnist.train.labels
#print(type(mnist.train.labels))
# print(type(y_train))

#randomly shuffer the train set labels
np.random.shuffle(y_train)

# print(type(y_train))
#print(y_train.shape)
X_validation, y_validation = mnist.validation.images, mnist.validation.labels
X_test, y_test = mnist.test.images, mnist.test.labels

print("Input shape: {}".format(X_train[0].shape))
print("Training Set:   {} samples".format(len(X_train)))
print("Validation Set: {} samples".format(len(X_validation)))
print("Test Set:       {} samples".format(len(X_test)))

# 0-Padding for LeNet-5's input size
#print(X_train.shape())
print(y_train[0])
print(mnist.train.images.shape, mnist.train.labels.shape)
#print(mnist.train.images.shape, y_train.shape)
print(mnist.validation.images.shape, mnist.validation.labels.shape)
print(mnist.test.images.shape, mnist.test.labels.shape)
#print(mnist.train.labels[0])
# for i in range(0, mnist.train.labels.shape[0]):
#     mnist.train.labels[i] = y_train[i]


X_train = np.pad(X_train, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')
X_validation = np.pad(X_validation, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')
X_test = np.pad(X_test, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')

print(X_train.shape)
print(X_validation.shape)
print(X_test.shape)


print("New Input shape: {}".format(X_train[0].shape))

lenet_network = Lenet5(X_train,y_train,X_test,y_test,X_validation,y_validation)
accuracy = lenet_network.train(epochs=150,batch_size=100)
print("Accuracy on test set: {:.3f}".format(accuracy))

'''
# TODO: some refactoring for restoring the model
lenet_network_restored = Lenet5(X_train, y_train, X_test, y_test, X_validation, y_validation)
lenet_network_restored.restore_model(path='tmp/model.ckpt')
'''
