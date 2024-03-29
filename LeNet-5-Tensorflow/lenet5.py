import tensorflow as tf
from tensorflow.contrib.layers import flatten
from sklearn.utils import shuffle
from tqdm import tqdm, trange
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

from matplotlib.legend_handler import HandlerLine2D



class Lenet5():

    def __init__(self, train_data, train_labels, test_data, test_labels, validation_data=None, validation_labels=None,
                 mean=0, stddev=0.3, learning_rate=0.001):

        self.train_data = train_data
        self.train_labels = train_labels
        assert (len(self.train_labels) == len(self.train_data))
        assert (self.train_data[0].shape[0] == 32 and self.train_data[0].shape[1] == 32)

        self.validation_data = validation_data
        self.validation_labels = validation_labels
        assert (len(self.validation_labels) == len(self.validation_data))
        assert (self.validation_data[0].shape[0] == 32 and self.validation_data[0].shape[1] == 32)

        self.test_data = test_data
        self.test_labels = test_labels
        assert (len(self.test_data) == len(self.test_labels))
        assert (self.test_data[0].shape[0] == 32 and self.test_data[0].shape[1] == 32)

        self.num_outputs = 10

        self.X = tf.placeholder(tf.float32, shape=(None, 32, 32, 1))
        self.y = tf.placeholder(tf.int32, shape=(None, self.num_outputs))

        self.mu = mean
        self.sigma = stddev

        # Layer 1: Input 32x32x1, Output 28x28x6
        self.conv1_kernels = tf.Variable(tf.truncated_normal(shape=[5, 5, 1, 6], mean=self.mu, stddev=self.sigma))
        self.conv1_biases = tf.get_variable(name="conv1_biases", shape=[6],
                                            initializer=tf.random_normal_initializer(stddev=0.3))
        self.conv1 = tf.nn.conv2d(self.X, self.conv1_kernels, [1, 1, 1, 1], padding='VALID') + self.conv1_biases
        # Pooling -> from 28x28 to 14x14
        self.pool1 = tf.nn.max_pool(self.conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
        # Activation
        self.conv1 = tf.nn.relu(self.pool1)

        # Layer 2: Input 14x14x6, Output 10x10x16
        self.conv2_kernels = tf.Variable(tf.truncated_normal(shape=[5, 5, 6, 16], mean=self.mu, stddev=self.sigma))
        self.conv2_biases = tf.get_variable(name="conv2_biases", shape=[16],
                                            initializer=tf.random_normal_initializer(stddev=self.sigma))
        self.conv2 = tf.nn.conv2d(self.conv1, self.conv2_kernels, [1, 1, 1, 1], padding='VALID') + self.conv2_biases
        # Pooling -> from 10x10x16 to 5x5x16
        self.pool2 = tf.nn.max_pool(self.conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
        # Activation 2
        self.conv2 = tf.nn.relu(self.pool2)

        # Flatten -> from 5x5x16 to 400x1
        self.flattened = flatten(self.conv2)

        # Fully Connected Layer n.1
        self.fcl1_weights = tf.Variable(tf.truncated_normal(shape=[400, 120], mean=self.mu, stddev=self.sigma))
        self.fcl1_biases = tf.get_variable(name="fc1_biases", shape=[120],
                                           initializer=tf.random_normal_initializer(stddev=self.sigma))
        self.fcl1 = tf.matmul(self.flattened, self.fcl1_weights) + self.fcl1_biases
        # Activation 3
        self.fcl1 = tf.nn.relu(self.fcl1)

        # Fully Connected Layer n.2
        self.fcl2_weights = tf.Variable(tf.truncated_normal(shape=[120, 84], mean=self.mu, stddev=self.sigma))
        self.fcl2_biases = tf.get_variable(name="fc2_biases", shape=[84],
                                           initializer=tf.random_normal_initializer(stddev=self.sigma))
        self.fcl2 = tf.matmul(self.fcl1, self.fcl2_weights) + self.fcl2_biases
        # Activation 4
        self.fcl2 = tf.nn.relu(self.fcl2)

        # Fully Connected Layer n.3
        self.fcl3_weights = tf.Variable(tf.truncated_normal(shape=[84, 10], mean=self.mu, stddev=self.sigma))
        self.fcl3_biases = tf.get_variable(name="fc3_biases", shape=[10],
                                           initializer=tf.random_normal_initializer(stddev=self.sigma))
        self.logits = tf.matmul(self.fcl2, self.fcl3_weights) + self.fcl3_biases

        # Loss and metrics
        self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=self.y)
        self.loss_op = tf.reduce_mean(self.cross_entropy)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        self.training_step = self.optimizer.minimize(self.loss_op)

        self.correct_prediction = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.y, 1))
        self.accuracy_operation = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
        self.saver = tf.train.Saver()


    def train(self, epochs, batch_size, auto_save=True):
        assert (epochs > 0 and batch_size > 0)

        loss_train = []
        accuracy_train = []
        loss_val = []
        accuracy_val = []
        loss_test = []
        accuracy_test = []
        E = []

        num_examples = len(self.train_data)
        print('Training the model . . .')
        with tf.Session() as session:
            session.run(tf.global_variables_initializer())
            total_steps = trange(epochs)
            for epoch in total_steps:
                E.append(epoch+1)
                ll = 0
                cc = 0
                self.train_data, self.train_labels = shuffle(self.train_data, self.train_labels)
                for offset in range(0, num_examples, batch_size):
                    end = offset + batch_size
                    X_batch, y_batch = self.train_data[offset:end], self.train_labels[offset:end]
                    _, acc, cross, loss_ = session.run([self.training_step, self.accuracy_operation, self.cross_entropy, self.loss_op],
                                                feed_dict={self.X: X_batch, self.y: y_batch})
                    ll += loss_ * batch_size
                    cc += acc * batch_size

                # loss_t, train_accuracy = self.evaluate(self.train_data, self.train_labels,  batch_size)
                loss_train.append(ll/num_examples)
                accuracy_train.append(cc/num_examples)
                print("Training set: Epoch: ", epoch+1, cc/num_examples, ll/num_examples)
                if self.validation_data is not None:
                    loss_v, validation_accuracy = self.evaluate(self.validation_data, self.validation_labels, batch_size)
                    loss_val.append(loss_v)
                    accuracy_val.append(validation_accuracy)

                    print("Valiation set: Epoch: ", epoch+1, validation_accuracy, loss_v)
                    # loss_t, train_accuracy = self.evaluate(self.train_data, self.train_labels, batch_size)
                    # print("Training set: Epoch: ", epoch+1, train_accuracy, loss_t)
                    loss_tt, test_accuracy = self.evaluate(self.test_data, self.test_labels, batch_size=batch_size)
                    print("Testing set: Epoch: ", epoch+1, test_accuracy, loss_tt)
                    # total_steps.set_description(
                    #     "Epoch {} - validation accuracy {:.3f} ".format(epoch + 1, validation_accuracy))
                    # # print("Epoch {} - validation accuracy {:.3f} ".format(epoch+1,validation_accuracy))
                    # tcotal_steps.set_description(
                    #     "Epoch {} - validation accuracy {:.3f} ".format(epoch + 1, validation_accuracy))
                    loss_test.append(loss_tt)
                    accuracy_test.append(test_accuracy)
                if auto_save and (epoch % 10 == 0):
                    save_path = self.saver.save(session, 'tmp/model.ckpt'.format(epoch))
            _, test_accuracy = self.evaluate(self.test_data, self.test_labels, batch_size=batch_size)
            line1, = plt.plot(E, loss_train, color='b', label = "train set loss", lw=2)
            line2, = plt.plot(E, loss_val,  color='r', label = "validation set loss", lw=2)
            line3, = plt.plot(E, loss_test, color='g', label = "test set loss", lw=2)
            #plt.plot(stoch_points, mean - int_conf, '--', color='r',lw=2)
            plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)}, prop={'size':15})
            #plt.show()
            plt.savefig("loss.pdf", bbox_inches='tight')
            plt.clf()

            line4, = plt.plot(E, accuracy_train, color='b', label = "train set accuracy", lw=2)
            line5, = plt.plot(E, accuracy_val,  color='r', label = "validation set accuracy", lw=2)
            line6, = plt.plot(E, accuracy_test, color='g', label = "test set accuracy", lw=2)
            #plt.plot(stoch_points, mean - int_conf, '--', color='r',lw=2)
            plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)}, prop={'size':15})
            plt.savefig("acc.pdf", bbox_inches='tight')
            return test_accuracy

    def evaluate(self, X_data, y_data, batch_size):
        num_examples = len(X_data)
        total_accuracy = 0
        total_loss = 0
        sess = tf.get_default_session()
        for offset in range(0, num_examples, batch_size):
            batch_x, batch_y = X_data[offset:offset + batch_size], y_data[offset:offset + batch_size]
            accuracy = sess.run(self.accuracy_operation, feed_dict={self.X: batch_x, self.y: batch_y})
            total_accuracy += (accuracy * len(batch_x))
        total_loss = sess.run(self.loss_op, feed_dict={self.X: X_data, self.y: y_data})
        return total_loss,  total_accuracy / num_examples

    def restore_model(self, path):
        with tf.Session() as session:
            self.saver.restore(sess=session, save_path=path)
