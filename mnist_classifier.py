import argparse
import sys
import tempfile

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

def conv_2d_layer(x, W, strides = [1,1,1,1]):
    """
    Creates the convolution2d layers where x is the input, W the filters and s the stride
    :param x: the input to the convolution
    :param W: the weights of the filter
    :return: a tf conv2d layer
    """
    return tf.nn.conv2d(x, W, strides=strides, padding="SAME")

def bias_variable(name, shape):
    return tf.get_variable(name=name, shape=shape, initializer=tf.zeros_initializer())

def weights_variable(name, shape):
    return tf.get_variable(name=name, shape=shape, initializer=tf.contrib.layers.xavier_initializer())

def max_pool_layer(x, size = [1,2,2,1], strides = [1,2,2,1]):
    return tf.nn.max_pool(x, ksize=size, strides=strides, padding="SAME")


def create_dnn(x_flattened, n_h, n_w, n_c, nr_classes):
    """
    Function that creates the actual CNN
    :param x_flattened: flattened array of pixels
    :param n_h: Height of image in pixels
    :param n_w: Width of image in pixels
    :param n_c: Nr. of channels for image, e.g. 3 for RGB
    :param nr_classes: Nr. of classes to classify image into
    :return:
    """

    with tf.name_scope('reshape'):
        x_image_reshaped = tf.reshape(x_flattened, shape=[-1, n_h, n_w, n_c])

    # First conv layer (convert 3 channels to 32 features)
    with tf.name_scope('conv1'):
        w1 = weights_variable("W1", [5,5,n_c,32])
        b1 = bias_variable("B1", [32])
        z1 = conv_2d_layer(x_image_reshaped, w1) + b1
        a1 = tf.nn.relu(z1)

    # Max-pooling layer 1, n_h and n_w would be halved. So a 28 X 28 pixel image would now be 14 X 14
    with tf.name_scope('pool1'):
        h1 = max_pool_layer(a1)

    # Second conv layer (map 32 features to 64 features)
    with tf.name_scope('conv2'):
        w2 = weights_variable("W2", [5,5,32,64])
        b2 = bias_variable("B2", [64])
        z2 = conv_2d_layer(h1, w2) + b2
        a2 = tf.nn.relu(z2)

    # Max-pooling layer 2, n_h and n_w would be halved. So a 14 X 14 pixel image would now be 7 X 7
    with tf.name_scope('pool2'):
        h2 = max_pool_layer(a2)

    # Flatten and map to 1024 features
    with tf.name_scope('fc1'):
        w3 = weights_variable("W3", [7 * 7 * 64, 1024])
        b3 = bias_variable("B3", [1024])
        h2_flattened = tf.reshape(h2, [-1, 7 * 7 * 64])
        z3 = tf.matmul(h2_flattened, w3) + b3
        a3 = tf.nn.relu(z3)

    # Adding a dropout layer to prevent overfitting.
    with tf.name_scope('dropout'):
        keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(a3, keep_prob)

    # FC layer 2 to map 1024 features to nr.of classes that need to be detected
    with tf.name_scope('fc2'):
        w4 = weights_variable("W4", [1024, nr_classes])
        b4 = bias_variable("B4", [nr_classes])
        z4 = tf.matmul(h_fc1_drop, w4) + b4
        a4 = tf.nn.relu(z4)

    return (a4, keep_prob)

def train():
    # Import data
    mnist = input_data.read_data_sets("/home/vijai/Dev/tmp", one_hot=True)

    # 24 X 24 pixel images
    x = tf.placeholder(tf.float32, shape=[None, 784])

    # Needs to be classified into 10 classes
    y_ = tf.placeholder(tf.float32, shape=[None, 10])

    # creating the graph
    y_hat, keep_prob = create_dnn(x, 28, 28, 1, 10)

    with tf.name_scope('cost_calculation'):
        cost = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits= y_hat)

    cost = tf.reduce_mean(cost)

    with tf.name_scope('adam_optimizer'):
        optimizer = tf.train.AdamOptimizer(1e-4).minimize(cost)

    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(y_hat, 1), tf.argmax(y_, 1))
        correct_prediction = tf.cast(correct_prediction, tf.float32)

    accuracy = tf.reduce_mean(correct_prediction)

    graph_location = tempfile.mkdtemp()
    print('Saving graph to: %s' % graph_location)
    train_writer = tf.summary.FileWriter(graph_location)
    train_writer.add_graph(tf.get_default_graph())

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(20000):
            batch = mnist.train.next_batch(50)
            if i % 100 == 0:
                train_accuracy = accuracy.eval(feed_dict={
                    x: batch[0], y_: batch[1], keep_prob: 1.0})
                print('step %d, training accuracy %g' % (i, train_accuracy))
            optimizer.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

        print('test accuracy %g' % accuracy.eval(feed_dict={
            x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

train()