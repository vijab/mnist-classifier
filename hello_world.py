import tensorflow as tf

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

def create_placeholders(n_x, n_y):
    X = tf.placeholder(dtype=tf.float32, name="X", shape=[None, n_x])
    Y = tf.placeholder(dtype=tf.float32, name="Y", shape=[None, n_y])
    return X,Y

def initialize_layers():
    """" X = 12288 inputs => Y = 6 outputs/classes, L = 3 layers """
    tf.set_random_seed(1)

    W1 = tf.get_variable("W1", [784, 256], initializer=tf.contrib.layers.xavier_initializer(seed = 1))
    b1 = tf.get_variable("b1", initializer=tf.zeros([256]))
    W2 = tf.get_variable("W2", [256, 256], initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b2 = tf.get_variable("b2", initializer=tf.zeros([256]))
    W3 = tf.get_variable("W3", [256, 10], initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b3 = tf.get_variable("b3", initializer=tf.zeros([10]))

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2,
                  "W3": W3,
                  "b3": b3}

    return parameters

def forward_propogation(X, parameters):

    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']

    Z1 = tf.add(tf.matmul(X, W1), b1)
    A1 = tf.nn.relu(Z1)
    Z2 = tf.add(tf.matmul(A1, W2), b2)
    A2 = tf.nn.relu(Z2)
    Z3 = tf.add(tf.matmul(A2, W3), b3)

    print(Z3)
    return Z3

def compute_cost(Z3, Y):

    logits = tf.transpose(Z3)
    labels = tf.transpose(Y)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=Z3))

    return cost

def train():
    # Parameters
    learning_rate = 0.1
    num_steps = 1500
    batch_size = 128
    display_step = 100

    (X, Y) = create_placeholders(mnist.train.images.shape[1], mnist.train.labels.shape[1])
    parameters = initialize_layers()

    Z3 = forward_propogation(X, parameters)

    cost = compute_cost(Z3, Y)

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        # Run the initializer
        sess.run(init)

        for step in range(1, num_steps + 1):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            # Run optimization op (backprop)
            _, acc = sess.run([optimizer, cost], feed_dict={X: batch_x, Y: batch_y})
            print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.3f}".format(acc))

        print("Optimization Finished!")

        # Calculate accuracy for MNIST test images
        print("Testing Accuracy:", \
              sess.run(cost, feed_dict={X: mnist.test.images,
                                            Y: mnist.test.labels}))


train()