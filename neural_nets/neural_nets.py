"""
Neural Network
"""

# Author: Ali Pala <alipala@buffalo.edu>
# Date of Creation: 01/12/2019

# To Do
# - handle subsequent batching
# - incorporate dropout into the model
# - early stopping
# - regularization
# - dropout


import numpy as np
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder
import warnings


# some helper functions
def _check_activation(activation):
    all_activation_functions = ['sigmoid', 'tanh', 'relu', 'softmax']
    if activation not in all_activation_functions:
        raise ValueError("Neural Network model supports only activation functions in %s, got"
                         " %s." % (all_activation_functions, activation))


def _check_layer_type(layer_type):
    all_layer_types = ['input', 'hidden', 'output']
    if layer_type not in all_layer_types:
        raise ValueError("Neural Network model expected layers %s, got"
                         " %s." % (all_layer_types, layer_type))


def _check_neuron_number(n_neurons):
    if n_neurons < 1:
        raise ValueError("Number of neurons expected positive, got non-positive.")


def _check_batch_method(batch_method):
    all_batch_methods = ['random', 'subsequent']
    if batch_method not in all_batch_methods:
        raise ValueError("Neural Network model expected match method %s, got"
                         " %s." % (all_batch_methods, batch_method))


def _check_optimizer(optimizer):
    all_optimizers = ['adam', 'gradient-decent']
    if optimizer not in all_optimizers:
        raise ValueError("Neural Network model expected optimizer %s, got"
                         " %s." % (all_optimizers, optimizer))


def _check_target(target):
    all_targets = ['regression', 'classification']
    if target not in all_targets:
        raise ValueError("Neural Network model expected target %s, got"
                         " %s." % (all_targets, target))


class Layer:

    def __init__(self, name, layer_type, n_neurons, activation):
        self.name = name
        self.n_neurons = n_neurons
        self.activation = activation
        self.layer_type = layer_type
        self.check_parameters()

    def __str__(self):
        return self.name

    def check_parameters(self):
        _check_layer_type(self.layer_type)
        _check_neuron_number(self.n_neurons)


class NeuralNetsTensorFlow:

    """
    This class develops a neural network model using Tensorflow.
    """

    number_of_features = None
    number_of_samples = None
    n_classes = None
    one_hot_encoder = None

    all_layers = []
    placeholders = {}
    session = None

    def __init__(self, target="classification", hidden_layers=(5,), learning_rate=0.01, epochs=1000,
                 optimizer="adam", hidden_layer_activation="relu", hold_rate=0.5,
                 reg_param=0.00001, n_steps=1, batch_size=32, batch_method='random', print_log=False,
                 print_step=100, seed=101, model_save=False, silence_warnings=True):
        self.target = target
        self.hidden_layers = hidden_layers
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.optimizer = optimizer
        self.hidden_layer_activation = hidden_layer_activation
        self.hold_rate = hold_rate
        self.reg_param = reg_param
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.batch_method = batch_method
        self.print_log = print_log
        self.print_step = print_step
        self.seed = seed
        self.model_save = model_save
        _check_batch_method(batch_method)
        _check_optimizer(optimizer)
        _check_target(target)
        if silence_warnings:
            warnings.filterwarnings("ignore")

    def __str__(self):
        return "somename"  # generate setup name

    def name_variables(self):
        weight_name = "weight" + str(len(self.all_layers))
        bias_name = "bias" + str(len(self.all_layers))
        return weight_name, bias_name

    def next_random_batch(self, X, y):
        rand_index = np.random.randint(0, len(X) - self.batch_size, self.batch_size)
        x_batch = X[rand_index]
        y_batch = y[rand_index]
        return x_batch, y_batch

    def generate_input_layer(self):
        with tf.variable_scope('input'):
            x = tf.placeholder(tf.float32, shape=(None, self.number_of_features))

        layer_name = "input_" + str(self.number_of_features)
        self.all_layers.append(Layer(layer_name, "input", self.number_of_features, x))
        self.placeholders['X'] = x
        if self.print_log:
            print("generated -- input layer")

    def add_layer(self, layer_type, n_neurons, activation):
        _check_activation(activation)
        weight_name, bias_name = self.name_variables()
        prev_n_neurons = self.all_layers[-1].n_neurons
        prev_activation = self.all_layers[-1].activation

        with tf.variable_scope("add_layer"):
            tf.set_random_seed(self.seed)
            weights = tf.get_variable(name=weight_name, shape=[prev_n_neurons, n_neurons], initializer=tf.contrib.layers.xavier_initializer())
            biases = tf.get_variable(name=bias_name, shape=[n_neurons], initializer=tf.zeros_initializer)
            if activation in ['relu']:
                activation_func = tf.nn.relu(tf.matmul(prev_activation, weights) + biases)
            elif activation in['sigmoid']:
                activation_func = tf.nn.sigmoid(tf.matmul(prev_activation, weights) + biases)
            elif activation in['tanh']:
                activation_func = tf.nn.tanh(tf.matmul(prev_activation, weights) + biases)
            else:
                activation_func = tf.nn.softmax(tf.matmul(prev_activation, weights) + biases)

        layer_name = layer_type + "_" + str(n_neurons)
        self.all_layers.append(Layer(layer_name, layer_type, n_neurons, activation_func))

    def generate_layers(self):
        # input layer
        self.generate_input_layer()
        # hidden layers
        for n_neurons in self.hidden_layers:
            self.add_layer("hidden", n_neurons, self.hidden_layer_activation)
            if self.print_log:
                print("generated -- hidden layer")
        # output layer
        if self.target in ['classification']:
            if self.n_classes > 2:
                self.add_layer("output", self.n_classes, "softmax")
            elif self.n_classes == 2:
                self.add_layer("output", 1, "sigmoid")
            else:
                raise ValueError("Neural Network model expected at least 2 classes but got less.")
        else:
            self.add_layer("output", 1, "relu")
        if self.print_log:
            print("generated -- output layer")

    def generate_cost_func(self):
        with tf.variable_scope("add_cost"):
            if self.target in ['classification']:
                if self.n_classes > 2:
                    y = tf.placeholder(tf.float32, shape=(None, self.n_classes))
                    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=self.all_layers[-1].activation))
                else:
                    y = tf.placeholder(tf.float32, shape=(None, 1))
                    cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=self.all_layers[-1].activation))
            else:
                y = tf.placeholder(tf.float32, shape=(None, 1))
                cost = tf.reduce_mean(tf.losses.mean_squared_error(self.all_layers[-1].activation, y))

        self.placeholders['y'] = y
        self.placeholders['cost'] = cost
        if self.print_log:
            print("generated -- cost function")

    def set_optimizer(self):
        with tf.variable_scope("set_optimizer"):
            if self.optimizer in ["adam"]:
                optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.placeholders.get("cost"))
            else:
                optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(self.placeholders.get("cost"))

        self.placeholders["optimizer"] = optimizer
        if self.print_log:
            print("generated -- optimizer")

    def set_saver(self):
        self.placeholders["model_saver"] = tf.train.Saver()
        if self.print_log:
            print("generated -- model saver")

    def build_network(self):
        self.generate_layers()
        self.generate_cost_func()
        self.set_optimizer()
        self.set_saver()

    def iterate(self, X, y):
        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())
        for epoch in range(self.epochs):
            for step in range(self.n_steps):
                if self.batch_method in ['random']:
                    batch_x, batch_y = self.next_random_batch(X, y)
                else:
                    batch_x, batch_y = X, y  # this line will ensure subsequent batching
                self.session.run(self.placeholders.get("optimizer"),
                                 feed_dict={self.placeholders.get("X"): batch_x, self.placeholders.get("y"): batch_y})
            if self.print_log:
                if epoch % self.print_step == 0:
                    training_cost = self.session.run(self.placeholders.get("cost"),
                                                     feed_dict={self.placeholders.get("X"): X,
                                                     self.placeholders.get("y"): y})
                    print("Epoch: {} - Training Cost: {}".format(epoch, training_cost))

        final_training_cost = self.session.run(self.placeholders.get("cost"),
                                               feed_dict={self.placeholders.get("X"): X,
                                               self.placeholders.get("y"): y})
        print("Training is complete! Final training cost: {}".format(final_training_cost))

        if self.model_save:
            path = "./logs/" + self.__str__() + "/" + self.__str__() + ".ckpt"
            save_path = self.placeholders.get("model_saver").save(self.session, path)
            print("Model saved: {}".format(save_path))

    def train(self, X, y):
        self.number_of_features = X.shape[1]
        self.number_of_samples = X.shape[0]
        if self.target in ["classification"]:
            self.n_classes = len(np.unique(y))
            if self.n_classes > 2:
                self.one_hot_encoder = OneHotEncoder()
                y = self.one_hot_encoder.fit_transform(y).toarray()
        self.build_network()
        self.iterate(X, y)

    def predict(self, X):
        return self.session.run(self.all_layers[-1].activation, feed_dict={self.placeholders.get("X"): X})

