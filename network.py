# code is inspired by the Lasagne example mnist.py on
#   https://github.com/Lasagne/Lasagne/tree/master/examples
#   and the Lasagne tutorial by Colin Raffel

import abc
import numpy as np
import os
import time

import theano
import theano.tensor as T
import lasagne

import visualization as V


'''
Image arrays have the shape (N, 3, 32, 32), where N is the size of the
corresponding set. This is the format used by Lasagne/Theano. To visualize the
images, you need to change the axis order, which can be done by calling
np.rollaxis(image_array[n, :, :, :], 0, start=3).

Each image has an associated 40-dimensional attribute vector. The names of the
attributes are stored in self.attr_names.
'''

data_path = "/home/lmb/Celeb_data"


class Network:

    def __init__(self):
        self.timestamp = int(time.time())
        self.viz = V.Visualization()

    def new_timestamp(self):
        self.timestamp = int(time.time())

    '''
        Method to load the celebA data
    '''
    def load_data(self):
        print("Loading data")
        self.train_images = np.float32(np.load(os.path.join(
                data_path, "train_images_32.npy"))) / 255.0
        self.train_labels = np.uint8(np.load(os.path.join(
                data_path, "train_labels_32.npy")))
        self.val_images = np.float32(np.load(os.path.join(
                data_path, "val_images_32.npy"))) / 255.0
        self.val_labels = np.uint8(np.load(os.path.join(
                data_path, "val_labels_32.npy")))
        self.test_images = np.float32(np.load(os.path.join(
                data_path, "test_images_32.npy"))) / 255.0
        self.test_labels = np.uint8(np.load(os.path.join(
                data_path, "test_labels_32.npy")))

        with open(os.path.join(data_path, "attr_names.txt")) as f:
            self.attr_names = f.readlines()[0].split()
        print("Done loading data")

    '''
        Return: A convulutional neural network with the provided number of outputs
                    and provided output nonlinearity
    '''
    def build_cnn(self, num_outputs, output_nonlinearity, input_var=None):
        # Create input layer of Network
        network = lasagne.layers.InputLayer(shape=(None,3,32,32),input_var=input_var)

        # Convolutional layer
        network = lasagne.layers.Conv2DLayer(
            network, num_filters=32, filter_size=(5, 5), pad='same',
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.HeNormal(gain='relu'))

        # Max pooling layer
        network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

        # Convolutional layer
        network = lasagne.layers.Conv2DLayer(
            network, num_filters=32, filter_size=(5, 5), pad='same',
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.HeNormal(gain='relu'))

        # Max pooling layer
        network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

        # Convolutional layer
        network = lasagne.layers.Conv2DLayer(
            network, num_filters=32, filter_size=(5, 5), pad='same',
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.HeNormal(gain='relu'))

        # Max pooling layer
        # After applying this layer, size of volume should be (*,10,8,8)
        network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

        network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=.5),
            num_units=256,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.HeNormal(gain='relu'))

        # Output layer
        network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=.5),
            num_units=num_outputs,
            nonlinearity=self.get_nonlinearity(output_nonlinearity),
            W=lasagne.init.HeNormal(gain='relu'))

        return network
    '''
    Method to iterate inputs and targets in batches of size batchsize
    Code of this function is fully copied from
      https://github.com/Lasagne/Lasagne/tree/master/examples
    '''
    def iterate_minibatches(self, inputs, targets, batchsize, shuffle):
        assert len(inputs) == len(targets)
        if shuffle:
            indices = np.arange(len(inputs))
            np.random.shuffle(indices)
        for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
            if shuffle:
                excerpt = indices[start_idx:start_idx + batchsize]
            else:
                excerpt = slice(start_idx, start_idx + batchsize)
            yield inputs[excerpt], targets[excerpt]

    '''
        Return: update mechanism corresponding to given descent_type
    '''
    def get_updates_mechanism(self, loss, params, descent_type):
        # Use different techniques for updates
        # The standard technique is Stochastic Gradient Descent
        updates = None
        if (descent_type=="sgd"):
            updates = lasagne.updates.sgd(loss, params, learning_rate=0.01)
        elif (descent_type=="momentum"):
            updates = lasagne.updates.momentum(loss, params, learning_rate=0.01)
        elif (descent_type=="adam"):
            updates = lasagne.updates.adam(loss, params, learning_rate=0.01)

        return updates

    '''
        Return: objective function corresponding to given loss_function
    '''
    def get_objective_function(self, prediction, true_output, loss_function):
        loss = None
        if (loss_function=="categorical_crossentropy"):
            loss = lasagne.objectives.categorical_crossentropy(prediction, true_output)
        elif (loss_function=="binary_crossentropy"):
            loss = lasagne.objectives.binary_crossentropy(prediction, true_output)

        return loss

    '''
        Return: nonlinearity corresponding to given nonlinearity_name
    '''
    def get_nonlinearity(self, nonlinearity_name):
        nonlinearity = None
        if (nonlinearity_name == "sigmoid"):
            nonlinearity = lasagne.nonlinearities.sigmoid
        elif (nonlinearity_name == "softmax"):
            nonlinearity = lasagne.nonlinearities.softmax

        return nonlinearity

    '''
        Return: accuracy of the given network on dataset X and given true output y
    '''
    def get_accuracy(self, network, input_var, X, y, column=None, batch_size=512):
        get_output = theano.function([input_var], lasagne.layers.get_output(network, deterministic=True))

        batches = 0
        acc = 0
        for batch in self.iterate_minibatches(X, y, batch_size, shuffle=False):
            inputs, targets = batch
            # Calculate batch accuracy
            output = get_output(inputs)
            predictions = np.round(output)
            # if we want the predictions on a certain attribute (=column) of y
            if column:
                predictions = predictions[:,column]
                targets = targets[:,column]
            acc  += np.mean(predictions == targets)
            batches += 1

        return acc / batches * 100

    def visualize_losses(self, train_loss, val_loss, name="", timestamp=0):
        self.viz.visualize_losses(train_loss, val_loss, name=name, timestamp=timestamp)

    def visualize_accuracy(self, accuracy, name="", timestamp=0):
        self.viz.visualize_accuracy(accuracy, name=name, timestamp=timestamp)

    def visualize_filters(self, network, layer=0, name="", timestamp=0):
        params = lasagne.layers.get_all_param_values(network)
        filters_FC1 = params[0]
        self.viz.visualize_filters(filters_FC1, name=name, timestamp=timestamp)

    @abc.abstractmethod
    def train_network(self, network, X_train, y_train, X_val, y_val, input_var,
                num_epochs=100, batch_size=512, descent_type="sgd",
                objective_function="categorical_crossentropy"):
        '''
            Method to train a network should be implemented at subclass level
        '''
        return

    @abc.abstractmethod
    def main(self, train_attribute, num_epochs=100, batch_size=512, name="",
                    downsample_train=None, downsample_val=None):
        '''
            Main method should be implemented at subclass level
        '''
        return
