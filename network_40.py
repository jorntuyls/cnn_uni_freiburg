# code is inspired by the Lasagne example mnist.py on
#   https://github.com/Lasagne/Lasagne/tree/master/examples
#   and the Lasagne tutorial by Colin Raffel

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


    def build_cnn(self, input_var=None):
        # Create input layer of Network
        network = lasagne.layers.InputLayer(shape=(None,3,32,32),input_var=input_var)

        # Convolutional layer
        network = lasagne.layers.Conv2DLayer(
            network, num_filters=10, filter_size=(5, 5), pad='same',
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.HeNormal(gain='relu'))

        # Max pooling layer
        network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

        # Convolutional layer
        network = lasagne.layers.Conv2DLayer(
            network, num_filters=10, filter_size=(5, 5), pad='same',
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
            num_units=40,
            nonlinearity=lasagne.nonlinearities.sigmoid,
            W=lasagne.init.HeNormal(gain='relu'))

        return network

    # Code of following function is fully copied from
    #   https://github.com/Lasagne/Lasagne/tree/master/examples
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

    def train_network(self, network, X_train, y_train, X_val, y_val, input_var,
                num_epochs=500, batch_size=512, descent_type="sgd"):
        print("Start training network")
        # Create Theano variable for output vector
        true_output = T.imatrix('targets')

        # determinitic = False because we want to use dropout in training
        #   the network
        prediction = lasagne.layers.get_output(network, deterministic=False)
        loss = lasagne.objectives.binary_crossentropy(prediction, true_output).mean()

        # prediction_fn = theano.function([input_var], prediction)
        # loss_fn = theano.function([input_var, true_output], loss)
        #
        # print("Prediction: {}".format(prediction_fn(X_train)))
        # print("Loss: {}".format(loss_fn(X_train, y_train)))
        # print("y_train: {}".format(y_train))

        val_prediction = lasagne.layers.get_output(network, deterministic=True)
        val_loss = lasagne.objectives.binary_crossentropy(val_prediction, true_output).mean()

        # Get all paramters from the network
        all_params = lasagne.layers.get_all_params(network)

        # Use different techniques for updates
        # The standard technique is Stochastic Gradient Descent
        updates = lasagne.updates.sgd(loss, all_params, learning_rate=0.01)
        if (descent_type=="sgd"):
            pass
        elif (descent_type=="adagrad"):
            pass

        train = theano.function([input_var, true_output], loss, updates=updates)
        val = theano.function([input_var, true_output], val_loss)

        val_acc_calc = T.mean(T.eq(T.argmax(val_prediction, axis=1), true_output),
                      dtype=theano.config.floatX)
        val_acc_fn = theano.function([input_var, true_output],[val_acc_calc])

        get_output = theano.function([input_var], lasagne.layers.get_output(network, deterministic=True))

        # Keep track of training loss, validation loss and validation accuracy for
        #   visualization purposes
        lst_loss_train = []
        lst_loss_val = []
        lst_acc = []
        for epoch in range(num_epochs):

            train_err = 0
            train_batches = 0
            start_time = time.time()
            for batch in self.iterate_minibatches(X_train, y_train, 10, shuffle=True):
                inputs, targets = batch
                #print("Output: {}".format(lasagne.layers.get_output(network,X_train).eval()))
                #print("Targets: {}".format(targets))
                train_err += train(inputs, targets)
                train_batches += 1

            val_err = 0
            val_acc = 0
            val_batches = 0
            for batch in self.iterate_minibatches(X_val, y_val, 10, shuffle=False):
                inputs, targets = batch
                err = val(inputs, targets)
                val_err += err
                #acc = val_acc_fn(inputs, targets)
                #val_acc += acc
                val_batches += 1

            train_output = get_output(X_train)
            train_predictions = np.round(train_output)
            # print("train_output: {}".format(train_predictions))
            # print("y_train: {}".format(y_train))
            train_accuracy = np.mean(train_predictions == y_train)

            # Compute the network's output on the validation data
            val_output = get_output(X_val)
            # The predicted class is just the index of the largest probability in the output
            val_predictions = np.round(val_output)
            # The accuracy is the average number of correct predictions
            accuracy = np.mean(val_predictions == y_val)

            # Add training loss, validation loss and accuracy to lists
            lst_loss_train.append(train_err / train_batches)
            lst_loss_val.append(val_err / val_batches)
            lst_acc.append(accuracy)

            # Add training loss and
            # Then we print the results for this epoch:
            print("Epoch {} of {} took {:.3f}s".format(
                epoch + 1, num_epochs, time.time() - start_time))
            print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
            print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
            print("  training accuracy: \t\t{:.2f} %".format(train_accuracy * 100))
            print("  validation accuracy: \t\t{:.2f} %".format(accuracy * 100))

        print("Network trained")
        return network, lst_loss_train, lst_loss_val, lst_acc



    def main(self):
        # load data
        self.load_data()

        # Create a Theano tensor for a 4 dimensional ndarray
        input_var = T.tensor4('inputs')
        # Create network
        net = self.build_cnn(input_var)

        # Get index of attribute "Male" in attribute list
        i = self.attr_names.index("Male")

        # Downsample training data to make it a bit faster for testing this code
        n_train_samples = 1000
        n_val_samples = 1000
        train_idxs = np.random.permutation(self.train_images.shape[0])[:n_train_samples]
        val_idxs = np.random.permutation(self.val_images.shape[0])[:n_val_samples]
        X_train = self.train_images[train_idxs]
        y_train = self.train_labels[train_idxs]
        X_val = self.val_images[val_idxs]
        y_val = self.val_labels[val_idxs]

##        X_train = self.train_images
##        X_val = self.val_images
##        y_train = self.train_labels[:,i]
##        y_val = self.val_labels[:,i]

        # Train network
        results = self.train_network(net, X_train, y_train, X_val,
                    y_val, num_epochs=10, input_var=input_var)

        train_loss = results[1]
        val_loss = results[2]
        acc = results[3]
        # Vizualize losses and accuracy
        viz = V.Visualization()
        viz.visualize_losses(train_loss,val_loss)
        viz.visualize_accuracy(acc)

net = Network()
net.main()
