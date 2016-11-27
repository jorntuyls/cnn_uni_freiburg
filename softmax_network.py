
import numpy as np
import time

import theano
import theano.tensor as T
import lasagne

from network import Network
import visualization as V

class SoftmaxNetwork(Network):

    def train_network(self, network, X_train, y_train, X_val, y_val, input_var,
                num_epochs=100, batch_size=512, descent_type="sgd"):
        print("Start training network")
        # Create Theano variable for output vector
        true_output = T.ivector('targets')

        # determinitic = False because we want to use dropout in training
        #   the network
        prediction = lasagne.layers.get_output(network, deterministic=False)
        loss = lasagne.objectives.categorical_crossentropy(prediction, true_output).mean()

        val_prediction = lasagne.layers.get_output(network, deterministic=True)
        val_loss = lasagne.objectives.categorical_crossentropy(val_prediction, true_output).mean()

        # Get all paramters from the network
        all_params = lasagne.layers.get_all_params(network)

        updates = self.get_updates_mechanism(loss, all_params, descent_type)

        train = theano.function([input_var, true_output], loss, updates=updates)
        val = theano.function([input_var, true_output], val_loss)

        get_output = theano.function([input_var], lasagne.layers.get_output(network, deterministic=True))

        # Keep track of training loss, validation loss and validation accuracy for
        #   visualization purposes
        lst_loss_train = []
        lst_loss_val = []
        lst_acc = []
        for epoch in range(num_epochs):

            train_err = 0
            train_batches = 0
            train_acc = 0
            start_time = time.time()
            for batch in self.iterate_minibatches(X_train, y_train, batch_size, shuffle=True):
                inputs, targets = batch
                # Calculate error
                train_err += train(inputs, targets)
                # Calculate accuracy of batch
                train_output = get_output(inputs)
                train_predictions = np.argmax(train_output, axis=1)
                train_acc += np.mean(train_predictions == targets)
                train_batches += 1

            val_err = 0
            val_acc = 0
            val_batches = 0
            for batch in self.iterate_minibatches(X_val, y_val, batch_size, shuffle=False):
                inputs, targets = batch
                err = val(inputs, targets)
                val_err += err
                # Calculate accuracy of batch
                val_output = get_output(inputs)
                # The predicted class is just the index of the largest probability in the output
                val_predictions = np.argmax(val_output, axis=1)
                val_acc += np.mean(val_predictions == targets)
                val_batches += 1

            # Add training loss, validation loss and accuracy to lists
            lst_loss_train.append(train_err / train_batches)
            lst_loss_val.append(val_err / val_batches)
            lst_acc.append(val_acc / val_batches * 100)

            # Add training loss and
            # Then we print the results for this epoch:
            print("Epoch {} of {} took {:.3f}s".format(
                epoch + 1, num_epochs, time.time() - start_time))
            print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
            print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
            print("  training accuracy: \t\t{:.2f} %".format(train_acc / train_batches * 100))
            print("  validation accuracy: \t\t{:.2f} %".format(val_acc / val_batches * 100))

        print("Network trained")
        return network, lst_loss_train, lst_loss_val, lst_acc

    def get_accuracy(self, network, input_var, X, y, batch_size=512):
        get_output = theano.function([input_var], lasagne.layers.get_output(network, deterministic=True))

        batches = 0
        acc = 0
        for batch in self.iterate_minibatches(X, y, batch_size, shuffle=False):
            inputs, targets = batch
            # Calculate batch accuracy
            output = get_output(inputs)
            predictions = np.argmax(output, axis=1)
            acc  += np.mean(predictions == targets)
            batches += 1

        print("  Test accuracy: \t\t{:.2f} %".format(acc / batches * 100))
        return acc / batches * 100

    def main(self, train_attribute, num_epochs=100, batch_size=512, name="softmax",
                    downsample_train=None, downsample_val=None):
        # load data
        self.load_data()

        if(downsample_train and downsample_val):
            # Downsample training data to make it a bit faster for testing this code
            n_train_samples = downsample_train
            n_val_samples = downsample_val
            train_idxs = np.random.permutation(self.train_images.shape[0])[:n_train_samples]
            val_idxs = np.random.permutation(self.val_images.shape[0])[:n_val_samples]
            X_train = self.train_images[train_idxs]
            X_val = self.val_images[val_idxs]
            X_test = self.test_images[val_idxs]
            y_val = self.val_labels[val_idxs]
            y_train = self.train_labels[train_idxs]
            y_test = self.test_labels[val_idxs]
        else:
            X_train = self.train_images
            X_val = self.val_images
            X_test = self.test_images
            y_train = self.train_labels
            y_val = self.val_labels
            y_test = self.test_labels

        # Create a Theano tensor for a 4 dimensional ndarray
        input_var = T.tensor4('inputs')

        # Create network
        net = self.build_cnn(num_outputs=2, output_nonlinearity="softmax", input_var=input_var)
        #net = self.build_cnn(input_var)

        # Get index of attribute "Male" in attribute list
        i = self.attr_names.index(train_attribute)

        # Get i'th column of labels
        y_train = y_train[:,i]
        y_val = y_val[:,i]
        y_test = y_test[:,i]

        # Train network
        results = self.train_network(net, X_train, y_train, X_val,
                    y_val, num_epochs=num_epochs, batch_size=batch_size, input_var=input_var)

        network = results[0]

        # Calculate test accuracy
        test_acc = self.get_accuracy(network, input_var, X_test, y_test, batch_size=batch_size)

        train_loss = results[1]
        val_loss = results[2]
        acc = results[3]

        self.visualize_losses(train_loss, val_loss, name=name, timestamp=self.timestamp)

        self.visualize_accuracy(acc, name=name, timestamp=self.timestamp)

        self.visualize_filters(net, layer=0, name=name, timestamp=self.timestamp)

        print("  Test accuracy: \t\t{:.2f} %".format(test_acc))
        f = open('test_accuracy','a')
        f.write("  Timestamp: {} \n  \t-Test: {}\n \t-Accuracy: {:.2f} % \n".format(self.timestamp, name, test_acc))
        f.close()
