
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import time

'''
The visualization class is responsible for visualizing results from training a cnn
'''

class Visualization:

    def visualize_losses(self, train_loss, val_loss, name="", timestamp=0):
        assert(len(train_loss) == len(val_loss))
        epochs = [i for i in range(1,len(train_loss)+1)]
        plt.figure("loss")
        plt.plot(epochs, train_loss, 'bs', label="training loss")
        plt.plot(epochs, val_loss, 'ro', label="validation loss")
        plt.xlabel("epochs")
        plt.ylabel("loss")
        plt.savefig("figures/losses_" + name + "_" +  str(timestamp) + ".png")
        plt.show()

    def visualize_accuracy(self, accuracy, name="", timestamp=0):
        epochs = [i for i in range(1,len(accuracy)+1)]
        plt.figure("accuracy")
        plt.plot(epochs, accuracy, 'bs')
        plt.xlabel("epochs")
        plt.ylabel("accuracy")
        plt.savefig("figures/accuracy_" + name + "_" + str(timestamp) + ".png")
        plt.show()


    def visualize_filters(self, filters, name="", timestamp=0):
        plt.figure("filters")
        num_filters = filters.shape[0]
        for i in range(num_filters):
            plt.subplot(num_filters/10 + 1, 10, i+1)
            temp = filters[i]
            temp_d = self.deprocess_image(temp)
            plt.imshow(temp_d)
            plt.axis('off')
        plt.tight_layout(pad=-1.0, w_pad=0, h_pad=-3.0)
        plt.savefig("figures/filters_" + name + "_" + str(timestamp) + ".png", bbox_inches='tight')
        plt.show()

    def visualize_filters_2(self, filters, name="", timestamp=0):
        plt.figure("filters")
        num_filters = filters.shape[0]
        for i in range(num_filters):
            plt.subplot(num_filters/10 + 1, 10, i+1)
            temp = filters[i]
            temp_trans = np.abs(temp.transpose())
            plt.imshow(temp_trans*255)
            plt.axis('off')
        plt.tight_layout(pad=-1.0, w_pad=0, h_pad=-3.0)
        plt.savefig("figures/filters_" + name + "_" + str(timestamp) + ".png", bbox_inches='tight')
        plt.show()

    # Code of this method is fully copied from
    #   https://blog.keras.io/how-convolutional-neural-networks-see-the-world.html
    def deprocess_image(self,x):
        # normalize tensor: center on 0., ensure std is 0.1
        x -= x.mean()
        x /= (x.std() + 1e-5)
        x *= 0.1

        # clip to [0, 1]
        x += 0.5
        x = np.clip(x, 0, 1)

        # convert to RGB array
        x *= 255
        x = x.transpose((1, 2, 0))
        x = np.clip(x, 0, 255).astype('uint8')
        return x
