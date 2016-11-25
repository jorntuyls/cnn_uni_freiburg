

from network import Network
from sigmoid_network import SigmoidNetwork
from softmax_network import SoftmaxNetwork

# train sigmoid network on all attributes
net = SigmoidNetwork()
net.main(   train_attribute="all",
            test_attribute="Male",
            num_epochs=10,
            batch_size=10,
            name="sigmoid_all_male",
            downsample_train=10,
            downsample_val=10)

# train sigmoid network on one attribute "Male"
net = SigmoidNetwork()
net.main(   train_attribute="Male",
            num_epochs=10,
            batch_size=10,
            name="sigmoid_male",
            downsample_train=10,
            downsample_val=10)

# train softmax network for one attribute "Male"
net = SoftmaxNetwork()
net.main(   train_attribute="Male",
            num_epochs=10,
            batch_size=10,
            name="softmax_male",
            downsample_train=10,
            downsample_val=10)
