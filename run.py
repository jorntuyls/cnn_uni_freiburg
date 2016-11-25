

from network import Network
from sigmoid_network import SigmoidNetwork
from softmax_network import SoftmaxNetwork

# train sigmoid network on all attributes
net = SigmoidNetwork()
net.main(   train_attribute="all",
            test_attribute="Male",
            name="sigmoid_all_male",
            downsample_x=1000,
            downsample_y=1000)

# train sigmoid network on one attribute "Male"
net = SigmoidNetwork()
net.main(   train_attribute="Male",
            name="sigmoid_male",
            downsample_x=1000,
            downsample_y=1000)

# train softmax network for one attribute "Male"
net = SoftmaxNetwork()
net.main(   train_attribute="Male",
            name="softmax_male",
            downsample_x=1000,
            downsample_y=1000)
