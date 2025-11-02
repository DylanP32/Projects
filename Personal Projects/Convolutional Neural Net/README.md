Convolutional Neural Network from scratch implementation only using python and CuPy.

I set out with the goal of implementing this algorithm from scratch so in the future whenever I dive into the many machine learning libraries
such as PyTorch and TensorFlow, I could understand what is going on behind the scenes and understand what they do further.

The current set up is 32 3x3 kernels with 2x2 max pooling followed by a layer of 64 3x3 kernels and another round of 2x2 max pooling.
This is then followed by a massive 3136 flattened layer to account for all those feature maps that feed into a 256 neuron hidden layer and a 10 neuron
ouput layer for the MNIST 0-9 handwritten number dataset.

Bottleneck: because this is a from scratch implementation, it is extremely unpotimized and actually takes forever to just finish a single pass with 1 image out of 59k.
But this is ok, I wanted to prove to myself I could build it not that it would be practical. With a small scale test of 5 images I proved that after 3 epochs that took almost an hour, it did start to learn. This is enough proof to me that if my implementation was optimized it could possibly be practical.