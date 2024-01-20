# Forward Direct Feedback Alignment

PyTorch implementation of the Forward Direct Feedback Alignement (FDFA) algorithm [1]. 
The FDFA algorithm uses Forward-Mode Automatic Differentiation to estimate backpropagation paths as feedback 
connections in DFA. This is achieved by using 
[PyTorch's dual tensors](https://pytorch.org/tutorials/intermediate/forward_ad_usage.html) to compute 
directional derivatives in random directions during inference.

DFA-related code from [the Direct Kolen Pollack github repository](https://github.com/webstah/dkp-gist) has been used 
as a base for this project.

## Run the code

### Fully-Connected Networks

The following command runs the training of a 2-layers fully-connected network on the MNIST dataset with the FDFA algorithm:


`python3 main.py --dataset=MNIST --train-mode=FDFA --n-layers=2`

### Shallow Convolutional Neural Network (CNN)

A shallow CNN (15C5-P2-40C5-P2-128-10) can be trained by using the `--conv` argument instead of `--n-layers`:

`python3 main.py --dataset=MNIST --train-mode=FDFA --conv`

### AlexNet (CIFAR100)

AlexNet is trained on CIFAR100 by running the following command:

`python3 main.py --dataset=CIFAR100 --train-mode=FDFA`

### Error Backpropagation Algorithm

Alternatively, networks can be train using the error backpropagation algorithm by replacing the `--train-mode` argument:

`python3 main.py --dataset=MNIST --train-mode=BP --n-layers=2`

### Random Direct Feedback Alignment

Random DFA [2] can be used for training by specifying DFA in the `--train-mode` argument:

`python3 main.py --dataset=MNIST --train-mode=DFA --n-layers=2`

### Direct Kolen Pollack

The DKP algorithm [3] can also be used for training. This is done by setting `--train-mode` to DKP:

`python3 main.py --dataset=MNIST --train-mode=DKP --n-layers=2`

## References

[1] Bacho, Florian and Chu, Dominique, Low-Variance Forward Gradients Using Direct Feedback Alignment and Momentum. 
[http://dx.doi.org/10.2139/ssrn.4474515](http://dx.doi.org/10.2139/ssrn.4474515)

[2] Nøkland, A.. (2016). Direct Feedback Alignment Provides Learning in Deep Neural Networks. 
[https://arxiv.org/abs/1609.01596](https://arxiv.org/abs/1609.01596)

[3] Matthew Bailey Webster, Jonghyun Choi, & changwook Ahn. (2021). Learning the Connections in Direct Feedback 
Alignment.