# MNIST-Deep-Neural-Network-Numpy
Training MNIST Dataset Deep Neural Network using NumPy only, MNIST consists of 70,000 grayscale images. Training Images are 60,000 Testing Images are 10,000. Neural Network Architecture: Input layer of 784 neurons (representing the 28x28 pixel images), one or more hidden layers, and an output layer of 10 neurons (one for each digit).

The activation functions used in the hidden layers of a neural network for MNIST classification are often sigmoid function. 

The output layer uses a SoftMax activation function to produce probabilities for each digit.

Every MNIST data point has two parts:	
      •	Image of a handwritten digit
      •	Corresponding label (number between 0 and 9 ) representing the digit drawn in the image.
The Labels for the above images are 5, 0, 4, and 1.
This Label will be used to compare the predicted digit (by the model) with the true digit (given by the data)

Training Network :

The preceding procedure gives us cost function, which we want to minimize in order to get a better model.
To minimize the cost, I used Gradient descent algorithm,
Where it shift each variable a little bit in the direction that reduces the cost.
Adjust the current weights and biases uses a learning rate.
There three parts to train this network, Apart from adding momentum term and Adam Optimizer

Forward Propagation :

Take an image and run through the network by applying activation functions and from that network you compute what your output is going to be

Hidden Layer = sigmoid ( For not activating all the neuron at same time for updating, so their output will be 0, dead neurons)

Output Layer = Softmax (To minimize value by probabilities between 0 and 1 , optimize cost function)

Back Propagation :

Here we go back in the model, Adjust weight and biases in order to optimize cost function at the current state.

First, we find the output Layer error by taking prediction and subtract the actual labels from them. then, we’re going to see how much each of the previous weights
and biases contributed to that error.   

Update Parameters :

Finally after backing up to the initial layer and calculating all the weights and biases
		[ W  =  W – Alpha(dW) ]
		[ b  =  b – Alpha(db) ]
We added the learning rate to each parameters and update the weights and biases.
So, with one hidden layer training a model for 10 epochs with 128 batch size
Test accuracy = 94.46%

Momentum : 

The momentum factor is a coefficient that is applied to an extra term in the weights update.
      •	Momentum is a technique to prevent sensitive movement stabilizes the momentum after getting computed every iterations
      •	Momentum is known to speed up learning and to help not getting stuck in local minima.
      •	I added the momentum 0.9
 Test accuracy = 95.14%

Adam Optimizer : 

There we have adaptive moment optimizer.

The algorithm computes the adaptive learning rates for each parameter and stores the first and second moments of the gradients. 
Adam optimizer is an extension of the stochastic gradient descent (SGD) algorithm that updates the learning rate adaptively.
    •	For every what it finds them their own learning rate.
        Test accuracy = 96.02%


OneHidden Test accuracy = 94.46%

Momentum Test accuracy = 95.14%

AdamOptimizer Test accuracy = 96.02%
