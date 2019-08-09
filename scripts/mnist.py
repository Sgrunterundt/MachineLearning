import numpy as np
import random

class Net(object):

    def __init__(self, sizes, activation_functions):
        """Initialize a net. Sizes is a list of the number of neurons in each layer. Activation_functions is a list of
        integers specifying the activation functions for each layer. 0 for sigmoid, 1 for relu and 2 for softmax """
        self.activation_functions = activation_functions
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

    def training(self, training_data, epochs, mini_batch_size, learning_rate, decay = 1, test_data=None):
        """Train the model via stochastic gradient descent. Training data is separated into mini batches of specified size,
         gradient is calculated for mini batch, and all training data is cycled through for a single epoch. If test data is
          given performance will be evaluated on test data after each epoch ad output to screen for monitoring"""
        if test_data: n_test = len(test_data)
        n = len(training_data)
        for i in range(epochs):                  #main training loop
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size] 
                for k in range(0, n, mini_batch_size)] #make mini_batches of training data and train on them
            for mini_batch in mini_batches:
                self.batch_update(mini_batch, learning_rate, decay) #In turn send each mini batch through the training routine
            if test_data:
                print("Epoch number {0} completed. Test result: {1} / {2}".format( #after the whole training set has been used once, do a test on the test data if it is included
                    i, self.evaluate(test_data), n_test))
                print(self.weights[1][0])
            else:
                print("Epoch {0} complete".format(i))  #otherwise just report the progress, then start new epoch

    def batch_update(self, mini_batch, learning_rate, decay):
        grad_b = [np.zeros(b.shape) for b in self.biases]
        grad_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:                                 #for each picture in the mini batch calculate the gradient, then add all the gradients of the batch before updating the weights and biases
            d_grad_b, d_grad_w = self.backprop(x, y)            #send one picture to gradient calculation
             grad_w = [gw+dgw for gw, dgw in zip(grad_w, d_grad_w)]
             grad_b = [gb+dgb for gb, dgb in zip(grad_b, d_grad_b)]  

        self.biases = [decay**learning_rate*b-(learning_rate/len(mini_batch))*gb
                        for b, gb in zip(self.biases, grad_b)]
        self.weights = [decay**learning_rate*w-(learning_rate/len(mini_batch))*gw
                        for w, gw in zip(self.weights, grad_w)]

    def backprop(self, x, y):
        grad_b = [np.zeros(b.shape) for b in self.biases]
        grad_w = [np.zeros(w.shape) for w in self.weights]
        # first send picture forwards through net to see how well the net does
        act = x
        acts = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w, f in zip(self.biases, self.weights, self.activation_functions):
            z = np.dot(w, act)+b
            zs.append(z)
            if (f==0):
                act = sigmoid(z)
            elif (f==1):
                act = relu(z)
            elif (f==2):
                act = softmax(z)
                    
            acts.append(act)
        # final layers activation is the result. Use to calculate the network loss
        delta = self.cost_derivative(acts[-1], y) #* dsigmoid(zs[-1]) #softmax with Cross-entropy loss makes this step simple
        #start backwards pass to propagate gradient of loss to the whole network
        grad_b[-1] = delta
        grad_w[-1] = np.dot(delta, acts[-2].transpose())
        """ For subseqent layers 'l' counts the layers from the result end. For a shallow network with
        only one hidden layer the following loop runs just once with l is set to one"""
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = dsigmoid(z) if (self.activation_functions[-l]==0) else drelu(z) #assumption that only the final layer is softmax
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            grad_b[-l] = delta
            grad_w[-l] = np.dot(delta, acts[-l-1].transpose())
        return (grad_b, grad_w)
    
     def feedforward(self, a):
        #Put a single piece of data through the net and return the result usefull for test and for using trained network
        for b, w, f in zip(self.biases, self.weights, self.activation_functions):
            a = (np.dot(w, a)+b)
            if (f==0):
                a = sigmoid(a)
            elif (f==1):
                a = relu(a) #This part seems a bit inellegant
            elif (f==2):
                a = softmax(a)
        return a

    def evaluate(self, test_data):
        """Evaluate the accuracy on the test data. """
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (output_activations-y)

#Activation functions and derivatives:

def relu(z):
    z[z<0]=0
    return z

def drelu(z):
    z[z<0]=0
    z[z>0]=1
    return z

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def dsigmoid(z):
    return sigmoid(z)*(1-sigmoid(z))

def softmax(z):
    shifted = z - np.max(z) #prevents overflow
    exps = np.exp(shifted)
    return exps/np.sum(exps)