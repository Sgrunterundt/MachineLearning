import torch
from torch.autograd import Variable
from torch.nn import ReLU

class Network(object):
    def __init__(self, sizes):
        
        self.num_layers=len(sizes)
        self.sizes = sizes
        self.weightInitializor()
        
    def weightInitializor(self):
        self.biases = [Variable(torch.randn(y, 1).cuda(), requires_grad=True) for y in self.sizes[1:]]
        self.weights = [Variable(torch.randn(y, x).cuda(), requires_grad=True) for x, y in zip(self.sizes[:-1], self.sizes[1:])]
        
    def feedForward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = ReLU((w*a)+b)
    

    