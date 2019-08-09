import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as func
import torch.optim as optim
import mnist_loader_torch

class torchnet(nn.Module):
    
    def __init__(self):
        super(torchnet, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 10, 3)
        self.conv2 = nn.Conv2d(10, 20, 3)
        self.conv3 = nn.Conv2d(20, 20, 3)
        
        self.fc1 = nn.Linear(5*5*20, 100)
        self.fc2 = nn.Linear(100,10)
        #self.SM = nn.Softmax(dim=1)
        
    def feedforward(self, x):
        
        x = func.max_pool2d(self.conv2(func.relu(self.conv1(x))), (2,2))
        x = func.max_pool2d(func.relu(self.conv3(x)), (2,2))
        x = x.view(-1, self.flat_size(x))
        x = func.relu(self.fc1(x))
        x = func.relu(self.fc2(x))
        #x = self.SM(x)
        return x
    
    def flat_size(self, x):
        
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def train(self, training_data, epochs, mini_batch_size, optimizer, criterion, test_data=None):
       
        if test_data: 
            n_test = len(test_data)
            t_d = torch.zeros(n_test, 1, 28, 28)
            t_r = torch.zeros(n_test, dtype=torch.int64)
            for j in range(n_test):
                t_d[j][0][:][:] = test_data[j][0]
                t_r[j] = test_data[j][1]
            
        n = len(training_data)
        for i in range(epochs):                  #main training loop
            print("Training epoch {0} started.".format(i))
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size] 
                for k in range(0, n, mini_batch_size)] #make mini_batches of training data and train on them
           # a = 0
            for mini_batch in mini_batches:
               # print(a)
               # a+=1
                mb = torch.zeros(mini_batch_size, 1, 28, 28)
                tg = torch.zeros(mini_batch_size, dtype=torch.int64)
                for j in range(len(mini_batch)):
                    mb[j][0][:][:] = mini_batch[j][0]
                    tg[j] = mini_batch[j][1]
                self.batch_update(mb, tg, optimizer, criterion) #In turn send each mini batch through the training routine
            if test_data:
                print("Epoch {0} complete, evaluating".format(i))
                print("Epoch number {0} Evaluated. Test result: {1} / {2}".format( #after the whole training set has been used once, do a test on the test data if it is included
                    i, self.evaluate(t_d,t_r), n_test))
            else:
                print("Epoch {0} complete".format(i))
            
    def batch_update(self, mini_batch, target, optimizer, criterion):
        optimizer.zero_grad()
        result = self.feedforward(mini_batch)
        loss = criterion(result, target)
        loss.backward()
        optimizer.step()
        
    def evaluate(self, x, y):
        test_results = zip(self.feedforward(x).max(1)[1].tolist(), y.tolist()) 
        print(test_results)
        return sum(int(x == y) for (x, y) in test_results)
        
    
def run():
    training_data, validation_data, test_data = mnist_loader_torch.load_data_wrapper()
    net = torchnet()
    optimizer = optim.SGD(net.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    net.train(training_data, 10, 30, optimizer, criterion, test_data=test_data)

run()

