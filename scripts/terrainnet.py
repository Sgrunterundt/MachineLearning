import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as func
import torch.optim as optim


class terrainnet(nn.Module):
    
    def __init__(self):
        """Define the layers of the network. 3 convolutional layers followed by 2 fully connected"""
        super(terrainnet, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 10, 3)
        self.conv2 = nn.Conv2d(10, 20, 3)
        self.conv3 = nn.Conv2d(20, 40, 3)
        self.conv4 = nn.Conv2d(40, 80, 3)
        self.conv5 = nn.Conv2d(80, 80, 3)
        self.conv6 = nn.Conv2d(80, 80, 3)
        self.conv7 = nn.Conv2d(80, 80, 3)
        self.conv8 = nn.Conv2d(80, 80, 3)
        self.conv9 = nn.Conv2d(160, 80, 3)
        self.conv10 = nn.Conv2d(160, 80, 3)
        self.conv11 = nn.Conv2d(160, 80, 3)
        self.conv12 = nn.Conv2d(160, 80, 3)
        self.conv13 = nn.Conv2d(120, 40, 3)
        self.conv14 = nn.Conv2d(60, 20, 3)
        self.conv15 = nn.Conv2d(30, 10, 3)
        self.conv16 = nn.Conv2d(11, 1, 3)
        
        
        
    def feedforward(self, x):
        
        x1 = func.max_pool2d(func.relu(self.conv1(x)), (2,2))
        x2 = func.max_pool2d(func.relu(self.conv2(x)), (2,2))
        x3 = func.max_pool2d(func.relu(self.conv3(x)), (2,2))
        x4 = func.max_pool2d(func.relu(self.conv4(x)), (2,2))
        x5 = func.max_pool2d(func.relu(self.conv5(x)), (2,2))
        x6 = func.max_pool2d(func.relu(self.conv6(x)), (2,2))
        x7 = func.max_pool2d(func.relu(self.conv7(x)), (2,2))
        x8 = func.max_pool2d(func.relu(self.conv8(x)), (2,2))
        
        
        return x
    
    def train(self, training_data, epochs, mini_batch_size, optimizer, criterion, device, test_data=None):
       
        if test_data: 
            n_test = len(test_data)
            t_d = torch.zeros(n_test, 1, 28, 28)
            t_r = torch.zeros(n_test, dtype=torch.int64)
            for j in range(n_test):
                t_d[j][0][:][:] = test_data[j][0] #there must be a better way...
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
                torch.cuda.empty_cache()
                mb = torch.zeros(mini_batch_size, 1, 28, 28)
                tg = torch.zeros(mini_batch_size, dtype=torch.int64)
                for j in range(len(mini_batch)):
                    mb[j][0][:][:] = mini_batch[j][0]
                    tg[j] = mini_batch[j][1]
                self.batch_update(mb.to(device), tg.to(device), optimizer, criterion) #In turn send each mini batch through the training routine
            torch.cuda.empty_cache()
            if test_data:
                print("Epoch {0} complete, evaluating".format(i))
                print("Epoch number {0} Evaluated. Test result: {1} / {2}".format( #after the whole training set has been used once, do a test on the test data if it is included
                    i, self.evaluate(t_d.to(device),t_r.to(device)), n_test))
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
        return sum(int(x == y) for (x, y) in test_results)
        
    

    

