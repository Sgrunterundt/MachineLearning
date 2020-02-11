import torchnet
import sys
import mnist_loader_torch
import torch
import torch.nn as nn
import torch.optim as optim
import os

if len(sys.argv) < 2:
    print("Please provide path to a .pth file of the network you want to train as an argument when calling this script. Eg.: \"python train.py ..\\data\\pretrained.pth\". If file does not exist a new network will be created, trained and saved under that name.")
else:

    training_data, validation_data, test_data = mnist_loader_torch.load_data_wrapper()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #device = torch.device("cpu")
    print(device)
    path = sys.argv[1]
    if os.path.isfile(path):
        net = torch.load(path).to(device)
    else:
        net = torchnet.torchnet().to(device)
    
    optimizer = optim.SGD(net.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    net.train(training_data, 10, 30, optimizer, criterion, device, test_data=test_data[0:1000])
    torch.save(net, path)