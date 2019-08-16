import sys
import torch
import torchnet
from matplotlib.image import imread
import numpy as np

if len(sys.argv) < 2:
    print("Please provide path to the image you want to read as an argument. Eg.: \"python evaluate.py ..\\data\\test.png\"")
else:

    img = imread(sys.argv[1])
    img = 1 - img

    im = torch.from_numpy(np.reshape(img[:,:,1], (1,1,28,28)))


    net = torch.load('..\data\pretrained.pth')

    a = net.feedforward(im).max(1)[1].data.tolist()[0]

    if a==0: print("Hmm, det ligner et 0.")
    else: print("Hmm, det ligner et {0}-tal".format(a))