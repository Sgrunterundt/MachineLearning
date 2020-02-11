from matplotlib.image import imread
import torch

img = imread(r"C:\users\Kasper\Documents\Machine Learning\MNIST\data\test.png")

img2 = torch.from_numpy(img)

print(img2)