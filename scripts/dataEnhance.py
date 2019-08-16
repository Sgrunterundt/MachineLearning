import torch

def dataEnhance(data, negative=True, shift=False, noise=False):
    if (negative):
        negdata = [(1-x[0]) for x in data]
        answers = [y[1] for y in data]
        data2 = list(zip(negdata, answers))
        data = data + data2
    if (shift):
        print("WARNING: Shifting data not implemented")
    if (noise):
        print("WARNING: Noise adding not implemented")
    
    return data
