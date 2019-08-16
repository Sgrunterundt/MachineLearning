"""Data loader modified from the data loader included in the Deep Learning and 
Neural Nets course. Changed to format the data for a convnet using pytorch"""

# Standard library
import _pickle as cPickle
import gzip

# Third-party libraries
import numpy as np
import torch

def load_data():
    """Return the MNIST data as a tuple containing the training data,
    the validation data, and the test data.

    The ``training_data`` is returned as a tuple with two entries.
    The first entry contains the actual training images.  This is a
    numpy ndarray with 50,000 entries.  Each entry is, in turn, a
    numpy ndarray with 784 values, representing the 28 * 28 = 784
    pixels in a single MNIST image.

    The second entry in the ``training_data`` tuple is a numpy ndarray
    containing 50,000 entries.  Those entries are just the digit
    values (0...9) for the corresponding images contained in the first
    entry of the tuple.

    The ``validation_data`` and ``test_data`` are similar, except
    each contains only 10,000 images.

    This is a nice data format, but for use in neural networks it's
    helpful to modify the format of the ``training_data`` a little.
    That's done in the wrapper function ``load_data_wrapper()``, see
    below.
    """
    f = gzip.open('../data/mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = cPickle.load(f, encoding='latin1')
    f.close()
    return (training_data, validation_data, test_data)

def load_data_wrapper():
    """Return a tuple containing ``(training_data, validation_data,
    test_data)``. Based on ``load_data``, but the format is more
    convenient for use in our implementation of neural networks.

    In particular, ``training_data`` is a list containing 50,000
    2-tuples ``(x, y)``.  ``x`` is a (1,1,28,28)-dimensional torch tensor
    containing the input image.  ``y`` is a 1-dimensional long integer
    torch tensor corresponding to the
    correct digit for ``x``.

    ``validation_data`` and ``test_data`` likewise are lists containing 10,000
    images and their corresponding correct digit"""
    tr_d, va_d, te_d = load_data()
    training_inputs = [torch.from_numpy(np.reshape(x, (1, 1, 28, 28))) for x in tr_d[0]]
    training_data = list(zip(training_inputs, torch.tensor(tr_d[1])))
    validation_inputs = [torch.from_numpy(np.reshape(x, (1, 1, 28, 28))) for x in va_d[0]]
    validation_data = list(zip(validation_inputs, torch.tensor(va_d[1])))
    test_inputs = [torch.from_numpy(np.reshape(x, (1, 1, 28, 28))) for x in te_d[0]]
    test_data = list(zip(test_inputs, torch.tensor(te_d[1])))
    return (training_data, validation_data, test_data)

def vectorized_result(j):
    """Return a 10-dimensional unit vector with a 1.0 in the jth
    position and zeroes elsewhere.  This is used to convert a digit
    (0...9) into a corresponding desired output from the neural
    network. Not used in the torch version."""
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e
