import os
import struct
import numpy as np

"""
Loosely inspired by http://abel.ee.ucla.edu/cvxopt/_downloads/mnist.py
which is GPL licensed.
"""

def read(dataset = "training", path = "."):
    """
    Python function for importing the MNIST data set.  It returns an iterator
    of 2-tuples with the first element being the label and the second element
    being a numpy.uint8 2D array of pixel data for the given image.
    """

    if dataset is "training":
        fname_img = os.path.join(path, 'train-images-idx3-ubyte.bdat')
        fname_lbl = os.path.join(path, 'train-labels-idx1-ubyte.bdat')
    elif dataset is "testing":
        fname_img = os.path.join(path, 't10k-images-idx3-ubyte.bdat')
        fname_lbl = os.path.join(path, 't10k-labels-idx1-ubyte.bdat')
    else:
        raise ValueError("dataset must be 'testing' or 'training'")

    # Load everything in some numpy arrays
    with open(fname_lbl, 'rb') as flbl:
        magic, num = struct.unpack(">II", flbl.read(8))
        lbl = np.fromfile(flbl, dtype=np.int8)

    with open(fname_img, 'rb') as fimg:
        magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
        img = np.fromfile(fimg, dtype=np.uint8).reshape(len(lbl), rows, cols)

    get_img = lambda idx: (lbl[idx], img[idx])

    # Create an iterator which returns each image in turn
    for i in range(len(lbl)):
        yield get_img(i)

from matplotlib import pyplot
import matplotlib as mpl

def show(image):
    """
    Render a given numpy.uint8 2D array of pixel data.
    """

    fig = pyplot.figure()
    ax = fig.add_subplot(1,1,1)
    imgplot = ax.imshow(image, cmap=mpl.cm.Greys)
    imgplot.set_interpolation('nearest')
    ax.xaxis.set_ticks_position('top')
    ax.yaxis.set_ticks_position('left')
    pyplot.show()

def ax_plot(ax, image):
    """Plot img with an axis handle."""

    imgplot = ax.imshow(image, cmap=mpl.cm.Greys)
    imgplot.set_interpolation('nearest')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

def show_ten(imagList):
    """Plot ten image on a figure."""

    nImages = len(imagList)

    if 10 != nImages:
        print("Wrong number of elements in list imagList.")
        return

    fig = pyplot.figure()

    for i in range(10):
        ax = fig.add_subplot(2, 5, i+1)
        ax_plot(ax, imagList[i])

    pyplot.show()


if __name__ == "__main__":
    training_data = list(read(dataset = "training"))
    print(len(training_data))
    label,img = training_data[0]
    print("label = %d" % (label))

    # for i in range(20):
    #     label,imgt = training_data[i]
    #     print("i = %d, label = %d." % (i, label))

    idx = [ 1, 3, 5, 7, 2, 0, 13, 15, 17, 4 ]
    imagList = [training_data[i][1] for i in idx]

    show_ten(imagList)
    