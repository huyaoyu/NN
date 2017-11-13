
# Python specific modules
import importlib

# Tools.
import numpy as np
import math
import matplotlib.pyplot as plt

# Project specific modules.
from TrainingNN import mnist

# PyTorch
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# =========== File-wide variables. ==================

# savePath = './SavedCNN/CNNNet.torch'
# savePath = './SavedCNN/CNNNet_Adam.torch'
savePathBase = './SavedCNN'

y0 = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype = float).reshape(10, 1)
y1 = np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 0], dtype = float).reshape(10, 1)
y2 = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0], dtype = float).reshape(10, 1)
y3 = np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0], dtype = float).reshape(10, 1)
y4 = np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 0], dtype = float).reshape(10, 1)
y5 = np.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0], dtype = float).reshape(10, 1)
y6 = np.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0], dtype = float).reshape(10, 1)
y7 = np.array([0, 0, 0, 0, 0, 0, 0, 1, 0, 0], dtype = float).reshape(10, 1)
y8 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 0], dtype = float).reshape(10, 1)
y9 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1], dtype = float).reshape(10, 1)

# y0 = Variable( torch.Tensor(y0) )
# y1 = Variable( torch.Tensor(y1) )
# y2 = Variable( torch.Tensor(y2) )
# y3 = Variable( torch.Tensor(y3) )
# y4 = Variable( torch.Tensor(y4) )
# y5 = Variable( torch.Tensor(y5) )
# y6 = Variable( torch.Tensor(y6) )
# y7 = Variable( torch.Tensor(y7) )
# y8 = Variable( torch.Tensor(y8) )
# y9 = Variable( torch.Tensor(y9) )

# ================= Functions. =====================

def interpret(y):
	"""Interpret y into a numpy ndarray."""

	if 0 == y:
		r = y0
	elif 1 == y:
		r = y1
	elif 2 == y:
		r = y2
	elif 3 == y:
		r = y3
	elif 4 == y:
		r = y4
	elif 5 == y:
		r = y5
	elif 6 == y:
		r = y6
	elif 7 == y:
		r = y7
	elif 8 == y:
		r = y8
	elif 9 == y:
		r = y9

	return r

class Net(nn.Module):
    def __init__(self, spec):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, spec[0], 5)
        self.conv2 = nn.Conv2d(spec[0], spec[1], 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(spec[1] * 4 * 4, spec[2])
        self.fc2 = nn.Linear(spec[2], spec[3])
        self.fc3 = nn.Linear(spec[3], 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

if __name__ == '__main__':
	# Obtain the training data.
	# training_data = list(mnist.read(path = "./TrainingNN/"))
	training_data = list(mnist.read(path = "./TrainingNN/", dataset = 'testing'))
	
	nTD = len(training_data)

	# Organize the training data.
	TD_x = []
	TD_y = []

	for td in training_data:
		TD_x.append(td[1] / 255)      # Normalize.
		# TD_y.append(interpret(td[0]))
		TD_y.append(int(td[0]))

		# spec = [6, 16, 120, 84]
	spec = [24, 64, 240, 168]

	net = Net(spec)

	overAllTrainCycle, learningRate = 10, 0.001

	savePath = '%s/%s_%s_CH%d_CH%d_L%d_L%d_EP%d_LR%.0e' % \
	(savePathBase, '20171112', 'Adam', spec[0], spec[1], spec[2], spec[3], overAllTrainCycle, learningRate)

	net.load_state_dict( torch.load( savePath + '.torch' ) )
	net.cuda()
	rowX = TD_x[0].shape[0]
	colX = TD_x[0].shape[1]

	nonvalidCount = 0

	for i in range(nTD):
		x = TD_x[i]
		y = TD_y[i]

		input  = Variable( torch.Tensor(x) ).unsqueeze(0).unsqueeze(0)
		# target = Variable( torch.LongTensor([y]) )

		input = input.cuda()
		# target = target.cuda()

		output = net(input)

		p, idx = output.data.max(1)

		if y != idx[0]:
			nonvalidCount += 1
		
		print("i = %d, y = %d, idx = %d, nvc = %d, pnvc = %f" % (i, y, idx[0], nonvalidCount, 100.0 * nonvalidCount / nTD))

