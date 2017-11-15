
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

def transfer_mnist_to_lists(mnistList, nb):
	x = []
	y = []

	n = len(mnistList)

	for d in mnistList:
		x.append( d[1] / nb )
		y.append( int(d[0]) )

	return x, y

if __name__ == '__main__':
	# Obtain the training data.
	training_data = list(mnist.read(path = "./TrainingNN/"))
	testing_data  = list(mnist.read(path = "./TrainingNN/", dataset = 'testing'))
	
	nTD = len(training_data)
	nTT = len(testing_data)

	TD_x, TD_y = transfer_mnist_to_lists(training_data, 255)
	TT_x, TT_y = transfer_mnist_to_lists(testing_data,  255)

	spec = [24, 64, 240, 168]

	net = Net(spec)
	net.cuda()
	rowX = TD_x[0].shape[0]
	colX = TD_x[0].shape[1]

	learningRate      = 0.001
	batchSize         = 100
	overAllTrainCycle = 10
	testBachCountMax  = 10

	# create your optimizer
	optimizer = optim.Adam(net.parameters(), lr = learningRate)
	criterion = nn.CrossEntropyLoss()

	testBachCount        = 0
	trainingLossAcc      = 0
	currentIdxInTestData = 0

	trainingLossList = []
	testingLossList  = []

	iRange = int(nTD/batchSize)

	for j in range(overAllTrainCycle):
		idx = 0

		for i in range(iRange):
			sliceIdx = idx+batchSize

			x = np.array(TD_x[idx:sliceIdx])
			y = np.array(TD_y[idx:sliceIdx])

			input  = Variable( torch.Tensor(x)).unsqueeze(1)
			target = Variable( torch.LongTensor(y))

			input = input.cuda()
			target = target.cuda()

			# in your training loop:
			optimizer.zero_grad()   # zero the gradient buffers
			output = net(input)
			loss = criterion(output, target)
			loss.backward()
			optimizer.step()    # Does the update

			# print("j = %d, i = %d, loss = %f" % (j, i, loss.data[0]))
			trainingLossAcc += loss.data[0]

			idx += batchSize
			testBachCount += 1

			if testBachCountMax == testBachCount:
				sliceIdxTestData = currentIdxInTestData + batchSize

				testX = np.array( TT_x[currentIdxInTestData:sliceIdxTestData] )
				testY = np.array( TT_y[currentIdxInTestData:sliceIdxTestData] )

				testInput  = Variable( torch.Tensor(testX) ).unsqueeze(1)
				testTarget = Variable( torch.LongTensor(testY) )

				testInput  = testInput.cuda()
				testTarget = testTarget.cuda()

				testOutput = net(testInput)
				testLoss = criterion( testOutput, testTarget )

				trainingLossList.append( trainingLossAcc / testBachCountMax )
				testingLossList.append( testLoss.data[0] )

				print('j = %d/%d, i = %d/%d, Avg. training loss = %f, Test loss = %f' %
				 (j, overAllTrainCycle, i, iRange, trainingLossList[-1], testingLossList[-1]))

				trainingLossAcc = 0

				currentIdxInTestData = sliceIdxTestData

				if currentIdxInTestData >= nTT:
					currentIdxInTestData = 0

				testBachCount = 0

	# Save the parameters of net into file.
	savePath = '%s/%s_%s_CH%d_CH%d_L%d_L%d_EP%d_LR%.0e' % \
	(savePathBase, '20171112', 'Adam', spec[0], spec[1], spec[2], spec[3], overAllTrainCycle, learningRate)
	torch.save( net.state_dict(), savePath + '.torch' )

	# Plot the loss curves.
	plt.plot( range(len(trainingLossList)), trainingLossList )
	plt.plot( range(len(testingLossList)),  testingLossList )
	plt.savefig(savePath + '.png')
	plt.show()
