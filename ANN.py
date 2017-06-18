
import numpy as np
import math

dataX = np.linspace(0, 2*3.14, 1000)
dataY = np.sin(dataX)

nX = dataX.shape[0]

nNL  = [1, 10, 5, 1]  # Number of neurons per every hiden layer.
nLayers = len(nNL)

def ReLU(x):
	"""Activation function."""

	y = x.copy()

	y[y<0.0] = 0.0

	return y



def FF(x, w, b):
	"""Feed forward."""

	nW = len(w)

	y = [x]

	tempInput = x
	for i in range(nW):
		wT = np.transpose(w[i])
		tempY = np.matmul(wT, tempInput) + b[i]

		tempY = ReLU(tempY)

		y.append(tempY)

		tempInput = tempY

	return y

def loss_func(y0, y1):
	"""Loss function"""

	diff = y0 - y1

	return 0.5 * np.transpose(diff).dot(diff)

if __name__ == '__main__':

	# Collection of parameters.

	wSpan  = 0.2
	wStart = -0.1

	wList = []
	bList = []

	for i in range(nLayers-1):
		tempW = np.random.rand(nNL[i], nNL[i+1]) * wSpan + wStart
		tempB = np.ones((nNL[i+1], 1)) * 0.1
		# tempW = np.ones((nNL[i], nNL[i+1]))
		# tempB = np.ones((nNL[i+1], 1)) * 1.0

		wList.append(tempW)
		bList.append(tempB)

	# Collection of neurons.
	# Feed forward.

	for i in range(nX):
		x_input = np.array(dataX[i]).reshape(nNL[ 0], 1)
		y_input = np.array(dataY[i]).reshape(nNL[-1], 1)

		neList = FF(x_input, wList, bList)

		loss = loss_func(y_input, neList[-1])

		print("x = %e, y = %e, loss = %e" % (dataX[i], dataY[i], loss))




