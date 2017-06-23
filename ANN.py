
# Imports.

import numpy as np
import math

# Raw data.

dataX = np.linspace(0, 2*3.14, 1000)
dataY = np.sin(dataX)

nX = dataX.shape[0]

nNL     = [1, 10, 5, 1]  # Number of neurons per every hiden layer.
nLayers = len(nNL)

class ANNEx(Exception):
	"""Base exception class."""
	def __init__(self, message):
		super(ANNEx, self).__init__()
		self.message = message
		
class ArgumentEx(ANNEx):
	"""Argument exception."""
	def __init__(self, message):
		super(ArgumentEx, self).__init__(message)

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

def get_pypg(y):
	"""
	Calculate partial y / partial g.
	"""
	temp = np.empty_like(y)

	for i in range(y.shape[0]):
		if y[i] >= 0 :
			temp[i] = 1.0
		else:
			temp[i] = 0.0

	return temp

def get_pLpg(pLpy, pypg):
	"""
	Calculate partial L / partial g.
	"""
	if pLpy.shape[0] != pypg.shape[0]:
		print("pLpg: Dimensions of pLpy and pypg do not match.\npLpy.shape[0] = %d, pypg.shape[0] = %d\n" % ( pLpy.shape[0], pypg.shape[0] ) )
		raise ANNEx("Argument error.")

	temp = pLpy * pypg

	return temp

def get_gradient(wList, bList, yList, Y):
	"""
	The number of pairs of neural networks is J.
	wList, bList both have J elements.
	yList has J + 1 elements with the first element being
	the x_input.
	"""

	# Get the number of pairs of neural networks.

	J   = len(wList)
	J_b = len(bList)
	J_y = len(yList)

	if J != J_b or J != J_y - 1:
		print("Dimensions of wList, bList and yList do not math.\nJ = %d, J_b = %b, J_y = %d" % (J, J_b, J_y))
		raise ANNEx("Argument error.")

	# The empty graient list.
	grad_x = []
	grad_w = []
	grad_b = []

	# pLpy with j = J.
	pLpy = yList[-1] - Y

	# Loop from J to 0.
	for j in range(J, 0, -1):
		idx = j-1

		x = yList[j-1]
		y = yList[j]
		
		w = wList[idx]
		b = bList[idx]

		pypg = get_pypg(y)
		pLpg = get_pLpg(pLpy, pypg)

		pLpw = np.matmul(x, pLpg.transpose())
		grad_w.append(pLpw)

		grad_b.append(pLpg)

		pLpx = np.matmul(w, pLpg)
		grad_x.append(pLpx)

		pLpy = pLpx

	return grad_w, grad_b, grad_x

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

	for i in range(nX):
		# Feed forward.
		x_input = np.array(dataX[i]).reshape(nNL[ 0], 1)
		y_input = np.array(dataY[i]).reshape(nNL[-1], 1)

		neList = FF(x_input, wList, bList)

		loss = loss_func(y_input, neList[-1])

		print("x = %e, y = %e, loss = %e" % (dataX[i], dataY[i], loss))

		# Backscatter prapogation.

		




