
# Imports.

import numpy as np
import math

# Raw data.

dataX = np.linspace(0, 2*3.14, 1000)
dataY = np.sin(dataX)

nNL     = [1, 10, 10, 1]  # Number of neurons per every hiden layer.
# display = False

class ANNEx(Exception):
	"""Base exception class."""
	def __init__(self, message):
		super(ANNEx, self).__init__()
		self.message = message

class ArgumentEx(ANNEx):
	"""Argument exception."""
	def __init__(self, message):
		super(ArgumentEx, self).__init__(message)

class ActivationFunc(object):
	"""docstring for ActivationFunc"""
	def __init__(self, name):
		super(ActivationFunc, self).__init__()

		self.name = name

	def apply(self, g):
		"""Apply this activation function."""

		return g

	def derivative(self, y):
		"""pass"""

		return y

class ReLU(ActivationFunc):
	"""docstring for ReLU"""
	def __init__(self):
		super(ReLU, self).__init__("Sigmoid")
		
		
	def apply(self, g):
		"""ReLU"""

		y = g.copy()

		y[y<0.0] = 0.0

		return y

	def derivative(self, y):
		"""Derivative of ReLU."""

		temp = np.zeros_like(y) # All the values are zero.

		temp[y > 0] = 1.0

		return temp

class Act_tanh(ActivationFunc):
	"""docstring for Sigmoid"""
	def __init__(self):
		super(Act_tanh, self).__init__("Act_tanh")

	def apply(self, g):
		"""Sigmoid."""

		return np.tanh(g)

	def derivative(self, y):
		"""Derivative of Sigmoid function."""

		return 1.0 - y * y

class Act_dummy(ActivationFunc):
	"""docstring for Act_dummy"""
	def __init__(self):
		super(Act_dummy, self).__init__("Dummy")

	def apply(self, g):
		"""Dummy function."""

		return g		
	
	def derivative(self, y):
		"""Dummy function."""

		return np.array(1.0).reshape(1, 1)

def get_random_w_b(nNL, wSpan, wStart, b):
	"""
	Create two lists that contain the w and b.
	The values of w are random.
	The values of b are fixed by b.
	"""

	nLayers = len(nNL)

	wList = []
	bList = []

	for i in range(nLayers-1):
		tempW = np.random.rand(nNL[i], nNL[i+1]) * wSpan + wStart
		tempB = np.ones((nNL[i+1], 1)) * b

		wList.append(tempW)
		bList.append(tempB)

	return wList, bList

def get_fixed_w_b(nNL, w, b):
	"""
	Create two lists that contain the w and b.
	The values of w and b are fixed by arguments w and b.
	"""

	nLayers = len(nNL)

	wList = []
	bList = []

	for i in range(nLayers - 1):
		tempW = np.ones((nNL[i], nNL[i+1])) * w
		tempB = np.ones((nNL[i+1], 1)) * b

		wList.append(tempW)
		bList.append(tempB)

	return wList, bList

def duplicate_list_of_numpy_elements(fList):
	"""
	Duplicate a list of numpy elements.
	"""

	temp = []

	nElements = len(fList)

	for e in fList:
		tempE = np.array(e, copy=True)

		temp.append(tempE)

	return temp

def FF(x, w, b, actFunc, actFuncFinal):
	"""Feed forward."""

	nW = len(w)

	y = [x]

	tempInput = x

	for i in range(nW):
		wT = np.transpose(w[i])
		tempY = np.matmul(wT, tempInput) + b[i]

		if i == nW - 1:
			tempY = actFuncFinal.apply(tempY)
		else:
			tempY = actFunc.apply(tempY)

		y.append(tempY)

		tempInput = tempY

	return y

def loss_func(y0, y1):
	"""Loss function"""

	diff = y0 - y1

	dp = np.transpose(diff).dot(diff)

	return 0.5 * dp, np.sqrt(dp)

def get_pypg(y, actFunc):
	"""
	Calculate partial y / partial g.
	"""

	return actFunc.derivative(y)

def get_pypg_final(y, actFunc):
	"""
	Calculate partial y / partial g for the final ouput.
	"""

	return actFunc.derivative(y)

def get_pLpg(pLpy, pypg):
	"""
	Calculate partial L / partial g.
	"""
	# global display
	if pLpy.shape[0] != pypg.shape[0]:
		print("pLpg: Dimensions of pLpy and pypg do not match.\npLpy.shape[0] = %d, pypg.shape[0] = %d\n" % ( pLpy.shape[0], pypg.shape[0] ) )
		raise ANNEx("Argument error.")
	# if display:
	# 	print '  -- pLpy: ', pLpy.T
	# 	print '  -- pypg: ', pypg.T
	temp = pLpy * pypg

	return temp

def get_gradient(wList, bList, yList, Y, actFunc, actFuncFinal):
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
	grad_x = [] # This will have an extra element than grad_w and grrad_b.
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

		if j == J:
			pypg = get_pypg_final(y, actFuncFinal)
		else:
			pypg = get_pypg(y, actFunc)

		pLpg = get_pLpg(pLpy, pypg)

		pLpw = np.matmul(x, pLpg.transpose())
		grad_w.append(pLpw)

		grad_b.append(pLpg)

		pLpx = np.matmul(w, pLpg)
		grad_x.append(pLpx)

		pLpy = pLpx

	return grad_w[::-1], grad_b[::-1], grad_x[::-1]

def correct_by_gradient(mList, gradList, alpha):
	"""
	Correct every element of mList by grad and learning rate alpha.
	mList contains the parameters.
	gradList contains the gradients.
	alpha is the learning rate factor. Should be positive.

	For an element m in mList and associated gradient element g in gradList,

	m = m - alpha * g

	"""

	n = len(mList)

	for i in range(n):
		m = mList[i]
		g = gradList[i]

		m = m - alpha * g

		mList[i] = m

def test_get_gradient():
	"""
	Test the function get_gradient().
	"""

	x = np.array(1.0).reshape(1, 1)
	Y = np.array(1.0).reshape(1, 1) # The expected Y value.

	w_ori   = 2.0
	w_delta = 0.01
	w_new   = w_ori + w_delta
	
	b_ori   = 1.0
	b_delta = 0.01
	b_new   = b_ori + b_delta

	# Get wList and bList with fixed values.
	(wList, bList) = get_fixed_w_b(nNL, w_ori, b_ori)

	# Activation functions.
	actFunc      = ReLU()
	actFuncFinal = ReLU()
	# actFunc      = Act_tanh()
	# actFuncFinal = Act_tanh()

	# Feed forward.
	yList = FF(x, wList, bList, actFunc, actFuncFinal)

	# Get loss.
	L0 = loss_func(Y, yList[-1])

	# Get gradients.
	(grad_w, grad_b, grad_x) = get_gradient(wList, bList, yList, Y, actFunc, actFuncFinal)

	# ====== Start looping for every w value. ==========

	# Get the number of layers.
	nLayers = len(nNL)

	print("\n=========== Test of w. ===========\n")

	for k in range(nLayers-1):
		# Get the value fo M and N of the current layer or neural network.
		(M, N) = wList[k].shape

		print("Layer %d, M = %d, N = %d:" % (k, M, N))

		for i in range(M):
			for j in range(N):
				# Make a copy of wList.
				wListDup = duplicate_list_of_numpy_elements(wList)

				# Modify w a little bit.
				wListDup[k][i][j] = w_new

				# Feed forward again.
				yList = FF(x, wListDup, bList, actFunc, actFuncFinal)

				# Get loss.
				L1 = loss_func(Y, yList[-1])

				# Calculate the approximate gradient.
				grad_w_app_single = ( L1 - L0 ) / w_delta

				# Shwo information.
				print("Layer %d, m = %d, n = %d, g0 = %e, g1 = %e, rel_diff = %e." % (k, i, j, grad_w[k][i][j], grad_w_app_single, (grad_w[k][i][j] - grad_w_app_single) / grad_w_app_single ))

	# ====== End of looping for every w value. =========

	# ====== Start looping for every b value. ==========

	print("\n=========== Test of b. ===========\n")

	for k in range(nLayers-1):
		# Get the value fo M and N of the current layer or neural network.
		(M, N) = wList[k].shape

		print("Layer %d, N = %d:" % (k, N))

		for j in range(N):
			# Make a copy of wList.
			bListDup = duplicate_list_of_numpy_elements(bList)

			# Modify w a little bit.
			bListDup[k][j][0] = b_new

			# Feed forward again.
			yList = FF(x, wList, bListDup, actFunc, actFuncFinal)

			# Get loss.
			L1 = loss_func(Y, yList[-1])

			# Calculate the approximate gradient.
			grad_b_app_single = ( L1 - L0 ) / b_delta

			# Shwo information.
			print("Layer %d, n = %d, g0 = %e, g1 = %e, rel_diff = %e." % (k, j, grad_b[k][j][0], grad_b_app_single, (grad_b[k][j][0] - grad_b_app_single) / grad_b_app_single ))

	# ====== End of looping for every b value. =========

	print("")
	print("Test done.\n")

def main():
	"""The main function."""

	nX = dataX.shape[0]

	# The learning rate.
	# Collection of parameters.
	# global display
	np.random.seed(7)

	(wList, bList) = get_random_w_b(nNL, 0.1, -0.05, 0.0001)

	# Activation functons.
	# actFunc      = ReLU()
	# actFuncFinal = ReLU()
	actFunc      = ReLU()
	actFuncFinal = Act_dummy()

	alpha = 0.01

	learningLoops = 500
	lossplot = []
	for j in range(learningLoops):
		print("========== LP = %d. ===============\n" % (j))

		randIdx = np.random.permutation(len(dataX))
		dataX_r = dataX[randIdx]
		dataY_r = dataY[randIdx]

		# dataX_r = dataX
		# dataY_r = dataY
		running_loss=0
		for i in range(nX):
			# if j==9 and i>700:
			# 	display=True
				# print '!!!'
			# Feed forward.
			x_input = np.array(dataX_r[i]).reshape(nNL[ 0], 1)
			y_input = np.array(dataY_r[i]).reshape(nNL[-1], 1)

			neList = FF(x_input, wList, bList, actFunc, actFuncFinal)

			(loss, loss2) = loss_func(y_input, neList[-1])

			# print("LL %4d, No. %4d, x = %+e, y = %+e, Y = %+e, n_loss = %+e" % (j, i, dataX_r[i], neList[-1][0][0], y_input, loss / dataY_r[i]))
			running_loss += loss2
			if i % 20 == 19:    # print every 20 mini-batches
				print('[%d, %5d] loss: %.5f' %
				(j + 1, i + 1, running_loss / 20))
				lossplot.append(running_loss / 20)
				running_loss = 0.0

			# Gradient calculation.

			(grad_w, grad_b, grad_x) = get_gradient(wList, bList, neList, y_input, actFunc, actFuncFinal)

			# Backscatter prapogation.

			correct_by_gradient(wList, grad_w, alpha)
			correct_by_gradient(bList, grad_b, alpha)

	import matplotlib.pyplot as plt
	lossplot = np.array(lossplot)
	lossplot = lossplot.reshape((-1,2))
	lossplot = lossplot.mean(axis=1)
	plt.plot(lossplot)
	plt.show()

	# Test.

	x = np.array(0.5).reshape(nNL[ 0], 1)
	Y = np.array(math.sin(0.5)).reshape(nNL[-1], 1)

	yList = FF(x, wList, bList, actFunc, actFuncFinal)


	print("Test.")

	print("y = %e, Y = %e.\n" % (yList[-1][0][0], Y[0][0]))

if __name__ == '__main__':

	# Run the main function.

	main()

	# Run the test.

	# test_get_gradient()

		
