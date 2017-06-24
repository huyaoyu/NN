
# Imports.

import numpy as np
import math

# Raw data.

dataX = np.linspace(0, 2*3.14, 1000)
dataY = np.sin(dataX)

nX = dataX.shape[0]

nNL     = [1, 10, 5, 1]  # Number of neurons per every hiden layer.

class ANNEx(Exception):
	"""Base exception class."""
	def __init__(self, message):
		super(ANNEx, self).__init__()
		self.message = message
		
class ArgumentEx(ANNEx):
	"""Argument exception."""
	def __init__(self, message):
		super(ArgumentEx, self).__init__(message)

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

		pypg = get_pypg(y)
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

	w_ori = 1.0
	w_new = 1.01

	b_ori = 1.0
	b_new = 1.01

	# Get wList and bList with fixed values.
	(wList, bList) = get_fixed_w_b(nNL, w_ori, b_ori)

	# Feed forward.
	yList = FF(x, wList, bList)

	# Get loss.
	L0 = loss_func(Y, yList[-1])

	# Get gradients.
	(grad_w, grad_b, grad_x) = get_gradient(wList, bList, yList, Y)

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
				yList = FF(x, wListDup, bList)

				# Get loss.
				L1 = loss_func(Y, yList[-1])

				# Calculate the approximate gradient.
				grad_w_app_single = ( L1 - L0 ) / (w_new - w_ori)

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
			yList = FF(x, wList, bListDup)

			# Get loss.
			L1 = loss_func(Y, yList[-1])

			# Calculate the approximate gradient.
			grad_b_app_single = ( L1 - L0 ) / (b_new - b_ori)

			# Shwo information.
			print("Layer %d, n = %d, g0 = %e, g1 = %e, rel_diff = %e." % (k, j, grad_b[k][j][0], grad_b_app_single, (grad_b[k][j][0] - grad_b_app_single) / grad_b_app_single ))

	# ====== End of looping for every b value. =========

	print("")
	print("Test done.\n")

def main():
	"""The main function."""

	# The learning rate.
	alpha = 0.001

	# Collection of parameters.

	(wList, bList) = get_random_w_b(nNL, 0.2, -0.1, 0.1)

	for i in range(nX):
		# Feed forward.
		x_input = np.array(dataX[i]).reshape(nNL[ 0], 1)
		y_input = np.array(dataY[i]).reshape(nNL[-1], 1)

		neList = FF(x_input, wList, bList)

		loss = loss_func(y_input, neList[-1])

		print("x = %e, y = %e, Y = %e, loss = %e" % (dataX[i], dataY[i], y_input, loss))

		# Gradient calculation.

		(grad_w, grad_b, grad_x) = get_gradient(wList, bList, neList, y_input)

		# Backscatter prapogation.

		correct_by_gradient(wList, grad_w, alpha)
		correct_by_gradient(bList, grad_b, alpha)

if __name__ == '__main__':

	# Run the main function.

	main()

	# Run the test.

	# test_get_gradient()

		




