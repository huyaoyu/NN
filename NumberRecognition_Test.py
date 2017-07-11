
# imports
import importlib

from SimpleANN import ANN

import numpy as np
import math
import matplotlib.pyplot as plt

from TrainingNN import mnist

y0 = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0]).reshape(10, 1)
y1 = np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 0]).reshape(10, 1)
y2 = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0]).reshape(10, 1)
y3 = np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0]).reshape(10, 1)
y4 = np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 0]).reshape(10, 1)
y5 = np.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0]).reshape(10, 1)
y6 = np.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0]).reshape(10, 1)
y7 = np.array([0, 0, 0, 0, 0, 0, 0, 1, 0, 0]).reshape(10, 1)
y8 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 0]).reshape(10, 1)
y9 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1]).reshape(10, 1)

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

def print_nomarlized_vector(x):
	"""Print a 1D array."""

	print("[ ", end="")

	for i in range(x.shape[0]):
		print("%.2f " % (x[i][0]), end="")

	print("]", end="")


if __name__ == "__main__":
	# Obtain the training data.
	testing_data = list(mnist.read(dataset = "testing", path = "./TrainingNN/"))
	
	nTD = len(testing_data)

	TD_x = []
	TD_y = []

	for td in testing_data:
		TD_x.append(td[1] / 255)
		TD_y.append(interpret(td[0]))

	dimInput  = TD_x[0].shape[0] * TD_x[0].shape[1]
	dimOutput = TD_y[0].shape[0] * TD_y[0].shape[1]

	print("nTD = %d, dimInput = %d, dimOutput = %d.\n" % (nTD, dimInput, dimOutput))

	pathName = "/home/yaoyu/SourceCodes/NN/SavedANN/NumberRecognition/"

	fcann = ANN.FCANN()

	try:
		fcann.load_from_file(pathName)
	except ANN.ANNEx as e:
		e.show_message()
		raise e

	accOK    = 0
	accWrong = 0

	for i in range(nTD):
	# for i in range(20):
		y = fcann.apply(TD_x[i].reshape(dimInput, 1))
		print("[%d]" % (i))
		print("Y = ", end = "")
		print_nomarlized_vector(TD_y[i])
		numberY_true = TD_y[i].argmax()
		print(" %d" % (numberY_true) )
		print("y = ", end = "")
		sy = ANN.Softmax(y)
		print_nomarlized_vector(sy)
		numberY_pred = sy.argmax()
		print(" %d" % (numberY_pred), end="")

		if numberY_true == numberY_pred:
			print(" OK\n")
			accOK = accOK + 1
		else:
			print(" Wrong\n")
			accWrong = accWrong + 1

	print("accOK = %d, accWrong = %d, ratio = %.2f" % (accOK, accWrong, accOK/nTD))
