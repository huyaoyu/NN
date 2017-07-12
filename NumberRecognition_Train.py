
# Description
# ===========
#
# This is the script for training the trained FCANN for number recognition.
#
# Author
# ======
#
# Yaoyu Hu <huyaoyu@sjtu.edu.cn>
#
# Date
# ====
#
# Created: 2017-07-11
#

# =============== Imports. =================

# Python specific modules
import importlib

# Tools.
import numpy as np
import math
import matplotlib.pyplot as plt

# Project specific modules.
from SimpleANN import ANN
from TrainingNN import mnist

# =========== File-wide variables. ==================

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

# ========== Self test. ================

if __name__ == "__main__":
	# Obtain the training data.
	training_data = list(mnist.read(path = "./TrainingNN/"))
	
	nTD = len(training_data)

	# Organize the training data.
	TD_x = []
	TD_y = []

	for td in training_data:
		TD_x.append(td[1] / 255)      # Normalize.
		TD_y.append(interpret(td[0]))

	dimInput  = TD_x[0].shape[0] * TD_x[0].shape[1]
	dimOutput = TD_y[0].shape[0] * TD_y[0].shape[1]

	print("nTD = %d, dimInput = %d, dimOutput = %d.\n" % (nTD, dimInput, dimOutput))

	# Define the ANN.

	nNL     = [dimInput, 100, 100, dimOutput]  # Number of neurons per every hidden layer.

	actFunc      = ANN.ReLU()
	# actFuncFinal = ANN.Act_dummy()
	actFuncFinal = ANN.ReLU()

	fcann = ANN.FCANN(nNL, actFunc, actFuncFinal)
	fcann.name = "NumberRecognition2"

	fcann.make_random_w_b(0.01, 0.0, 0.001)

	# Train the ANN.

	# lossFunc = ANN.SumOfSquares()
	lossFunc = ANN.CrossEntropy()

	fcann.train(TD_x, TD_y, 2, 0.01,\
		randomizeData = False, showFigure = True,\
		lossFunc = lossFunc)

	# Save the trained ANN.

	pathName = "/home/yaoyu/SourceCodes/NN/SavedANN/"

	fcann.save_to_file(pathName)

