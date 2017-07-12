
# imports
import importlib

import ANN

import numpy as np
import math
import matplotlib.pyplot as plt

if __name__ == "__main__":
	dataX = np.linspace(0, 2*3.14, 1000).reshape(1000, 1)
	dataY = np.sin(dataX)

	nNL     = [1, 10, 10, 1]  # Number of neurons per every hidden layer.

	actFunc      = ANN.ReLU()
	actFuncFinal = ANN.Act_dummy()

	fcann = ANN.FCANN(nNL, actFunc, actFuncFinal)

	fcann.make_random_w_b(0.1, -0.05, 0.0001)

	lossFunc = ANN.SumOfSquares()

	fcann.train(dataX, dataY, 200, 0.02,\
		randomizeData = True, showFigure = True,\
		lossFunc = lossFunc)

	pathName = "/home/yaoyu/SourceCodes/NN/SimpleANN/SavedANN"

	fcann.save_to_file(pathName)

	# Test.

	print("========== Test. ==============")

	dataX = np.linspace(0, 2*3.14, 2000)
	dataY = np.sin(dataX)

	yList = []

	for i in range(dataX.shape[0]):
		x = np.array(dataX[i]).reshape(fcann.layerDesc[ 0], 1)
		Y = np.array(dataY[i]).reshape(fcann.layerDesc[-1], 1)

		# yList = FF(x, self.wList, self.bList, self.actFunc, self.actFuncFinal)
		y = fcann.apply(x)
		yList.append(y[0][0])

		# print("y = %e, Y = %e.\n" % (yList[-1][0][0], Y[0][0]))
		print("y = %+e, Y = %+e, n_diff = %+e." % (y[0][0], Y[0][0], (y[0][0] - Y[0][0]) / Y[0][0]))

	# Make a new array.
	yArray = np.array(yList)

	# matplotlib.pyplot.
	fig, ax = plt.subplots()

	lineOriginal,  = ax.plot(dataX, dataY, 'b', label = "original")
	linePredicted, = ax.plot(dataX, yArray, 'r', label = "predicted")
	ax.legend(loc = "upper right")
	ax.set_xlabel("input")
	ax.set_ylabel("output")
	ax.set_title("Test FCANN")
	plt.show()
