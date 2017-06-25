
# imports
import importlib

import ANN

import numpy as np
import math

if __name__ == "__main__":
	dataX = np.linspace(0, 2*3.14, 1000)
	dataY = np.sin(dataX)

	nNL     = [1, 10, 10, 1]  # Number of neurons per every hiden layer.

	actFunc      = ANN.ReLU()
	actFuncFinal = ANN.Act_dummy()

	fcann = ANN.FCANN(nNL, actFunc, actFuncFinal)

	fcann.make_random_w_b(0.1, -0.05, 0.0001)

	fcann.train(dataX, dataY, 500, 0.01, randomizeData = True)

	# Test.

	print("========== Test. ==============")

	for i in range(dataX.shape[0]):
		x = np.array(dataX[i]).reshape(fcann.layerDesc[ 0], 1)
		Y = np.array(dataY[i]).reshape(fcann.layerDesc[-1], 1)

		# yList = FF(x, self.wList, self.bList, self.actFunc, self.actFuncFinal)
		y = fcann.apply(x)

		# print("y = %e, Y = %e.\n" % (yList[-1][0][0], Y[0][0]))
		print("y = %+e, Y = %+e, n_diff = %+e." % (y[0][0], Y[0][0], (y[0][0] - Y[0][0]) / Y[0][0]))
