
# imports
import importlib

import ANN

import numpy as np
import math
import matplotlib.pyplot as plt

if __name__ == "__main__":
	fcann = ANN.FCANN()

	pathName = "/home/yaoyu/SourceCodes/NN/SimpleANN/SavedANN/fcann"

	try:
		fcann.load_from_file(pathName)
	except ANN.ANNEx as e:
		e.show_message()
		raise e

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
	plt.show()
