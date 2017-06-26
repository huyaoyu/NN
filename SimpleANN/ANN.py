# Imports.

import numpy as np
import math
import copy
import json
import os

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

# ==================== Class definitions. ==========================

class ANNEx(Exception):
	"""Base exception class."""
	def __init__(self, typeStr, message):
		super(ANNEx, self).__init__()
		self.typeStr = typeStr
		self.message = message

	def show_message(self):
		"""
		Show the message of the exception.
		"""

		print("%s: %s" % (self.typeStr, self.message) )

class ArgumentEx(ANNEx):
	"""Argument exception."""
	def __init__(self, message):
		super(ArgumentEx, self).__init__("Argument exception", message)

class StateEx(ANNEx):
	"""docstring for StateEx"""
	def __init__(self, message):
		super(StateEx, self).__init__("State exception", message)

class IOEx(ANNEx):
	"""docstring for IOEx"""
	def __init__(self, message):
		super(IOEx, self).__init__("IO exception", message)	
		
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
		super(ReLU, self).__init__("ReLU")
		
		
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
		super(Act_dummy, self).__init__("Act_dummy")

	def apply(self, g):
		"""Dummy function."""

		return g		
	
	def derivative(self, y):
		"""Dummy function."""

		return np.array(1.0).reshape(1, 1)

class FCANN(object):
	"""docstring for FCANN"""
	def __init__(self, layerDesc = [], actFunc = 0, actFuncFinal = 0):
		super(FCANN, self).__init__()
		self.layerDesc    = copy.deepcopy(layerDesc) # Layer description, a list describe number of neurals of each layer.
		self.actFunc      = copy.deepcopy(actFunc)
		self.actFuncFinal = copy.deepcopy(actFuncFinal)

		self.trained = 0 # Not trained yet.

		self.nLayers = len(layerDesc)

		# Make default w and b for the FCANN.
		self.wList = []
		self.bList = []

		# Name.
		self.name = "fcann"

	def apply(self, x):
		"""
		Apply this FCANN to calculate an output.
		"""

		if 0 == self.trained:
			# This is an exception.

			raise StateEx("The FCANN has not been trained.")

		nW = self.nLayers - 1

		tempInput = x

		for i in range(nW):
			wT = np.transpose(self.wList[i])
			tempY = np.matmul(wT, tempInput) + self.bList[i]

			if i == nW - 1:
				tempY = self.actFuncFinal.apply(tempY)
			else:
				tempY = self.actFunc.apply(tempY)

			tempInput = tempY

		return tempY

	def save_to_file(self, dirName):
		"""
		Save this FCANN to the file system.
		dirName is the name of the directory.
		"""
		# Test the validity of the path.
		dirWithName = "%s/%s" % (dirName, self.name)

		if not os.path.isdir(dirWithName):
			os.makedirs(dirWithName)

		# Save every w matrix to a single file.
		i = 0

		for w in self.wList:
			# Make the full file name.
			fn = "%s/w%02d" % (dirWithName, i)

			np.save(fn, w)

			i = i + 1

		# Save every b vector to a single file.
		i = 0
		for b in self.bList:
			fn = "%s/b%02d" % (dirWithName, i)

			np.save(fn, b)

			i = i + 1

		# Save other member variables to json file.
		fn = "%s/%s.json" % (dirWithName, "MemberVariables")
		fp = open(fn, 'w')

		# Single dictionary.
		sd = { "name":self.name,\
			   "layerDesc":self.layerDesc, "nLayers":self.nLayers,\
			   "actFunc":self.actFunc.name, "actFuncFinal":self.actFuncFinal.name,\
			   "trained":self.trained }

		json.dump(sd, fp, indent = 0)

		fp.close()


	def load_from_file(self, dirName):
		"""
		Load a FCANN from the file system.
		dirName is the name of the directory.
		"""

		# Check the validity of dirName

		if not os.path.isdir(dirName):
			str = "The specified dirName is not valid. dirName = %s" % (dirName)
			raise IOEx(str)

		# Load the member variables.

		memberVariablesFileName = "%s/%s" % (dirName, "MemberVariables.json")

		fp = open(memberVariablesFileName, 'r')

		sd = json.load(fp)

		fp.close()

		self.parse_dictionary(sd)

		# Member variables other than wList and bList are ready.

		# Load w.
		self.load_w(dirName)

		# Load b.
		self.load_b(dirName)


	def parse_dictionary(self, sd):
		"""Parse a dictionary. Assign values to the member variables."""

		self.name      = copy.deepcopy(sd["name"])
		self.layerDesc = copy.deepcopy(sd["layerDesc"])
		self.nLayers   = sd["nLayers"]
		self.trained   = sd["trained"]

		actFuncName = sd["actFunc"]

		self.assign_activation_function(actFuncName)

		actFuncFinalName = sd["actFuncFinal"]

		self.assign_activation_function(actFuncFinalName, actFuncType = "final")

	def assign_activation_function(self, name, actFuncType = "normal"):
		"""
		Assign an activation object to a member variable.
		actFuncType - could use "normal" or "final"
		name - the name of the ativation function.
		"""

		if "ReLU" == name:
			actFunc = ReLU()
		elif "Act_tanh" == name:
			actFunc = Act_tanh()
		elif "Act_dummy" == name:
			actFunc = Act_dummy()
		else:
			s = "No activation function with the name of %s" % (name)
			raise IOEx(s)

		if "normal" == actFuncType:
			self.actFunc = actFunc
		elif "final" == actFuncType:
			self.actFuncFinal = actFunc
		else:
			s = "No activation function type of %s" % (actFuncType)
			raise IOEx(s)

	def load_w(self, dirName):
		"""Load every w from path of dirName."""

		# Check if dirName is valid.
		if not os.path.isdir(dirName):
			s = "Wrong dirName: " % (dirName)
			raise IOEx(s)

		self.wList = []

		for i in range(self.nLayers - 1):
			fn = "%s/w%02d.npy" % (dirName, i)

			w = np.load(fn)

			self.wList.append(w)

	def load_b(self, dirName):
		"""Load every b from path of dirName."""

		# Check if dirName is valid.
		if not os.path.isdir(dirName):
			s = "Wrong dirName: " % (dirName)
			raise IOEx(s)

		self.bList = []

		for i in range(self.nLayers - 1):
			fn = "%s/b%02d.npy" % (dirName, i)

			b = np.load(fn)

			self.bList.append(b)

	def make_random_w_b(self, wSpan, wStart, b):
		"""
		Make a pair of w and b. w is random.
		"""

		# Clear the original lists.
		self.wList = []
		self.bList = []

		# Create new lists.
		for i in range(self.nLayers - 1):
			tempW = np.random.rand(self.layerDesc[i], self.layerDesc[i+1]) * wSpan + wStart
			tempB = np.ones((self.layerDesc[i+1], 1)) * b

			self.wList.append(tempW)
			self.bList.append(tempB)

	def make_fixed_w_b(self, w, b):
		"""
		Create two lists that contain the w and b.
		The values of w and b are fixed by arguments w and b.
		"""

		# Clear original list.
		wList = []
		bList = []

		# Make new lists.
		for i in range(self.nLayers - 1):
			tempW = np.ones((self.layerDesc[i], self.layerDesc[i+1])) * w
			tempB = np.ones((self.layerDesc[i+1], 1)) * b

			self.wList.append(tempW)
			self.bList.append(tempB)

	def train(self, dataX, dataY, learningLoops, alpha,\
		randomizeData = False, randomizeDataFixed = False, randomizeDataFixedSeed = 7.0,\
		showFigure = False):
		"""
		Train the current FCANN.
		dataX, dataY - the input training data, stored as NumPy array.
		learningLoops - the number of loops that this training will perform on dataX and dataY.
		alpha - the learning rate.
		randomizeData - flag indicating whether the training data will be randomized.

		After the training is finished, the member variable trained will turn to 1.
		"""

		nX = dataX.shape[0]

		# Collection of parameters.
		# global display
		if True == randomizeDataFixed:
			np.random.seed(randomizeDataFixedSeed)
		
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
				x_input = np.array(dataX_r[i]).reshape(self.layerDesc[ 0], 1)
				y_input = np.array(dataY_r[i]).reshape(self.layerDesc[-1], 1)

				neList = FF(x_input, self.wList, self.bList, self.actFunc, self.actFuncFinal)

				(loss, loss2) = loss_func(y_input, neList[-1])

				# print("LL %4d, No. %4d, x = %+e, y = %+e, Y = %+e, n_loss = %+e" % (j, i, dataX_r[i], neList[-1][0][0], y_input, loss / dataY_r[i]))
				running_loss += loss2
				if i % 20 == 19:    # print every 20 mini-batches
					print('[%d, %5d] loss: %.5f' %
					(j + 1, i + 1, running_loss / 20))
					lossplot.append(running_loss / 20)
					running_loss = 0.0

				# Gradient calculation.

				(grad_w, grad_b, grad_x) = get_gradient(self.wList, self.bList, neList, y_input, self.actFunc, self.actFuncFinal)

				# Backscatter prapogation.

				correct_by_gradient(self.wList, grad_w, alpha)
				correct_by_gradient(self.bList, grad_b, alpha)

		if True == showFigure:
			import matplotlib.pyplot as plt
			lossplot = np.array(lossplot)
			lossplot = lossplot.reshape((-1,2))
			lossplot = lossplot.mean(axis=1)
			plt.plot(lossplot)
			plt.show(block = False)

		self.trained = 1

		
