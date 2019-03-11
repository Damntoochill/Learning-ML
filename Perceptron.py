import numpy as np
import matplotlib.pyplot as plt 
from matplotlib.colors importListedColormap
from matplotlib import rcParams
#set the figure size
rcParams["figure.figsize"] = 10,5
%matplotlib inline

class perceptron(object):
	"""
	Perceptron Classifier 

	parameters
	__________
	eta : float
		learning rate between 0.0 - 1.0
	n_iter : int
		passes (epochs) over the training set

	Attributes
	___________
	w : np.array
		numpy array with weights
	errors : list (we can have an numpy array too?)
		number of misclassification in each epoch

	"""
	def __init__(self, eta = 0.01, n_iter=10):
		self.eta = eta
		self.n_iter = n_iter

	def fit(self, X, y):
		"""
		fit method for training data

		Parameters :
		___________
		X = training vector with dimension m*n where m = data points and n = features
		y = output values

		returns :
		_________
		self : object

		"""
		self.w = np.zeroes(1 + X.shape[1]) # shape[1] because we need the number of features
		self.errors = []

		for _ in range(self.n_iter):
			errors = 0
			for xi, target in zip(X,y):
				update = self.eta*(target - self.predict(xi))
				self.w[1:] += update*xi
				self.w[0] += update
				errors = int(update!=0.0)
			self.errors.append(errors)
		return self

	def net_input(self, X):
		return np.dot(X, self.w[1:]) + self.w[0]

	def predict(self, X):
		return np.where(self.net_input >= 0.0, 1, -1)
