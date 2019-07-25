import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def sigmoid(x):
	return 1.0/(1 + np.exp(-x))

def sigmoid_derivative(x):
	return x * (1.0 - x)

class NeuralNetwork:
	def __init__ (self, x, y):
		self.input 	  = x
		self.weights1 = np.random.rand(self.input.shape[1], 4)
		self.weights2 = np.random.rand(4, 1)
		self.y 		  = y
		self.output   = np.zeros(self.y.shape)

	def feedforward(self):
		self.layer1   = sigmoid(np.dot(self.input, self.weights1))
		self.output	  = sigmoid(np.dot(self.layer1, self.weights2))

	def backprop(self):
		# chain rule to find derivative of loss function w.r.t. to weights1 and weights2
		d_weights2	= np.dot(self.layer1.T, 2 * (self.y - self.output) * sigmoid_derivative(self.output))
		d_weights1  = np.dot(self.input.T, (np.dot(2 * (self.y - self.output) * sigmoid_derivative(self.output), self.weights2.T) * sigmoid_derivative(self.layer1)))

		# update weights with derivate weights
		self.weights1 += d_weights1
		self.weights2 += d_weights2

if __name__ == "__main__":
	
	#==================================================== Chapter 1
	# X = np.array([[0,0,1], [0,1,1], [1,0,1], [1,1,1]])
	# y = np.array([[0], [1], [1], [0]])
	# nn = NeuralNetwork(X, y)

	# for i in range(1500):
	# 	nn.feedforward()
	# 	nn.backprop()

	# print(nn.output)

	#==================================================== Chapter 2
	URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
	df  = pd.read_csv(URL, names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class'])

	# print(df.info())
	# print(df.head(10))

	# df2 = df.loc[df['sepal_length'] > 5.0, ]
	# print(df2.head(10))

	# marker_shapes = ['.','^', '*']
	
	# ax = plt.axes()
	# for i, species in enumerate(df['class'].unique()):
	# 	species_data = df[df['class'] == species]
	# 	species_data.plot.scatter(x = 'sepal_length',
	# 		y = 'sepal_width',
	# 		marker = marker_shapes[i], 
	# 		s = 100, 
	# 		title = "Sepal Width vs Length by Species", 
	# 		label = species, 
	# 		figsize = (10, 7),
	# 		ax = ax)
	# plt.show()

	# df['petal_length'].plot.hist(title='Histogram of Petal Length')
	# plt.show()

	# df.plot.box(title='Boxplot of Sepal Length & Width, and Petal Length & Width')
	# plt.show()

	# Encoding
	# df2 = pd.DataFrame({'Day': ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']})
	# print(pd.get_dummies(df2))

	# Inputting Missing Values: Remove some values
	random_index = np.random.choice(df.index, replace=False, size=10)
	df.loc[random_index,'sepal_length'] = None
	print(df.isnull().any())

	# Drop NA method
	print("Number of rows before deleting: %d" % (df.shape[0]))
	df2 = df.dropna()
	print("Number of rows after deleting: %d" % (df2.shape[0]))

	# Replace missing with means
	df.sepal_length = df.sepal_length.fillna(df.sepal_length.mean())
	print(df.isnull().any())