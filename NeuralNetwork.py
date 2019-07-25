class NeuralNetwork:
	def __init__ (self, x, y):
		self.input 	  = x
		self.weights1 = np.random.rand(self.input.shape[1], 4)
		self.weights2 = np.random.rand(4, 1)
		self.y 		  = y
		self.output   = np.zeros(y.shape)

	def feedforward(self):
		self.layer1   = sigmoid(np.dot(self.input, self.weights1))
		self.output	  = sigmoid(np.dot(self.input, self.weights2))

	def backprop(self):
		# chain rule to find derivative of loss function w.r.t. to weights1 and weights2
		d_weights2	= np.dot(self.layer1.T, 2 * (self.y - self.output) * sigmoid_derivative(self.output))
		d_weights1  = np.dot(self.input.T, (np.dot(2 * (self.y - self.output) * sigmoid_derivative(self.output), self.weights2.T) * sigmoid_derivative(self.layer1)))

		# update weights with derivate weights
		self.weights1 += d_weights1
		self.weights2 += d_weights2
