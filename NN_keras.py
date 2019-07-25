from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
import numpy as np

# Fix random seed
np.random.seed(9)

# Dummy data
X = np.array([[0,0,1],[0,1,1],[1,0,1],[1,1,1]])
y = np.array([[0], [1], [1], [0]])

model = Sequential()

# Layer 1
model.add(Dense(units = 4, activation = 'sigmoid', input_dim = 3))
# Output Layer
model.add(Dense(units = 1, activation = 'sigmoid'))
# View Structure
print(model.summary())
# Set optimizer, compile model, lr = learning rate
sgd = optimizers.SGD(lr = 1)
model.compile(loss = 'mean_squared_error', optimizer = sgd)

# Train model for 1500 iterations
model.fit(X, y, epochs = 1500, verbose = False)

print(model.predict(X))