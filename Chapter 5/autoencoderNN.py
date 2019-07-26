# Autoencoder NN
# Form on unsupervised learning
# Goal: learn a latent reprentaiton of the input - usualkly a compress representation of the input
# Encoder and Decoder:
# 	Encorder encodes the input to a learned compressed representation
#	Decoder reconstructs the original input using the compressed representation

# Latent Representation
# The most salient representation of the input. The most relevant characteristics of the input
# Ex. For cats and dogs -> Latent Representation could be: Shape of Ears, Whiskers, Snout Size, Tongue
# With the latent representation we can do the following:
# 	Reduce dimensionality of the input data.
#	Remove any noise from the input data (denoising)

# Data Compression
# Autoencoders are poor at generalized compression (img and audio) because the latent representation
#	is only based on the data on which it was trained. So only works well for similar images
# Lossy form of data comrpession, output has less info than input
import random

from matplotlib import pyplot as plt
from keras.datasets import mnist # handwritten digits 28x28 px
from keras.models import Sequential
from keras.layers import Dense

training_set, testing_set = mnist.load_data()

X_train, y_train = training_set
X_test, y_test = testing_set

# Preprocess Data
# Reshape 28x28 to 784x1: X_train.shape[0] is the number of samples
X_train_reshaped = X_train.reshape((X_train.shape[0], X_train.shape[1]*X_train.shape[2]))
X_test_reshaped  = X_test.reshape((X_test.shape[0], X_test.shape[1]*X_test.shape[2]))

# Normalize vector between 0 and 1
X_train_reshaped = X_train_reshaped/255.
X_test_reshaped = X_test_reshaped/255.

# Plot some didigts to visualize
# fig, ((ax1, ax2, ax3, ax4, ax5), (ax6, ax7, ax8, ax9, ax10)) = plt.subplots(2, 5, figsize=(10,5))
 
# for idx, ax in enumerate([ax1,ax2,ax3,ax4,ax5, ax6,ax7,ax8,ax9,ax10]):
# 	for i in range(1000):
# 		if y_test[i] == idx:
# 			ax.imshow(X_test[i], cmap='gray')
# 			ax.grid(False)
# 			ax.set_xticks([])
# 			ax.set_yticks([])
# 			break

# plt.tight_layout()
#plt.show()

# Build a simple autoencoder

# Hidden layer:
# • Should be of a smaller dimension than the input data
# • Sufficiently small enough to represent a compressed representation of the input features
# • Sufficiently large enough for the decoder to reconstruct the original input without too much loss
# • Hyperparameter that needs to be selected carefully

def create_basic_autoencoder(hidden_layer_size):
	model = Sequential()
	# Add hidden layer
	model.add(Dense(units = hidden_layer_size, input_shape=(784,), activation = 'relu'))
	# Add output layer 784 = 28x28 px
	model.add(Dense(units = 784, activation='sigmoid'))

	return model

model = create_basic_autoencoder(hidden_layer_size = 1)
# Create models with different hidden layer sizes
hiddenLayerSize_2_model = create_basic_autoencoder(hidden_layer_size=2)
hiddenLayerSize_4_model = create_basic_autoencoder(hidden_layer_size=4)
hiddenLayerSize_8_model = create_basic_autoencoder(hidden_layer_size=8)
hiddenLayerSize_16_model = create_basic_autoencoder(hidden_layer_size=16)
hiddenLayerSize_32_model = create_basic_autoencoder(hidden_layer_size=32)

# Compile models
model.compile(optimizer='adam', loss='mean_squared_error')
hiddenLayerSize_2_model.compile(optimizer='adam', loss='mean_squared_error')
hiddenLayerSize_4_model.compile(optimizer='adam', loss='mean_squared_error')
hiddenLayerSize_8_model.compile(optimizer='adam', loss='mean_squared_error')
hiddenLayerSize_16_model.compile(optimizer='adam', loss='mean_squared_error')
hiddenLayerSize_32_model.compile(optimizer='adam', loss='mean_squared_error')


# Train autoencoder
# 10 Epochs, Input = Output because we trying to train it to produce identical to output
model.fit(X_train_reshaped, X_train_reshaped, epochs = 10, verbose = 0)

# Train other models
hiddenLayerSize_2_model.fit(X_train_reshaped, X_train_reshaped, epochs = 10, verbose = 0)
hiddenLayerSize_4_model.fit(X_train_reshaped, X_train_reshaped, epochs = 10, verbose = 0)
hiddenLayerSize_8_model.fit(X_train_reshaped, X_train_reshaped, epochs = 10, verbose = 0)
hiddenLayerSize_16_model.fit(X_train_reshaped, X_train_reshaped, epochs = 10, verbose = 0)
hiddenLayerSize_32_model.fit(X_train_reshaped, X_train_reshaped, epochs = 10, verbose = 0)

# Apply on testing set
output = model.predict(X_test_reshaped)

# Output other models
output_2_model = hiddenLayerSize_2_model.predict(X_test_reshaped)
output_4_model = hiddenLayerSize_4_model.predict(X_test_reshaped)
output_8_model = hiddenLayerSize_8_model.predict(X_test_reshaped)
output_16_model = hiddenLayerSize_16_model.predict(X_test_reshaped)
output_32_model = hiddenLayerSize_32_model.predict(X_test_reshaped)

# Compare before and after encoder to visually see if images are decoded well enough
fig, axes = plt.subplots(7, 5, figsize=(15,15))
randomly_selected_imgs = random.sample(range(output.shape[0]),5)
outputs = [X_test, output, output_2_model, output_4_model, output_8_model, output_16_model, output_32_model]
 
# Iterate through each subplot and plot accordingly
for row_num, row in enumerate(axes):
	for col_num, ax in enumerate(row):
		ax.imshow(outputs[row_num][randomly_selected_imgs[col_num]].reshape(28,28), cmap='gray')
		ax.grid(False)
		ax.set_xticks([])
		ax.set_yticks([])
plt.tight_layout()
plt.show()