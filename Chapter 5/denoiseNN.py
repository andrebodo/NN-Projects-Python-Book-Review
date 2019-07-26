import random
import numpy as np

from keras.layers import Dense
from keras.models import Sequential
from keras.datasets import mnist 
from matplotlib import pyplot as plt


def create_basic_autoencoder(hidden_layer_size):
	model = Sequential()
	# Add hidden layer
	model.add(Dense(units = hidden_layer_size, input_shape=(784,), activation = 'relu'))
	# Add output layer 784 = 28x28 px
	model.add(Dense(units = 784, activation='sigmoid'))

	return model

training_set, testing_set = mnist.load_data()

# Train and Test datasets
X_train, y_train = training_set
X_test, y_test = testing_set

# Reshape 28x28 to 784x1: X_train.shape[0] is the number of samples
X_train_reshaped = X_train.reshape((X_train.shape[0], X_train.shape[1]*X_train.shape[2]))
X_test_reshaped  = X_test.reshape((X_test.shape[0], X_test.shape[1]*X_test.shape[2]))

# Add noise
X_train_noisy = X_train_reshaped + np.random.normal(0, 0.5, size=X_train_reshaped.shape)
X_test_noisy = X_test_reshaped + np.random.normal(0, 0.5, size=X_test_reshaped.shape)

# Clip between 0 and 1 to normalize images
X_train_noisy = np.clip(X_train_noisy, a_min=0, a_max=1)
X_test_noisy = np.clip(X_test_noisy, a_min=0, a_max=1)


# Denoise autoencoder
basic_denoise_autoencoder = create_basic_autoencoder(hidden_layer_size=16)

# Train
basic_denoise_autoencoder.compile(optimizer='adam', loss='mean_squared_error')
basic_denoise_autoencoder.fit(X_train_noisy, X_train_reshaped, epochs=10)

# Output
output = basic_denoise_autoencoder.predict(X_test_noisy)


def plot_compare(X_test_reshaped, X_test_noisy, output):
	fig, ((ax1, ax2, ax3, ax4, ax5), (ax6, ax7, ax8, ax9, ax10), (ax11,ax12,ax13,ax14,ax15)) = plt.subplots(3, 5, figsize=(20,13))
	randomly_selected_imgs = random.sample(range(output.shape[0]),5)
 
	# 1st row for original images
	for i, ax in enumerate([ax1,ax2,ax3,ax4,ax5]):
		ax.imshow(X_test_reshaped[randomly_selected_imgs[i]].reshape(28,28),  cmap='gray')
		if i == 0:
			ax.set_ylabel("Original \n Images", size=30)
			ax.grid(False)
			ax.set_xticks([])
			ax.set_yticks([])
 
	# 2nd row for input with noise added
	for i, ax in enumerate([ax6,ax7,ax8,ax9,ax10]):
		ax.imshow(X_test_noisy[randomly_selected_imgs[i]].reshape(28,28), cmap='gray')
		if i == 0:
			ax.set_ylabel("Input With \n Noise Added", size=30)
			ax.grid(False)
			ax.set_xticks([])
			ax.set_yticks([])
 
	# 3rd row for output images from our autoencoder
	for i, ax in enumerate([ax11,ax12,ax13,ax14,ax15]):
		ax.imshow(output[randomly_selected_imgs[i]].reshape(28,28), cmap='gray')
		if i == 0:
			ax.set_ylabel("Denoised \n Output", size=30)
			ax.grid(False)
			ax.set_xticks([])
			ax.set_yticks([])
 
	plt.tight_layout()
	plt.show()

plot_compare(X_test_reshaped, X_test_noisy, output)