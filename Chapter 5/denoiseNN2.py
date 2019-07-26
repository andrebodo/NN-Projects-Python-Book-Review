import os
import numpy as np
import random

from matplotlib import pyplot as plt

from keras.models import Sequential
from keras.layers import Dense, Conv2D
from keras.preprocessing.image import load_img, img_to_array


# Import University of California Irvine noisy office
noisy_imgs_path = 'noisy/'
clean_imgs_path = 'clean/'

X_train_noisy = []
 
for file in sorted(os.listdir(noisy_imgs_path)):
	img = load_img(noisy_imgs_path+file, color_mode='grayscale', target_size=(420,540))
	img = img_to_array(img).astype('float32')/255
	X_train_noisy.append(img)
 
# convert to numpy array
X_train_noisy = np.array(X_train_noisy)

# verify shape
print(X_train_noisy.shape)

X_train_clean = []
for file in sorted(os.listdir(clean_imgs_path)):
	img = load_img(clean_imgs_path+file, color_mode='grayscale', target_size=(420,540))
	img = img_to_array(img).astype('float32')/255
	X_train_clean.append(img)

# convert to numpy array
X_train_clean = np.array(X_train_clean)

# look at images
fig, ((ax1,ax2), (ax3,ax4), (ax5,ax6)) = plt.subplots(3, 2, figsize=(10,12))
randomly_selected_imgs = random.sample(range(X_train_noisy.shape[0]),3)

# plot noisy images on the left
for i, ax in enumerate([ax1,ax3,ax5]):
	ax.imshow(X_train_noisy[i].reshape(420,540), cmap='gray')
	if i == 0:
		ax.set_title("Noisy Images", size=30)
	ax.grid(False)
	ax.set_xticks([])
	ax.set_yticks([])
 
# plot clean images on the right
for i, ax in enumerate([ax2,ax4,ax6]):
	ax.imshow(X_train_clean[i].reshape(420,540), cmap='gray')
	if i == 0:
		ax.set_title("Clean Images", size=30)
	ax.grid(False)
	ax.set_xticks([])
	ax.set_yticks([])
 
plt.tight_layout()
plt.show()

# use the first 20 noisy images as testing images
X_test_noisy = X_train_noisy[0:20,]
X_train_noisy = X_train_noisy[21:,]
 
# use the first 20 clean images as testing images
X_test_clean = X_train_clean[0:20,]
X_train_clean = X_train_clean[21:,]

conv_autoencoder = Sequential()
# Encoder
conv_autoencoder.add(Conv2D(filters=32, kernel_size=(3,3), input_shape=(420,540,1),  activation='relu', padding='same'))
conv_autoencoder.add(Conv2D(filters=16, kernel_size=(3,3), activation='relu', padding='same'))
conv_autoencoder.add(Conv2D(filters=8, kernel_size=(3,3), activation='relu', padding='same'))
# Decoder
conv_autoencoder.add(Conv2D(filters=8, kernel_size=(3,3), activation='relu', padding='same'))
conv_autoencoder.add(Conv2D(filters=16, kernel_size=(3,3), activation='relu', padding='same'))
conv_autoencoder.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu', padding='same'))
# Output
conv_autoencoder.add(Conv2D(filters=1, kernel_size=(3,3), activation='sigmoid', padding='same'))

conv_autoencoder.summary()

# Train
conv_autoencoder.compile(optimizer = 'adam', loss='binary_crossentropy')
conv_autoencoder.fit(X_train_noisy, X_train_clean, epochs = 20)
output = conv_autoencoder.predict(X_test_noisy)

# Plot output
fig, ((ax1,ax2,ax3),(ax4,ax5,ax6)) = plt.subplots(2,3, figsize=(20,10))
randomly_selected_imgs = random.sample(range(X_test_noisy.shape[0]),2)

for i, ax in enumerate([ax1, ax4]):
	idx = randomly_selected_imgs[i]
	ax.imshow(X_test_noisy[idx].reshape(420,540), cmap='gray')
	if i == 0:
		ax.set_title("Noisy Images", size=30)
	ax.grid(False)
	ax.set_xticks([])
	ax.set_yticks([])
 
for i, ax in enumerate([ax2, ax5]):
	idx = randomly_selected_imgs[i]
	ax.imshow(X_test_clean[idx].reshape(420,540), cmap='gray')
	if i == 0:
		ax.set_title("Clean Images", size=30)
	ax.grid(False)
	ax.set_xticks([])
	ax.set_yticks([])
 
for i, ax in enumerate([ax3, ax6]):
	idx = randomly_selected_imgs[i]
	ax.imshow(output[idx].reshape(420,540), cmap='gray')
	if i == 0:
		ax.set_title("Output Denoised Images", size=30)
	ax.grid(False)
	ax.set_xticks([])
	ax.set_yticks([])

plt.tight_layout()
plt.show()