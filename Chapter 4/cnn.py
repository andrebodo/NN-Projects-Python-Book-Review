# Dataset from https://www.microsoft.com/en-us/download/confirmation.aspx?id=54765

# CNN design
# Layers: Convolution, Max Pooling, Convolution, Max Pooling, Fully Connected (Dense), Dense
# Dense Layer Activation: sigmoid for binary classifications, softmax for multiclass

# Some CNNs
# VGG16 -> CNN with a small convolution filter (3x3). Disadvantage -> many more params to train therefore longer train time
# Inception -> Google created, higher acc than VGG16 but fewer params, online Inception-v4 is most recent
# ResNet (2015) -> residual block technique: deeper NN with moder number of params

import os
import shutil
import piexif
import random
from matplotlib import pyplot as plt

# Get list of filenames
_, _, cat_images = next(os.walk('PetImages/Cat'))

# Prepare 3x3 plot
fig, ax = plt.subplots(3, 3, figsize = (20, 10))

# Randomly select and plot images
for idx, img in enumerate(random.sample(cat_images, 9)):
	img_read = plt.imread('PetImages/Cat/' + img)
	img_read = plt.imread('PetImages/Cat/'+img)
	ax[int(idx/3), idx%3].imshow(img_read)
	ax[int(idx/3), idx%3].axis('off')
	ax[int(idx/3), idx%3].set_title('Cat/'+img)
#plt.show()

# Images have different dimensions.
# Subjects are mostly centered
# Subject have different orientations

# Dog Images
# Get list of filenames
_, _, cat_images = next(os.walk('PetImages/Dog'))

# Prepare 3x3 plot
fig, ax = plt.subplots(3, 3, figsize = (20, 10))

# Randomly select and plot images
for idx, img in enumerate(random.sample(cat_images, 9)):
	img_read = plt.imread('PetImages/Dog/' + img)
	img_read = plt.imread('PetImages/Dog/'+ img)
	ax[int(idx/3), idx%3].imshow(img_read)
	ax[int(idx/3), idx%3].axis('off')
	ax[int(idx/3), idx%3].set_title('Dog/'+img)
#plt.show()

# Too many images to load in ram.
# Keras has flow_from_directory method that generates batches of images as output
# Can be used to preprocess and train netowrk
# To use flow_from_directory we need file and folder schema:
#	subdirs for train and test data, within each need a subdir per class
# helper function to remove corrupt exif data from Microsoft's dataset
def remove_exif_data(src_folder):
	_, _, cat_images = next(os.walk(src_folder+'Cat/'))
	for img in cat_images:
		try:
			piexif.remove(src_folder+'Cat/'+img)
		except:
			pass

	_, _, dog_images = next(os.walk(src_folder+'Dog/'))
	for img in dog_images:
		try:
			piexif.remove(src_folder+'Dog/'+img)
		except:
			pass

def train_test_split(src_folder, train_size = 0.8):
	# Make sure we remove any existing folders and start from a clean slate
	shutil.rmtree(src_folder+'Train/Cat/', ignore_errors=True)
	shutil.rmtree(src_folder+'Train/Dog/', ignore_errors=True)
	shutil.rmtree(src_folder+'Test/Cat/', ignore_errors=True)
	shutil.rmtree(src_folder+'Test/Dog/', ignore_errors=True)

	# Now, create new empty train and test folders
	os.makedirs(src_folder+'Train/Cat/')
	os.makedirs(src_folder+'Train/Dog/')
	os.makedirs(src_folder+'Test/Cat/')
	os.makedirs(src_folder+'Test/Dog/')

	# Get the number of cats and dogs images
	_, _, cat_images = next(os.walk(src_folder+'Cat/'))
	files_to_be_removed = ['Thumbs.db', '666.jpg', '835.jpg']
	for file in files_to_be_removed:
		cat_images.remove(file)
	num_cat_images = len(cat_images)
	num_cat_images_train = int(train_size * num_cat_images)
	num_cat_images_test = num_cat_images - num_cat_images_train

	_, _, dog_images = next(os.walk(src_folder+'Dog/'))
	files_to_be_removed = ['Thumbs.db', '11702.jpg']
	for file in files_to_be_removed:
		dog_images.remove(file)
	num_dog_images = len(dog_images)
	num_dog_images_train = int(train_size * num_dog_images)
	num_dog_images_test = num_dog_images - num_dog_images_train

	# Randomly assign images to train and test
	cat_train_images = random.sample(cat_images, num_cat_images_train)
	for img in cat_train_images:
		shutil.copy(src=src_folder+'Cat/'+img, dst=src_folder+'Train/Cat/')
	cat_test_images  = [img for img in cat_images if img not in cat_train_images]
	for img in cat_test_images:
		shutil.copy(src=src_folder+'Cat/'+img, dst=src_folder+'Test/Cat/')

	dog_train_images = random.sample(dog_images, num_dog_images_train)
	for img in dog_train_images:
		shutil.copy(src=src_folder+'Dog/'+img, dst=src_folder+'Train/Dog/')
	dog_test_images  = [img for img in dog_images if img not in dog_train_images]
	for img in dog_test_images:
		shutil.copy(src=src_folder+'Dog/'+img, dst=src_folder+'Test/Dog/')

	# remove corrupted exif data from the dataset
	remove_exif_data(src_folder+'Train/')
	remove_exif_data(src_folder+'Test/')

# This takes a while
# src_folder = 'PetImages/'
# train_test_split(src_folder)


# Image Augmentation -> Using image dataset, artificially create more images by rotating, translating, flipping, zooming
from keras.preprocessing.image import ImageDataGenerator

image_generator = ImageDataGenerator(rotation_range = 30,
									 width_shift_range = 0.2,
									 height_shift_range = 0.2,
									 zoom_range = 0.2,
									 horizontal_flip = True,
									 fill_mode = 'nearest')
# Test Augmentation on Dogs
fig, ax = plt.subplots(2, 3, figsize = (20, 10))
all_images = []
_, _, dog_images = next(os.walk('PetImages/Train/Dog/'))
random_img = random.sample(dog_images, 1)[0]
random_img = plt.imread('PetImages/Train/Dog/' + random_img)
all_images.append(random_img)
random_img = random_img.reshape((1,) + random_img.shape)
sample_augmented_images = image_generator.flow(random_img)
for _ in range(5):
	augmented_imgs = sample_augmented_images.next()
	for img in augmented_imgs:
		all_images.append(img.astype('uint8'))
for idx, img in enumerate(all_images):
	ax[int(idx/3), idx%3].imshow(img)
	ax[int(idx/3), idx%3].axis('off')
	if idx == 0:
		ax[int(idx/3), idx%3].set_title('Original Image')
	else:
		ax[int(idx/3), idx%3].set_title('Augmented Image {}'.format(idx))
# plt.show()

# CNN 
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator

model = Sequential()

# Hyperparams:
# Convolutional layer filter size 3x3
# Number of filters: 32
# Input size 32x32 pixels -> compress original image
# Max pooling size: 2x2 (halves the input later dimensions)
# Batch size: Corresponds to the number of training samples to use in each mini batch during gradient descent
# 	Larger batch sizes = more accurate train but longer time and more mem. Use 16 batch size
# Steps per epoch: Number of iters in each train epoch. = # Train Samples / Batch Size
# Epochs: 10

FILTER_SIZE  	= 3
NUM_FILTERS  	= 32
INPUT_SIZE   	= 32
MAXPOOL_SIZE 	= 2
BATCH_SIZE	 	= 16
STEPS_PER_EPOCH = 20000 // BATCH_SIZE
EPOCHS 			= 10

# Add first conv. layer
model.add(Conv2D(NUM_FILTERS, (FILTER_SIZE, FILTER_SIZE), input_shape = (INPUT_SIZE, INPUT_SIZE, 3), activation = 'relu'))
# Max pooling layer
model.add(MaxPooling2D(pool_size = (MAXPOOL_SIZE, MAXPOOL_SIZE)))
# Add second conv. layer
model.add(Conv2D(NUM_FILTERS, (FILTER_SIZE, FILTER_SIZE), input_shape = (INPUT_SIZE, INPUT_SIZE, 3), activation = 'relu'))
# Max pooling layer
model.add(MaxPooling2D(pool_size = (MAXPOOL_SIZE, MAXPOOL_SIZE)))
# Flatten layer (conv result to Dense layer)
model.add(Flatten())
# Fully connected layer with 128 nodes
model.add(Dense(units = 128, activation = 'relu'))
# Dropout layer -> Good pactice, Randomly set 50% of inputs to 0
model.add(Dropout(0.5))
# Fully connected layer (Binary Classification)
model.add(Dense(units = 1, activation = 'sigmoid'))
# Compile
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Training with flow from memory
training_data_generator = ImageDataGenerator(rescale = 1./255)
training_set = training_data_generator.flow_from_directory('PetImages/Train/',
															target_size = (INPUT_SIZE, INPUT_SIZE),
															batch_size = BATCH_SIZE,
															class_mode = 'binary')

# Train model
model.fit_generator(training_set, steps_per_epoch = STEPS_PER_EPOCH, epochs = EPOCHS, verbose  = 1)


# Evaluate Model on testing 
testing_data_generator = ImageDataGenerator(rescale = 1./255)
test_set = training_data_generator.flow_from_directory('PetImages/Test/',
														target_size = (INPUT_SIZE, INPUT_SIZE),
														batch_size = BATCH_SIZE,
														class_mode = 'binary')

score = model.evaluate_generator(test_set, steps = len(test_set))
for idx, metric in enumerate(model.metrics_names):
	print("{}: {}".format(metric, score[idx]))

# Leverage on pre-trained models using transfer learning
# Model trained on certain task is modified to make predictions for another task:
#	ex. Use car classification model to train trucks
# 	Involves freezing the convolution-pooling layers and only retaining the final fully connected layers
#	New task must be similar to old task 

