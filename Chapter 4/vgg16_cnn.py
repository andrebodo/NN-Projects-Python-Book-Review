import random
import shutil
import piexif

from matplotlib import pyplot as plt
from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.layers import Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator


INPUT_SIZE   	= 128

# Do not import fully connected layers
vgg16 = VGG16(include_top = False, weights = 'imagenet', input_shape = (INPUT_SIZE, INPUT_SIZE, 3))

# Freeze convolution/pool max layers
for layer in vgg16.layers:
	layer.trainable = False

# Add fully connected layer (manual way vs .add() function) - This is the only layer we are training
input_  = vgg16.input
output_ = vgg16(input_)
last_layer = Flatten(name = 'flatten')(output_)
last_layer = Dense(1, activation = 'sigmoid')(last_layer)
model = Model(input = input_, output = last_layer)

# Define hyperparameters
BATCH_SIZE	 	= 16
STEPS_PER_EPOCH = 200
EPOCHS 			= 3

model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

training_data_generator = ImageDataGenerator(rescale = 1./255)
testing_data_generator  = ImageDataGenerator(rescale = 1./255)

training_set = training_data_generator.flow_from_directory('PetImages/Train/',
															target_size = (INPUT_SIZE, INPUT_SIZE),
															batch_size = BATCH_SIZE,
															class_mode = 'binary')
test_set = testing_data_generator.flow_from_directory('PetImages/Test/',
														target_size = (INPUT_SIZE, INPUT_SIZE),
														batch_size = BATCH_SIZE,
														class_mode = 'binary')

model.fit_generator(training_set, steps_per_epoch = STEPS_PER_EPOCH, epochs = EPOCHS, verbose = 1)


# score = model.evaluate_generator(test_set, len(test_set))
 
# print(f'\n')
# for idx, metric in enumerate(model.metrics_names):
# 	print("{}: {}".format(metric, score[idx]))

# Result Analysis
# • Strongly right predictions: The model predicted these images correctly, and the output value is > 0.8 or < 0.2
# • Strongly wrong predictions: The model predicted these images wrongly, and the output value is > 0.8 or < 0.2
# • Weakly wrong predictions: The model predicted these images wrongly, and the output value is between 0.4 and 0.6

strongly_wrong_idx = []
strongly_right_idx = []
weakly_wrong_idx = []

test_set = testing_data_generator.flow_from_directory('PetImages/Test/',
									target_size = (INPUT_SIZE,INPUT_SIZE),
									batch_size = 1,
									class_mode = 'binary')

for i in range(test_set.__len__()):

	img = test_set.__getitem__(i)[0]
	pred_prob = model.predict(img)[0][0]
	pred_label = int(pred_prob >0.5)
	actual_label = int(test_set.__getitem__(i)[1][0])

	if pred_label != actual_label and (pred_prob >0.8 or pred_prob <0.2): 
		strongly_wrong_idx.append(i)
	elif pred_label != actual_label and (pred_prob >0.4 and pred_prob <0.6): 
		weakly_wrong_idx.append(i)
	elif pred_label == actual_label and (pred_prob >0.8 or pred_prob <0.2): 
		strongly_right_idx.append(i)
	# stop once we have enough images to plot
	if (len(strongly_wrong_idx)>=9 and len(strongly_right_idx)>=9 and len(weakly_wrong_idx)>=9): 
		break
def plot_on_grid(test_set, idx_to_plot, img_size=INPUT_SIZE):
	fig, ax = plt.subplots(3, 3, figsize=(20,10))
	for i, idx in enumerate(random.sample(idx_to_plot, 9)):
		img = test_set[idx][0].reshape(img_size, img_size, 3)
		ax[int(i/3), i%3].imshow(img)
		ax[int(i/3), i%3].axis('off')
	plt.show()

plot_on_grid(test_set, strongly_right_idx)
plot_on_grid(test_set, strongly_wrong_idx)
