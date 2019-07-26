import os
import cv2 #OpenCV - open source computer vision library for computer vision tasks
import random
import numpy as np

from keras import backend as K
from keras.models import Sequential, Input, Model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Lambda
from keras.preprocessing.image import load_img, img_to_array
from matplotlib import pyplot as plt

face_cascades = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def detect_faces(img, draw_box=True):
	# convert image to grayscale
	grayscale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	 
	# detect faces
	faces = face_cascades.detectMultiScale(grayscale_img, scaleFactor=1.6)
	 
	# draw bounding box around detected faces
	for (x, y, width, height) in faces:
		if draw_box:
			cv2.rectangle(img, (x, y), (x+width, y+height), (0, 255, 0), 5)
		face_box = img[y:y+height, x:x+width]
		face_coords = [x,y,width,height]
	return img, face_box, face_coords

# files = os.listdir('sample_faces')
# images = [file for file in files if 'jpg' in file]
# for image in images:
# 	img = cv2.imread('sample_faces/' + image)
# 	detected_faces, _, _ = detect_faces(img)
# 	cv2.imwrite('sample_faces/detected_faces/' + image, detected_faces)

# Facial recognition system requirements
# 1. Speed
# 2. Scalability (Millions of Different Users)
# 3. Sufficient Accuracy


# One-Shot Learning
# Retrieved stored image obtained during onboarding process (true image)
# Take test image
# NN similary score between true and test
# Threshold score

# Naive one-shot prediction: Euclidean Distance
# Better method:
# • Use convolutional layers to extract identifying features from faces. 
# 	The output from the convolutional layers should be a mapping of the image to a lower-dimension feature space 
# 	(for example, a 128 x 1 vector). The convolutional layers should map faces from the same subject close to one 
# 	another in this lower-dimension feature space and vice versa, faces from different subjects should be as far 
# 	away as possible in this lower-dimension feature space.
#
# • Using the Euclidean distance, measure the difference of the two lower-dimension vectors output from the convolutional layers. 
#	Note that there are two vectors, because we are comparing two images (the true image and the test image). 
#	The Euclidean distance is inversely proportional to the similarity between the two images.
#
# • since we are feeding two images into our neural network simultaneously, 
#   we need two separate sets of convolutional layers. However, we require the two separate sets of convolutional 
#   layers to share the same weights, because we want similar faces to be mapped to the same point in the lower-dimension
#   feature space.
#
#	Contrasitve Loss:
#		Cannot use cross-entropy etc.. Need a new loss function which is distance based
#			contrastive loss function ensures that our Siamese neural network learns to predict a small distance when the faces 
#			in the true and test images are the same, and a large distance when the faces in the true and test images are 
#			different.


faces_dir = 'att_faces/'

X_train, Y_train = [], []
X_test, Y_test = [], []
# Get list of subfolders from faces_dir, Each subfolder contains images from one subject
subfolders = sorted([f.path for f in os.scandir(faces_dir) if f.is_dir()])

# Iterate through the list of subfolders (subjects) whereIdx is the subject ID
for idx, folder in enumerate(subfolders):
	for file in sorted(os.listdir(folder)):
		img = load_img(folder+"/"+file, color_mode='grayscale')
		img = img_to_array(img).astype('float32')/255
		if idx < 35:
			X_train.append(img)
			Y_train.append(idx)
		else:
			X_test.append(img)
			Y_test.append(idx - 35)

X_train = np.array(X_train)
X_test = np.array(X_test)
Y_train = np.array(Y_train)
Y_test = np.array(Y_test)

# # Plot some of the images for subject 4
# subject_idx = 4
# fig, ((ax1,ax2,ax3),(ax4,ax5,ax6), (ax7,ax8,ax9)) = plt.subplots(3,3,figsize=(10,10))
# subject_img_idx = np.where(Y_train==subject_idx)[0].tolist()
 
# for i, ax in enumerate([ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8,ax9]):
# 	img = X_train[subject_img_idx[i]]
# 	img = np.squeeze(img)
# 	ax.imshow(img, cmap='gray')
# 	ax.grid(False)
# 	ax.set_xticks([])
# 	ax.set_yticks([])
# plt.tight_layout()
# plt.show()

# # Plot the first 9 subjects
# subjects = range(10)
# fig, ((ax1,ax2,ax3),(ax4,ax5,ax6), (ax7,ax8,ax9)) = plt.subplots(3,3,figsize=(10,12))
# subject_img_idx = [np.where(Y_train == i)[0].tolist()[0] for i in subjects]
 
# for i, ax in enumerate([ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8,ax9]):
# 	img = X_train[subject_img_idx[i]]
# 	img = np.squeeze(img)
# 	ax.imshow(img, cmap='gray')
# 	ax.grid(False)
# 	ax.set_xticks([])
# 	ax.set_yticks([])
# 	ax.set_title("Subject {}".format(i))
# plt.tight_layout()
# plt.show()

def create_shared_network(input_shape):
	model = Sequential()
	model.add(Conv2D(filters = 128, kernel_size=(3, 3), activation = 'relu', input_shape = input_shape))
	model.add(MaxPooling2D())
	model.add(Conv2D(filters = 64, kernel_size=(3,3), activation = 'relu'))
	model.add(Flatten())
	model.add(Dense(units = 128, activation = 'sigmoid'))
	return model

# Create network
input_shape = X_train.shape[1:]
shared_network = create_shared_network(input_shape)
# Specify input for top and bottom layers using input class
input_top = Input(shape = input_shape)
input_bottom = Input(shape = input_shape)
# Stack shared network
output_top = shared_network(input_top)
output_bottom = shared_network(input_bottom)


# Calculate Euclidean Dist
def euclidean_distance(vectors):
	vector1, vector2 = vectors
	sum_squre = K.sum(K.square(vector1 - vector2), axis = 1, keepdims = True)
	return K.sqrt(K.maximum(sum_squre, K.epsilon()))

# Add eucliean distance to a lamdbda layer (wrap a layer by an arbitrary function)
distance = Lambda(euclidean_distance, output_shape = (1,))([output_top, output_bottom])
model = Model(inputs = [input_top, input_bottom], outputs = distance)

print(model.summary())

# The following function creates pairs of arrays of images and their labels from X_train and Y_train:
# We should alernate bewtween positive and negative pairs (matches and non-matches) to avoid bias
def create_pairs(X,Y, num_classes):
	pairs, labels = [], []
	# index of images in X and Y for each class
	class_idx = [np.where(Y==i)[0] for i in range(num_classes)]
	# The minimum number of images across all classes
	min_images = min(len(class_idx[i]) for i in range(num_classes)) - 1
	for c in range(num_classes):
		for n in range(min_images):
			# create positive pair
			img1 = X[class_idx[c][n]]
			img2 = X[class_idx[c][n+1]]
			pairs.append((img1, img2))
			labels.append(1)
			# create negative pair
			# list of classes that are different from the current class
			neg_list = list(range(num_classes))
			neg_list.remove(c)
			# select a random class from the negative list. 
			# This class will be used to form the negative pair.
			neg_c = random.sample(neg_list,1)[0]
			img1 = X[class_idx[c][n]]
			img2 = X[class_idx[neg_c][n]]
			pairs.append((img1,img2))
			labels.append(0)
	return np.array(pairs), np.array(labels)

num_classes = len(np.unique(Y_train))
training_pairs, training_labels = create_pairs(X_train, Y_train, len(np.unique(Y_train)))
test_pairs, test_labels = create_pairs(X_test, Y_test, len(np.unique(Y_test)))

# Contrastive loss function
def contrastive_loss(Y_true, D):
	margin = 1
	return K.mean(Y_true*K.square(D)+(1 - Y_true)*K.maximum((margin-D),0))

model.compile(loss=contrastive_loss, optimizer='adam', metrics=[utils.accuracy])
model.fit([training_pairs[:, 0], training_pairs[:, 1]], training_labels, batch_size=64, epochs=10)

# Analyze results
for i in range(5):
	for n in range(0,2):
		fig, (ax1, ax2) = plt.subplots(1,2, figsize=(7,5))
		img1 = np.expand_dims(test_pairs[i*20+n, 0], axis=0)
		img2 = np.expand_dims(test_pairs[i*20+n, 1], axis=0)
		dissimilarity = model.predict([img1, img2])[0][0]
		img1, img2 = np.squeeze(img1), np.squeeze(img2)
		ax1.imshow(img1, cmap='gray')
		ax2.imshow(img2, cmap='gray')
		 
		for ax in [ax1, ax2]:
			ax.grid(False)
			ax.set_xticks([])
			ax.set_yticks([])
			 
			plt.tight_layout()
			fig.suptitle("Dissimilarity Score = {:.3f}".format(dissimilarity), size=20)
plt.show()