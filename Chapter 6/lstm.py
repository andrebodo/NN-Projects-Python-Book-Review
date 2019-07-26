import seaborn as sns

from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense

from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix

n_words = 10000

training_set, testing_set = imdb.load_data(num_words = n_words) # Max number of unique words to be loaded

X_train, y_train = training_set
X_test, y_test = testing_set

# Zero padding (makes all movie review vectors the same length)
X_train_padded = sequence.pad_sequences(X_train, maxlen = 100)
X_test_padded = sequence.pad_sequences(X_test, maxlen = 100)


def train_model(Optimizer, X_train, y_train, X_val, y_val, verbose):
	# Hyperparams
	OUTPUT_DIM = 128

	model = Sequential()	
	model.add(Embedding(input_dim = n_words, output_dim = OUTPUT_DIM)) # Word embedding layer
	model.add(LSTM(units = 128)) # LSTM Layer
	model.add(Dense(units = 1, activation = 'sigmoid')) # Fully Connected Layer
	model.compile(loss = 'binary_crossentropy', optimizer = Optimizer, metrics=['accuracy'])
	scores = model.fit(X_train_padded, y_train, batch_size = 128, epochs = 10, validation_data = (X_test_padded, y_test), verbose = verbose)
	return scores, model

# SGD_score, SGD_model = train_model(Optimizer = 'sgd', X_train = X_train_padded, y_train = y_train, 
# 	X_val = X_test_padded, y_val = y_test, verbose = 1)

RMSprop_score, RMSprop_model = train_model(Optimizer = 'RMSprop', X_train = X_train_padded, y_train = y_train, 
 	X_val = X_test_padded, y_val = y_test, verbose = 1)

# Adam_score, Adam_model = train_model(Optimizer = 'adam', X_train = X_train_padded, y_train = y_train, 
# 	X_val = X_test_padded, y_val = y_test, verbose = 1)

# Analyze Results
plt.plot(range(1,11), RMSprop_score.history['acc'], label='Training Accuracy')
plt.plot(range(1,11), RMSprop_score.history['val_acc'], label='Validation Accuracy')
plt.axis([1, 10, 0, 1])
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Train and Validation Accuracy using RMSprop Optimizer')
plt.legend()
plt.show()

# Confusion Matrix
# • True negative: The actual class is negative (negative sentiment), and the model also predicted negative
# • False positive: The actual class is negative (negative sentiment), but the model predicted positive
# • False negative: The actual class is positive (positive sentiment), but the model predicted negative
# • True positive: The actual class is positive (positive sentiment), and the model predicted positive

plt.figure(figsize=(10,7))
sns.set(font_scale=2)
y_test_pred = RMSprop_model.predict_classes(X_test_padded)
c_matrix = confusion_matrix(y_test, y_test_pred)
ax = sns.heatmap(c_matrix, annot=True, xticklabels=['Negative Sentiment', 'Positive Sentiment'], 
	yticklabels=['Negative Sentiment', 'Positive Sentiment'], cbar=False, cmap='Blues', fmt='g')
ax.set_xlabel("Prediction")
ax.set_ylabel("Actual")
plt.show()