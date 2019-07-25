import numpy as np
import pandas as pd
import seaborn as sns

from matplotlib import pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve

from keras.models import Sequential
from keras.layers import Dense



df = pd.read_csv('diabetes.csv')

#print(df.head())
#df.hist()
#plt.show()

# plt.subplots(3, 3, figsize = (15, 15))
# # Plot a density plot for eahc variable
# for idx, col in enumerate(df.columns):
# 	ax = plt.subplot(3, 3, idx + 1)
# 	ax.yaxis.set_ticklabels([])
# 	sns.distplot(df.loc[df.Outcome == 0][col], hist = False, axlabel = False,
# 		kde_kws = {'linestyle':'-', 'color':'black', 'label':"No Diabetes"})
# 	sns.distplot(df.loc[df.Outcome == 1][col], hist = False, axlabel = False,
# 		kde_kws = {'linestyle':'-', 'color':'red', 'label':"Diabetes"})
# 	ax.set_title(col)
# # Hide 9th subplot (bottom right since there are only 8 plots)
# plt.subplot(3, 3, 9).set_visible(False)
# plt.show()

# Checck for missing values
# print(df.isnull().any())
# print(df.describe())

# print("Number of rows with 0 values for each variable")
# for col in df.columns:
# 	missing_rows = df.loc[df[col]==0].shape[0]
# 	print(col + ": " + str(missing_rows))

# Fix 0 values for bmi, glucose, skinthickness, etc.. with means
df['Glucose'] = df['Glucose'].replace(0, np.nan)
df['BloodPressure'] = df['BloodPressure'].replace(0, np.nan)
df['SkinThickness'] = df['SkinThickness'].replace(0, np.nan)
df['Insulin'] = df['Insulin'].replace(0, np.nan)
df['BMI'] = df['BMI'].replace(0, np.nan)

# Check to see if there are zero values (should be np.nan)
# print("Number of rows with 0 values for each variable")
# for col in df.columns:
# 	missing_rows = df.loc[df[col]==0].shape[0]
# 	print(col + ": " + str(missing_rows))

# Replace nan values with means
df['Glucose'] = df['Glucose'].fillna(df['Glucose'].mean())
df['BloodPressure'] = df['BloodPressure'].fillna(df['BloodPressure'].mean())
df['SkinThickness'] = df['SkinThickness'].fillna(df['SkinThickness'].mean())
df['Insulin'] = df['Insulin'].fillna(df['Insulin'].mean())
df['BMI'] = df['BMI'].fillna(df['BMI'].mean())

# Data standardization
df_scaled = preprocessing.scale(df)
# Convert back to pandas dataframe object
df_scaled = pd.DataFrame(df_scaled, columns = df.columns)
# Outcome column should not be scaled
df_scaled['Outcome'] = df['Outcome']

df = df_scaled

print(df.describe().loc[['mean', 'std', 'max'], ].round(2).abs())

# Split into training and testing, validation
# Training = Train NN
# Validation = Hyperparameter tuning
# Testing = Final eval NN

# Original Dataset -> 80:20 (train:test) 
# Train dataset -> 80:20 train:validate
X = df.loc[:, df.columns != 'Outcome']
y = df.loc[:, 'Outcome']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
X_train, X_val, y_train, y_val   = train_test_split(X_train, y_train, test_size = 0.2)

# Multi-Layered Perceptron Network

# Input layer -> 8 Features = 8 Nodes (8 Columns in X_Train)
# 2 hidden Layers -> Specify nodes as 32 for first hidden layer. 16 for second 
#     This 32 and 16 are hyper parameters that needs to be tuned later
# Activation Functions : Rectified Linear Unit (ReLU) and sigmoid
# ReLU Notes: Rule of thumb, use as activation for intermediate hidden layers. Most popular choice for DNN 

model = Sequential()

# Add First hidden layer
model.add(Dense(32, activation = 'relu', input_dim = 8))
model.add(Dense(16, activation = 'relu'))
# Output Layer
model.add(Dense(1, activation='sigmoid'))
# Compile:
# Loss Fn = binary_crossentropy this is binary classification
# Metrics =  accuracy (% correctly classified)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics = ['accuracy'])

# Train
model.fit(X_train, y_train, epochs = 200,  verbose = False)


# Result analysis
# testing accuracy
# confusion matrix
# reciever operation characteristic (ROC) curve


# Accuracy
scores = model.evaluate(X_train, y_train)
print("Training Accuracy: %.2f%%\n" % (scores[1]*100))

scores = model.evaluate(X_test, y_test)
print("Testing Accuracy: %.2f%%\n" % (scores[1]*100))

# Confusion Matrix:
# Data viz tool. Provdies Analysis of true negative, false positive, false negative, and true positives of model
# • True negative: Actual class is negative (no diabetes), and the model predicted negative (no diabetes)
# • False positive: Actual class is negative (no diabetes), but the model predicted positive (diabetes)
# • False negative: Actual class is positive (diabetes), but the model predicted negative (no diabetes)
# • True positive: Actual class is positive (diabetes), and the model predicted positive (diabetes)
y_test_pred = model.predict_classes(X_test)
c_matrix = confusion_matrix(y_test, y_test_pred)
ax = sns.heatmap(c_matrix, annot = True,
	xticklabels = ['No Diabetes', 'Diabetes'],
	yticklabels = ['No Diabetes', 'Diabetes'],
	cbar = False, cmap = 'Blues')
ax.set_xlabel("Prediction")
ax.set_ylabel("Actual")
plt.show()


# ROC Curve:
# The ROC curve is a plot with the True Positive Rate (TPR) on the y axis and the False Positive Rate (FPR) on the x axis.
# TPR = True Pos / (True Pos + False Neg), FPR = False Pos / (True Neg + False Pos)
# Look at area under curve to eval performance of model. A large area indicates model is able to differentiate the respective
# 	classes with high accuracy. Low area means model makes poor, often wrong predictions.
#	ROC curve that is at the 45 means the model does no better than random
y_test_pred_probs = model.predict(X_test)
FPR, TPR, _ = roc_curve(y_test, y_test_pred_probs)
plt.plot(FPR, TPR)
plt.plot([0,1],[0,1],'--', color='black') #diagonal line
plt.title('ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()

# Can feature engineer the data (feature selection) using decision trees:
# Decision trees are a separate class of machine learning models with a tree-like data structure. 
# Decision trees are useful as they calculate and rank the most important features according to certain statistical criteria.
# We can first fit the data using the decision tree, and then use the output from the decision tree to remove features that 
# are deemed unimportant, before providing the reduced dataset to our neural network. Again, feature selection is a double-edged 
# sword that can potentially affect model performance.


