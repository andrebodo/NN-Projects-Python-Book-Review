import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from keras.models import Sequential
from keras.layers import Dense

df = pd.read_csv('NYC_taxi.csv', parse_dates = ['pickup_datetime'], nrows = 500000)

# df.drop(labels='key', axis = 1, inplace = True, errors='raise')

# Restrict to NYC

# Range of Longitudes
nyc_min_longitude = -74.05
nyc_max_longitude = -73.75

# Range of Latitudes
nyc_min_latitude = 40.63
nyc_max_latitude = 40.85

# df2 = df.copy(deep = True)

# for long in ['pickup_longitude', 'dropoff_longitude']:
# 	df2 = df2[(df2[long] > nyc_min_longitude) & (df2[long] < nyc_max_longitude)]

# for lat in ['pickup_latitude', 'dropoff_latitude']:
# 	df2 = df2[(df2[lat] > nyc_min_latitude) & (df2[lat] < nyc_max_latitude)]

landmarks = {'JFK Airport': (-73.78, 40.643),
			'Laguardia Airport': (-73.87, 40.77),
			'Midtown': (-73.98, 40.76),
			'Lower Manhattan': (-74.00, 40.72),
			'Upper Manhattan': (-73.94, 40.82),
			'Brooklyn': (-73.95, 40.66)}

def plot_lat_long(df, landmarks, points = 'Pickup'):
	plt.figure(figsize = (12, 12))
	if points == 'Pickup':
		plt.plot(list(df.pickup_longitude), list(df.pickup_latitude), '.', markersize = 1)
	else:
		plt.plot(list(df.dropoff_longitude), list(df.dropoff_latitude), '.', markersize = 1)
	for landmark in landmarks:
		plt.plot(landmarks[landmark][0], landmarks[landmark][1], '*', markersize=15, alpha=1, color='r')
		plt.annotate(landmark, (landmarks[landmark][0]+0.005, landmarks[landmark][1]+0.005), color='r', backgroundcolor='w')
	plt.title("{} Locations in NYC Illustrated".format(points))
	plt.grid(None)
	plt.xlabel("Latitude")
	plt.ylabel("Longitude")
	plt.show()

#plot_lat_long(df2, landmarks)
#plot_lat_long(df2, landmarks, points='Drop Off')

# Ridership by day and hour
# df2['year'] = df2['pickup_datetime'].dt.year
# df2['month'] = df2['pickup_datetime'].dt.month
# df2['day'] = df2['pickup_datetime'].dt.day
# df2['day_of_week'] = df2['pickup_datetime'].dt.dayofweek
# df2['hour'] = df2['pickup_datetime'].dt.hour

# df2['day_of_week'].plot.hist(bins=np.arange(8)-0.5, ec='black')
# plt.xlabel('Day of Week (0=Monday, 6=Sunday)')
# plt.title('Day of Week Histogram')
# plt.show()

# df2['hour'].plot.hist(bins=24, ec='black')
# plt.title('Pickup Hour Histogram')
# plt.xlabel('Hour')
# plt.show()

# # Data preprocessing
# print(df.isnull().sum())

# # Remove NA rows
# # df = df.dropna()
# print(df.describe())

# # Inspect Negative and Really High Fares
# df['fare_amount'].hist(bins=500)
# plt.title("Histogram of Fares")
# plt.show()

# # Inspect Passanger Count
# df['passenger_count'].hist(bins=6, ec='black')
# plt.xlabel("Passenger Count")
# plt.title("Histogram of Passenger Count")
# plt.show()

# # Inspect logitude and latitude data for outliers
# df.plot.scatter('pickup_longitude', 'pickup_latitude')
# plt.show()


def preprocess(df):

	# Remove missing values
	def remove_missing_values(df):
		df = df.dropna()
		return df

	# Remove Fare Outliers (Negative and Really High Fares, Since there are few based on histogram)
	def remove_fare_amount_outliers(df, lower_bound, upper_bound):
		df = df[(df['fare_amount'] > lower_bound) & (df['fare_amount'] <= upper_bound)]
		return df
	
	# Replace 0 passenger count with the mode passenger count
	def replace_passenger_count_outliers(df):
		mode = df['passenger_count'].mode().values[0]
		df.loc[df['passenger_count'] == 0, 'passenger_count'] = mode
		return df
	# remove outliers in latitude and longitude
	def remove_lat_long_outliers(df):

		# Range of Longitudes
		nyc_min_longitude = -74.05
		nyc_max_longitude = -73.75

		# Range of Latitudes
		nyc_min_latitude = 40.63
		nyc_max_latitude = 40.85

		for long in ['pickup_longitude', 'dropoff_longitude']:
			df = df[(df[long] > nyc_min_longitude) & (df[long] < nyc_max_longitude)]

		for lat in ['pickup_latitude', 'dropoff_latitude']:
			df = df[(df[lat] > nyc_min_latitude) & (df[lat] < nyc_max_latitude)]

		return df
		
	df = remove_missing_values(df)
	df = remove_fare_amount_outliers(df, lower_bound = 0, upper_bound = 100)
	df = replace_passenger_count_outliers(df)
	df = remove_lat_long_outliers(df)

	return df

# FEATURES ================

# # Inspect datetime, NN can't handle the format
# print(df.head()['pickup_datetime'])

# # Separate into numerical data
# df['year'] = df['pickup_datetime'].dt.year
# df['month'] = df['pickup_datetime'].dt.month
# df['day'] = df['pickup_datetime'].dt.day
# df['day_of_week'] = df['pickup_datetime'].dt.dayofweek
# df['hour'] = df['pickup_datetime'].dt.hour

# # look at new output
# print(df.loc[:5,['pickup_datetime', 'year', 'month', 'day', 'day_of_week', 'hour']])

# # drop pickup_datetime
# df = df.drop(['pickup_datetime'], axis=1)

# # Geolocation
# def euc_dist(lat1, long1, lat2, long2):
# 	return(((lat1-lat2)**2 + (long1-long2)**2)**0.5)

# # Apply distance to df
# df['distance'] = euc_dist(df['pickup_latitude'], df['pickup_longitude'],  df['dropoff_latitude'], df['dropoff_longitude'])

# # Plot distance vs fare
# df.plot.scatter('fare_amount', 'distance')
# plt.show()

# # Notices 3 'lines' where distance doesn't matter in fare

# # Calculate distances for the airports
# airports = {'JFK_Airport': (-73.78,40.643),
# 			'Laguardia_Airport': (-73.87, 40.77),
# 			'Newark_Airport' : (-74.18, 40.69)}

# for airport in airports:
# 	df['pickup_dist_' + airport] = euc_dist(df['pickup_latitude'], df['pickup_longitude'], 
# 		airports[airport][1], airports[airport][0])
# 	df['dropoff_dist_' + airport] = euc_dist(df['dropoff_latitude'], df['dropoff_longitude'], 
# 		airports[airport][1], airports[airport][0])

# print(df[['key', 'pickup_longitude','pickup_latitude',
# 	'dropoff_longitude','dropoff_latitude', 'pickup_dist_JFK_Airport',
# 	'dropoff_dist_JFK_Airport']].head())

# df = df.drop(['key'], axis=1)

def feature_engineer(df):
	# create new columns for year, month, day, day of week and hour
	def create_time_features(df):
		df['year'] = df['pickup_datetime'].dt.year
		df['month'] = df['pickup_datetime'].dt.month
		df['day'] = df['pickup_datetime'].dt.day
		df['day_of_week'] = df['pickup_datetime'].dt.dayofweek
		df['hour'] = df['pickup_datetime'].dt.hour
		df = df.drop(['pickup_datetime'], axis=1)
		return df
	# Calc Euclidiean Dist
	def euc_distance(lat1, long1, lat2, long2):
		return(((lat1-lat2)**2 + (long1-long2)**2)**0.5)

	# create new column for the distance travelled
	def create_pickup_dropoff_dist_features(df):
		df['travel_distance'] = euc_distance(df['pickup_latitude'], df['pickup_longitude'], 
			df['dropoff_latitude'], df['dropoff_longitude'])
		return df

	# create new column for the distance away from airports
	def create_airport_dist_features(df):
		airports = {'JFK_Airport': (-73.78,40.643), 'Laguardia_Airport': (-73.87, 40.77), 'Newark_Airport' : (-74.18, 40.69)}
		for k in airports:
			df['pickup_dist_'+k] = euc_distance(df['pickup_latitude'], df['pickup_longitude'], 
				airports[k][1], airports[k][0])
			df['dropoff_dist_'+k] = euc_distance(df['dropoff_latitude'], df['dropoff_longitude'],
				airports[k][1],	airports[k][0]) 
		return df
	df = create_time_features(df)
	df = create_pickup_dropoff_dist_features(df)
	df = create_airport_dist_features(df)
	df = df.drop(['key'], axis=1)
	return df

df = preprocess(df)
df = feature_engineer(df)

# Feature Scaling
df_prescaled = df.copy()
# Drop output
df_scaled = df.drop(['fare_amount'], axis=1)
df_scaled = scale(df_scaled)
cols = df.columns.tolist()
cols.remove('fare_amount')
df_scaled = pd.DataFrame(df_scaled, columns=cols, index=df.index)
df_scaled = pd.concat([df_scaled, df['fare_amount']], axis=1)
df = df_scaled.copy()

# DEEP FF NN:
# Input layer -> 17 features
# 4 Hidden layers, First hidden 128 nodes
# Each sucessive layer halving nodes of prior (64, 32, 8)
# ReLU activation between hidden layers
# Output Layer: only 1 node in output layer because regression problem -> no ReLU for output layer
# Loss function: Regression problems commonly use RMSE

# Split data
X = df.loc[:, df.columns != 'fare_amount']
y = df.loc[:, 'fare_amount']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

model = Sequential()
model.add(Dense(128, activation = 'relu', input_dim = X_train.shape[1]))
model.add(Dense(64, activation = 'relu'))
model.add(Dense(32, activation = 'relu'))
model.add(Dense(68, activation = 'relu'))
model.add(Dense(1))

# Check to see if model is ok
model.summary()

# Compile and train
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.fit(X_train, y_train, epochs=1)

# Pull a random row from the testing set, and feed it to model for prediction
def predict_random(df_prescaled, X_test, model):
	sample = X_test.sample(n=1, random_state=np.random.randint(low=0, high=10000))
	idx = sample.index[0]
 
	actual_fare = df_prescaled.loc[idx,'fare_amount']
	day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
	day_of_week = day_names[df_prescaled.loc[idx,'day_of_week']]
	hour = df_prescaled.loc[idx,'hour']
	predicted_fare = model.predict(sample)[0][0]
	rmse = np.sqrt(np.square(predicted_fare-actual_fare))
 
	print("Trip Details: {}, {}:00hrs".format(day_of_week, hour)) 
	print("Actual fare: ${:0.2f}".format(actual_fare))
	print("Predicted fare: ${:0.2f}".format(predicted_fare))
	print("RMSE: ${:0.2f}".format(rmse))

# Run some random predictions
# predict_random(df_prescaled, X_test, model)
# print(f'\n')
# predict_random(df_prescaled, X_test, model)
# print(f'\n')
# predict_random(df_prescaled, X_test, model)
# print(f'\n')
# predict_random(df_prescaled, X_test, model)
# print(f'\n')
# predict_random(df_prescaled, X_test, model)

train_pred = model.predict(X_train)
train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))

test_pred = model.predict(X_test)
test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
 
print("Train RMSE: {:0.2f}".format(train_rmse))
print("Test RMSE: {:0.2f}".format(test_rmse))
