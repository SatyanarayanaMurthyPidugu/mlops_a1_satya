# misc.py 

import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import mean_squared_error

def load_data(): 
	import pandas as pd 
	import numpy as np 
	data_url = "http://lib.stat.cmu.edu/datasets/boston" 
	raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None) 
	# now we split this into data and target 
	data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]]) 
	target = raw_df.values[1::2, 2] 
	# These are the Feature names based on the original dataset 
	feature_names = [ 
	'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 
	'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT' 
	] 
	# Create a DataFrame 
	df = pd.DataFrame(data, columns=feature_names) 
	df['MEDV'] = target # here MEDV is our target variable 
	return df 

def preprocess_data(df): 
	""" 
	Preprocesses the data by splitting it into features (X) and target (y), 
	and then into training and testing sets. 
	""" 
	X = df.drop('MEDV', axis=1) 
	y = df['MEDV'] 
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 
	return X_train, X_test, y_train, y_test 

def train_model(X_train, y_train, model): 
	""" 
	A generic function to train a machine learning model. """ 
	model.fit(X_train, y_train) 
	return model 

def evaluate_model(model, X_test, y_test): 
	""" Evaluates the model and returns the Mean Squared Error (MSE). """ 
	y_pred = model.predict(X_test) 
	mse = mean_squared_error(y_test, y_pred)
	return mse