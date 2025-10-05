# train2.py 
from sklearn.kernel_ridge import KernelRidge 
from misc import load_data, preprocess_data, train_model, evaluate_model 

def main(): 
	# Load and preprocess data 
	df = load_data() 
	X_train, X_test, y_train, y_test = preprocess_data(df) 

	# Initialize, train, and evaluate the Kernel Ridge model 
	kr_model = KernelRidge(alpha=1.0) 
	# alpha is a key hyperparameter 
	kr_model_trained = train_model(X_train, y_train, kr_model) 
	mse = evaluate_model(kr_model_trained, X_test, y_test) 

	# Display the average MSE score [cite: 25] 
	print(f"Kernel Ridge Regressor - Average MSE: {mse}") 

if __name__ == "__main__": 
	main()