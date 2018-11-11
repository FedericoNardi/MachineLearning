import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split

def IsingEnergies(states,L):
	J=np.zeros((L,L),)
	for i in range(L):
		J[i,(i+1)%L] -= 1.0
	return np.einsum('...i,ij,...j->...',states,J,states)

def sigmoid(x):
	return 1/(1+np.exp(-x))

def FeedForward(input):
    # weighted sum of inputs to the hidden layer
    z_h = np.matmul(input, hidden_weights) + hidden_bias
    # activation in the hidden layer
    a_h = sigmoid(z_h)
    
    # weighted sum of inputs to the output layer
    z_o = np.matmul(a_h, output_weights) + output_bias
    
    return a_h, z_o

def BackPropagation(X,Y):
	a_h, z_o = FeedForward(X)
	error_output = z_o - np.reshape(Y,(len(Y),1))
	error_hidden = np.matmul(error_output, output_weights.T) * a_h * (1 - a_h)

	#output layer gradients
	output_weights_gradient = np.matmul(a_h.T,error_output)
	output_bias_gradient = np.sum(error_output,axis=0)

	#hidden layer gradients
	hidden_weights_gradient = np.matmul(X.T, error_hidden)
	hidden_bias_gradient = np.sum(error_hidden, axis=0)

	return output_weights_gradient, output_bias_gradient, hidden_weights_gradient, hidden_bias_gradient

#===========================================================================================

# initialize seed 
np.random.seed(42)

# define the lattice
L=40 #size of system

states=np.random.choice([-1,1], size=(10000,L))
energies = IsingEnergies(states,L)

# Splitting data into train and test
train_size = 0.75
test_size=1-train_size

states_train, states_test, energies_train, energies_test = train_test_split(states, energies, train_size=train_size, test_size=test_size)

print("---- Training over",states_train.shape[0],"spin configurations ----")
print("States per configuration: ",states_train.shape[1])
print("---- Test over",states_test.shape[0],"spin configurations ----")
print("States per configuration: ",states_test.shape[1])
"""
# Building the neural network
n_inputs, n_features = states_train.shape
n_hidden_neurons=42
n_categories = 1

# initialize weights and biases
hidden_weights = np.random.randn(n_features, n_hidden_neurons) # randomly distributed
hidden_bias = np.zeros(n_hidden_neurons) + 0.01

output_weights = np.random.randn(n_hidden_neurons, n_categories)
output_bias = np.zeros(n_categories) + 0.01

# set up the training
eta = 0.01
lmbd = 0.01

for i in range(1000):
	dWo, dBo, dWh, dBh = BackPropagation(states_train, energies_train)

	# regularization terms gradients
	dWo += lmbd*output_weights
	dWh += lmbd*output_bias

	# update weights and biases
	output_weights -= eta*dWo
	output_bias -= eta*dBo
	hidden_weights -= eta*dWh
	hidden_bias -= eta*dBh
"""

class NeuralNetwork:
	def __init__(
		self,
		Xdata,
		Ydata,
		n_hidden_neurons=50,
		n_categories=1,
		epochs=10,
		batch_size=100,
		eta=0.1,
		lmbd=0.0,

	):
		self.Xdata_full = Xdata
		self.Ydata_full = Ydata

		self.n_inputs = Xdata.shape[0]
		self.n_features = Xdata.shape[1]
		self.n_hidden_neurons=n_hidden_neurons
		self.n_categories = n_categories

		self.epochs = epochs
		self.batch_size = batch_size
		self.iterations = self.n_inputs//self.batch_size
		self.eta = eta
		self.lmbd = lmbd

		self.create_biases_and_weights()

	def create_biases_and_weights(self):
		self.hidden_weights = np.random.randn(self.n_features,self.n_hidden_neurons)
		self.hidden_bias = np.zeros(self.n_hidden_neurons)+0.01
		self.output_weights = np.random.randn(self.n_hidden_neurons,self.n_categories)
		self.output_bias = np.zeros(self.n_categories)+0.01

	def FeedForward(self):
		# weighted sum of inputs to the hidden layer
		self.z_h = np.matmul(self.Xdata, self.hidden_weights) + self.hidden_bias
		# activation in the hidden layer
		self.a_h = sigmoid(self.z_h)			
		# weighted sum of inputs to the output layer
		self.z_o = np.matmul(self.a_h, self.output_weights) + self.output_bias
		


	def FeedForward_out(self,X):
	    # weighted sum of inputs to the hidden layer
	    z_h = np.matmul(X, self.hidden_weights) + self.hidden_bias
	    # activation in the hidden layer
	    a_h = sigmoid(z_h)
	    # weighted sum of inputs to the output layer
	    z_o = np.matmul(a_h, self.output_weights) + self.output_bias
	    return z_o

	def BackPropagation(self):
		error_output = self.z_o - np.reshape(self.Ydata,(len(self.Ydata),1))
		error_hidden = np.matmul(error_output, self.output_weights.T) * self.a_h * (1 - self.a_h)
		#output layer gradients
		self.output_weights_gradient = np.matmul(self.a_h.T,error_output)
		self.output_bias_gradient = np.sum(error_output,axis=0)
		#hidden layer gradients
		self.hidden_weights_gradient = np.matmul(self.Xdata.T, error_hidden)
		self.hidden_bias_gradient = np.sum(error_hidden, axis=0)

		if self.lmbd > 0:
			self.output_weights_gradient += self.lmbd*self.output_weights
			self.hidden_weights_gradient += self.lmbd*self.hidden_weights

		self.output_weights -= self.eta*self.output_weights_gradient
		self.output_bias -= self.eta*self.output_bias_gradient
		self.hidden_weights -= self.eta*self.hidden_weights_gradient
		self.hidden_bias -= self.eta*self.hidden_bias_gradient

	def train(self):
		data_indices = np.arange(self.n_inputs)

		for i in range(self.epochs):
			for j in range(self.iterations):
				# pick data points
				chosen_datapoints = np.random.choice(data_indices,size=self.batch_size,replace=False)
				# training on minibatch
				self.Xdata = self.Xdata_full[chosen_datapoints]
				self.Ydata = self.Ydata_full[chosen_datapoints]

				self.FeedForward()
				self.BackPropagation()

	def predict(self, X):
		model_prediction = self.FeedForward_out(X)
		return model_prediction




# set up neural network

epochs = 100
batch_size = 100

eta_vals = np.logspace(-6, 1, 2)
lmbd_vals = np.logspace(-6, 1, 2)

# store the models for later use
#DNN_models = np.zeros((len(eta_vals), len(lmbd_vals)), dtype=object)
r2_train = np.zeros([len(eta_vals), len(lmbd_vals)])
r2_test = np.zeros([len(eta_vals), len(lmbd_vals)])

# import metrics
from sklearn.metrics import mean_squared_error, r2_score

for i, eta in enumerate(eta_vals):
	for j, lmbd in enumerate(lmbd_vals):

		DNN = NeuralNetwork(
		states_train,energies_train,
		eta=eta,
		lmbd=lmbd,
		epochs=epochs,
		batch_size=batch_size,
		n_hidden_neurons=42,
		)
		DNN.train()
		train_pred = DNN.predict(states_train)
		test_pred = DNN.predict(states_test)
		print(train_pred,test_pred)

		r2_train = r2_score(energies_train,train_pred)
		r2_test = r2_score(energies_test,test_pred)

		print("Learning rate: ",eta)
		print("Lambda: ",lmbd)
