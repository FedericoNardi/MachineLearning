import numpy as np
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def IsingEnergies(states,L):
	J=np.zeros((L,L),)
	for i in range(L):
		J[i,(i+1)%L] -= 1.0
	return np.einsum('...i,ij,...j->...',states,J,states)

def sigmoid(x):
	return 1/(1+np.exp(-x))

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
		exp_term = np.exp(self.z_o)
		self.probabilities = exp_term/np.sum(exp_term,axis=1,keepdims=True)
		


	def FeedForward_out(self,X):
	    # weighted sum of inputs to the hidden layer
	    z_h = np.matmul(X, self.hidden_weights) + self.hidden_bias
	    # activation in the hidden layer
	    a_h = sigmoid(z_h)
	    # weighted sum of inputs to the output layer
	    z_o = np.matmul(a_h, self.output_weights) + self.output_bias

	    exp_term = np.exp(z_o)
	    probabilities = exp_term/(np.sum(exp_term,axis=1,keepdims=True))

	    return probabilities

	def BackPropagation(self):
		error_output = self.z_o - np.reshape(self.Ydata,(len(self.Ydata),1))
		error_hidden = np.matmul(error_output, self.output_weights.T) * self.a_h * (1 - self.a_h)
		#print(error_output)
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
		probabilities = self.FeedForward_out(X)
		return np.argmax(probabilities, axis=1)

	def predict_probabilities(self, X):
		probabilities = self.FeedForward_out(X)
		return probabilities

#================================================================================================================

# Set lattice and import configurations

# initialize seed
np.random.seed(42) 

# Ising model parameters
L=40 # linear system size
J=-1.0 # Ising interaction
T=np.linspace(0.25,4.0,16) # set of temperatures
T_c=2.26 # Onsager critical temperature in the TD limit
###### define ML parameters
num_classes=2
train_to_test_ratio=0.01 # training samples

# path to data directory
#path_to_data="C:\Anaconda\Programes/IsingData/"

import pickle
# load data
file_name = "IsingData/Ising2DFM_reSample_L40_T=All.pkl" # this file contains 16*10000 samples taken in T=np.arange(0.25,4.0001,0.25)
#data = pickle.load(open(path_to_data+file_name,'rb'))
data = pickle.load(open(file_name,'rb')) # pickle reads the file and returns the Python object (1D array, compressed bits)
data = np.unpackbits(data).reshape(-1, 1600) # Decompress array and reshape for convenience
data=data.astype('int')
data[np.where(data==0)]=-1 # map 0 state to -1 (Ising variable can take values +/-1)

file_name = "IsingData/Ising2DFM_reSample_L40_T=All_labels.pkl" # this file contains 16*10000 samples taken in T=np.arange(0.25,4.0001,0.25)
#slabels = pickle.load(open(path_to_data+file_name,'rb')) # pickle reads the file and returns the Python object (here just a 1D array with the binary labels)
labels = pickle.load(open(file_name,'rb'))
#-------------------------------------------------------------------------------
# split data into train and test

# divide data into ordered, critical and disordered
X_ordered=data[:70000,:]
Y_ordered=labels[:70000]

X_critical=data[70000:100000,:]
Y_critical=labels[70000:100000]

X_disordered=data[100000:,:]
Y_disordered=labels[100000:]

# reduce size of data by randomly selecting
def reduce(X,Y,size):
	indices = np.arange(X.shape[0])
	chosen = np.random.choice(indices, size=size, replace=False)
	return X[chosen,:], Y[chosen]

X_ordered,Y_ordered = reduce(X_ordered, Y_ordered, 7000)
X_critical,Y_critical = reduce(X_critical, Y_critical,3000)
X_disordered,Y_disordered = reduce(X_disordered, Y_disordered, 6000)

del data,labels

# define training and test data sets
X=np.concatenate((X_ordered,X_disordered))
Y=np.concatenate((Y_ordered,Y_disordered))

# pick random data points from ordered and disordered states 
# to create the training and test sets
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,train_size=0.7)

# full data set
X=np.concatenate((X_critical,X))
Y=np.concatenate((Y_critical,Y))

print('X_train shape:', X_train.shape)
print('Y_train shape:', Y_train.shape)
print()
print(X_train.shape[0], 'train samples')
print(X_critical.shape[0], 'critical samples')
print(X_test.shape[0], 'test samples')

#-----------------------------------------------------------------------------------------

# set up neural network

eta_vals = [0.001, 0.0001, 0.00001]#np.logspace(-2, -1, 2)
lmbd_vals = [1.0, 0.1, 0.01, 0.001]#np.logspace(-6, -5, 2)
"""
epochs = 100
batch_size = 100

train_accuracy = np.zeros((len(eta_vals), len(lmbd_vals)))
test_accuracy = np.zeros((len(eta_vals), len(lmbd_vals)))

for i, eta in enumerate(eta_vals):
	for j, lmbd in enumerate(lmbd_vals):

		DNN = NeuralNetwork(
		X_train,Y_train,
		eta=eta,
		lmbd=lmbd,
		epochs=epochs,
		batch_size=batch_size,
		n_hidden_neurons=60,
		n_categories=2
		)
		DNN.train()
		train_pred = DNN.predict(X_train)
		test_pred = DNN.predict(X_test)

		train_accuracy[i][j] = accuracy_score(Y_train, train_pred)
		test_accuracy[i][j] = accuracy_score(Y_test, test_pred)

		print("Lambda: ",lmbd)
		print("Eta: ",eta)



# Show search results
import seaborn as sns

sns.set()

fig, ax = plt.subplots(figsize = (10, 10))
sns.heatmap(train_accuracy, annot=True, ax=ax, cmap="viridis")
ax.set_title(r"accuracy score on training data",fontsize=16)
ax.set_ylabel("$\eta$",fontsize=13)
ax.set_xlabel("$\lambda$",fontsize=13)
plt.savefig("figures/MLPclass_train")
plt.show()

fig, ax = plt.subplots(figsize = (10, 10))
sns.heatmap(test_accuracy, annot=True, ax=ax, cmap="viridis")
ax.set_title(r"accuracy score on test data",fontsize=16)
ax.set_ylabel("$\eta$",fontsize=13)
ax.set_xlabel("$\lambda$",fontsize=13)
plt.savefig("figures/MLPclass_test")
plt.show()


"""
eta= 0.001
lmbd= 0.1

epochs = 100
batch_size = 100

DNN = NeuralNetwork(
		X_train,Y_train,
		eta=eta,
		lmbd=lmbd,
		epochs=epochs,
		batch_size=batch_size,
		n_hidden_neurons=60,
		n_categories=2
		)
DNN.train()
train_pred = DNN.predict(X_train)
test_pred = DNN.predict(X_test)

print(accuracy_score(Y_train,train_pred))
print(accuracy_score(Y_test,test_pred))
