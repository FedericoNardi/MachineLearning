import numpy as np
import random
import scipy as scl
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model

# compute Franke function
def FrankeFunction(x,y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4

# function for Plotting the Franke function
def MeshplotFranke(x,y):
	fig = plt.figure()
	axs = fig.gca(projection='3d')
	x_plt, y_plt = np.meshgrid(sorted(list(x)),sorted(list(y)))
	z_plt = FrankeFunction(x_plt,y_plt) + 0.5*np.array([[random.random() for i in range(size)] for k in range(size)])
	print(z_plt.shape)
	surf = axs.plot_surface(x_plt,y_plt,z_plt, cmap=cm.viridis, linewidth=0, antialiased=False) #cm.coolwarm
	axs.set_zlim(-0.10,1.40)
	axs.set_xlabel(r'xlabel')
	axs.set_ylabel(r'ylabel')
	axs.set_zlabel(r'zlabel')
	axs.zaxis.set_major_locator(LinearLocator(10))
	axs.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
	fig.colorbar(surf, shrink=0.5, aspect=5)
	plt.show()

# r^2 score function
def r2score(prediction, model):
	avg = np.sum( model[i] for i in range(len(model)) )/len(model)
	numerator = np.sum( (model[i] - prediction[i])**2 for i in range(len(model)) )
	denominator = np.sum( (model[i] - avg)**2 for i in range(len(model)) )
	return 1 - (numerator/denominator)

# Mean square error function
def MSE(prediction, model):
	return np.sum( (model[i] - prediction[i])**2 for i in range(len(model)) )/len(model)


# Function for linear regression
def LinearRegression(x,y,z,degree, model, resampling):
	poly = PolynomialFeatures(degree=degree)
	DataSet = poly.fit_transform(np.concatenate((x,y), axis=1))
	if resampling == "True":
		MeanSquareError = 0
		rSquareScore = 0
		sample_steps = 10
		VarBeta = np.zeros([DataSet.shape[1],1])
		Beta_boot = np.zeros([DataSet.shape[1],sample_steps])
		Beta = np.zeros([DataSet.shape[1],1])
		pool = np.zeros(DataSet.shape)
		z_sample = np.zeros([DataSet.shape[0],1])
		for i in range(sample_steps):
			for k in range(DataSet.shape[0]):
				index = np.random.randint(DataSet.shape[0])
				pool[k] = DataSet[index]
				z_sample[k] = z[index]	
			H = (pool.T).dot(pool)
			Beta_sample = scl.linalg.inv(H).dot(pool.T).dot(z_sample)
			Beta_boot.T[i] = Beta_sample.T
		for i in range(DataSet.shape[1]):
			Beta[i] = np.mean(Beta_boot[i]) 
			VarBeta[i] = np.var(Beta_boot[i])
			print(Beta_boot[1:4])
			foo 
		z_fit = DataSet.dot(Beta)
		MeanSquareError_sample = MSE(z_fit, model)	
		rSquareScore = r2score(z_fit, model)
	else:
		H = (DataSet.T).dot(DataSet)
		Hinv = scl.linalg.inv(H)
		Beta = Hinv.dot(DataSet.T).dot(z)
		z_fit = DataSet .dot(Beta)
		MeanSquareError = MSE(z_fit, model)
		VarBeta = np.array(np.diag(Hinv) *MeanSquareError * len(model)/(len(model)-DataSet.shape[1]-1))[np.newaxis] # Make estimator unbiased
		VarBeta = VarBeta.T
		rSquareScore = r2score(z_fit, model)
	print("-------- fit with",degree,"th degree polynomial --------")
	print("MSE: ",MeanSquareError,"\nr2 score: ", rSquareScore)
	return MeanSquareError, rSquareScore, Beta, VarBeta




# Initialize seed
np.random.seed(42) #Life, Universe and Everything

# Generate data set
size = 10
x = np.random.rand(size,1)
y = np.random.rand(size,1)
noise = 0.1*np.random.rand(size,1)
z = FrankeFunction(x,y) + noise
z_franke = FrankeFunction(x,y)


#MeshplotFranke(x,y)

MeanSquareLS = [0]*5
rSquareLS = [0]*5
MeanSquareLS_res = [0]*5
rSquareLS_res = [0]*5

#for k in range(1):
k=4
MeanSquareLS[k], rSquareLS[k], BetaLS, VarBetaLS = LinearRegression(x,y,z,k+1,z_franke,"False")
MeanSquareLS_res[k], rSquareLS_res[k], BetaLS_res, VarBetaLS_res = LinearRegression(x,y,z,k+1,z_franke,"True")
print("NO RESAMPLING:","\n",BetaLS)
print("RESAMPLING:","\n",BetaLS_res)
"""
plt.plot(BetaLS,VarBetaLS,linestyle='none',marker='o')
plt.plot(VarBetaLS_res,VarBetaLS_res, linestyle='none',marker='o')
plt.legend([r"OLS",r"OLS with resampling"])
axs=plt.gca()
plt.show()
"""

	



