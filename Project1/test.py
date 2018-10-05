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
	return np.sum( (model[i] - prediction[i])**2 for i in range(len(model)) )


# Function for linear regression
def LinearRegression(x,y,z,degree, model, resampling):
	poly = PolynomialFeatures(degree=degree)
	DataSet = poly.fit_transform(np.concatenate((x,y), axis=1))
	if resampling == "True":
		MeanSquareError = 0
		rSquareScore = 0
		sample_steps = 500
		VarBeta = np.array([[0.] for i in range(DataSet.shape[1])])
		Beta = np.array([[0.] for i in range(DataSet.shape[1])])
		for i in range(sample_steps):
			pool = np.array([DataSet[random.randrange(DataSet.shape[0])].tolist() for i in range(DataSet.shape[0])])
			H = (pool.T).dot(pool)
			Beta_sample = scl.linalg.inv(H).dot(pool.T).dot(z)
			Beta += Beta_sample
			z_fit = pool.dot(Beta_sample)
			MeanSquareError_sample = MSE(z_fit, model)			
			MeanSquareError += MeanSquareError_sample
			VarBeta += np.array([[np.diag(H .dot(MeanSquareError_sample * np.eye(H.shape[1])))[i]] for i in range(DataSet.shape[1])])
			rSquareScore += r2score(z_fit, model)
		MeanSquareError = MeanSquareError/sample_steps
		rSquareScore = rSquareScore/sample_steps
		Beta = Beta/sample_steps
		VarBeta = VarBeta/sample_steps
		print('gotcha')
	else:
		H = (DataSet.T).dot(DataSet)
		Beta = scl.linalg.inv(H).dot(DataSet.T).dot(z)
		z_fit = DataSet .dot(Beta)
		MeanSquareError = MSE(z_fit, model)
		VarBeta = np.diag(H .dot(MeanSquareError * np.eye(H.shape[1])))
		print('hey madaffukka')
		rSquareScore = r2score(z_fit, model)
	print("-------- fit with",degree,"th degree polynomial --------")
	print("MSE: ",MeanSquareError,"\nr2 score: ", rSquareScore)
	return MeanSquareError, rSquareScore, Beta, VarBeta


# Generate data set
# Initialize seed
random.seed(0)
size = 500
x = (np.array([random.random() for i in range(size)])[np.newaxis]).T
y = (np.array([random.random() for i in range(size)])[np.newaxis]).T
noise = 0.1*(np.array([random.random() for i in range(size)])[np.newaxis]).T
z = FrankeFunction(x, y) + noise
z_franke = FrankeFunction(x,y)


#MeshplotFranke(x,y)

MeanSquareLS = [0]*5
rSquareLS = [0]*5
MeanSquareLS_res = [0]*5
rSquareLS_res = [0]*5

#for k in range(1):
k=4
random.seed(1)
MeanSquareLS[k], rSquareLS[k], BetaLS, VarBetaLS = LinearRegression(x,y,z,k+1,z_franke,"False")
MeanSquareLS_res[k], rSquareLS_res[k], BetaLS_res, VarBetaLS_res = LinearRegression(x,y,z,k+1,z_franke,"True")
"""
plt.plot(BetaLS,VarBetaLS,linestyle='none',marker='o')
plt.plot(VarBetaLS_res,VarBetaLS_res, linestyle='none',marker='o')
plt.legend([r"OLS",r"OLS with resampling"])
axs=plt.gca()
plt.show()
"""

	









