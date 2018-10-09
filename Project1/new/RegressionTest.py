import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import linear_model
from imageio import imread
from sklearn.linear_model import LinearRegression
import random
import numpy as np
import scipy as scl

# I create the functions

# Franke Function
def FrankeFunction(x,y): 
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1)) 
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2)) 
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4

# OLS linear regression
def LinReg(x,y,z,degree,resampling):
    poly = PolynomialFeatures(degree=degree)
    data = poly.fit_transform(np.concatenate((x, y), axis=1))
    if resampling==True:
        print("------ OLS with resampling ------")
        sample_steps = 100
        Beta_boot = np.zeros([data.shape[1],sample_steps])
        Beta = np.zeros([data.shape[1],1])
        VarBeta = np.zeros([data.shape[1],1])
        pool = np.zeros(data.shape)
        z_sample = np.zeros([data.shape[0],1])
        for i in range(sample_steps):
            for k in range(data.shape[0]):
                index = np.random.randint(data.shape[0])
                pool[k] = data[index]
                z_sample[k] = z[index]
            U, D, Vt = scl.linalg.svd(pool)
            V = Vt.T    
            diag = np.zeros([V.shape[1],U.shape[1]])
            diagVar = np.zeros([V.shape[0],V.shape[1]])
            np.fill_diagonal(diag,D**(-1))
            np.fill_diagonal(diagVar,D**(-2))
            Beta_sample = V.dot(diag).dot(U.T).dot(z) 
            Beta_boot.T[i] = Beta_sample.T
        for i in range(data.shape[1]):
            Beta[i] = np.mean(Beta_boot[i])
            VarBeta[i] = np.var(Beta_boot[i])
    else:
        print("------ OLS without resampling ------")
        U, D, Vt = scl.linalg.svd(data)
        V = Vt.T
        diag = np.zeros([V.shape[1],U.shape[1]])
        diagVar = np.zeros([V.shape[0],V.shape[1]])
        np.fill_diagonal(diag,D**(-1))
        np.fill_diagonal(diagVar,D**(-2))
        Beta = V.dot(diag).dot(U.T).dot(z)
        z_model = data.dot(Beta)
        MSE = np.sum( (z_model - z)**2 )/data.shape[0]
        VarBeta = MSE*(np.diag(V.dot(diagVar).dot(Vt))[np.newaxis]).T
    return Beta, VarBeta

def Ridge(x, y, z, biasR, degree, resampling):
    poly = PolynomialFeatures(degree=degree)
    data = poly.fit_transform(np.concatenate((x, y), axis=1))
    if resampling==True:
        print("------ RIDGE with resampling ------")
        sample_steps = 10
        VarBeta = np.zeros([data.shape[1],1])
        Beta_boot = np.zeros([data.shape[1],sample_steps])
        Beta = np.zeros([data.shape[1],1])
        pool = np.zeros(data.shape)
        z_sample = np.zeros([data.shape[0],1])
        for i in range(sample_steps):
            for k in range(data.shape[0]):
                index = np.random.randint(data.shape[0])
                pool[k] = data[index]
                z_sample[k] = z[index]
            H = data.T .dot(data)
            Beta_sample = scl.linalg.inv(H + biasR*np.eye(H.shape[0])) .dot(data.T) .dot(z)
            Beta_boot.T[i] = Beta_sample.T
        for i in range(data.shape[1]):
            Beta[i] = np.mean(Beta_boot[i]) 
            VarBeta[i] = np.var(Beta_boot[i])
    else:
        print("------ RIDGE without resampling ------")
        H = data.T .dot(data)
        Beta = scl.linalg.inv(H + biasR*np.eye(H.shape[1])) .dot(data.T) .dot(z) 
        z_model = data.dot(Beta)
        MSE = np.sum( (z_model - z)**2 )/data.shape[0]
        VarBeta = MSE * np.diag((H + biasR*H.shape[1]) .dot(H) .dot((H + biasR*H.shape[1]).T)) 
    return Beta, VarBeta

def Lasso(x, y, z, biasL, degree, resampling):
    poly = PolynomialFeatures(degree=degree)
    data = poly.fit_transform(np.concatenate((x, y), axis=1))
    lasso_reg = linear_model.Lasso(alpha=biasL, fit_intercept=False)
    if resampling==True:
        print("------ LASSO with resampling ------")
        sample_steps = 10
        VarBeta = np.zeros([data.shape[1],1])
        Beta_boot = np.zeros([data.shape[1],sample_steps])
        Beta = np.zeros([data.shape[1],1])
        pool = np.zeros(data.shape)
        z_sample = np.zeros([data.shape[0],1])
        for i in range(sample_steps):
            for k in range(data.shape[0]):
                index = np.random.randint(data.shape[0])
                pool[k] = data[index]
                z_sample[k] = z[index]
            lasso_reg.fit(pool,z_sample)
            Beta_sample = lasso_reg.coef_
            Beta_boot.T[i] = Beta_sample.T
        for i in range(data.shape[1]):
            Beta[i] = np.mean(Beta_boot[i]) 
            VarBeta[i] = np.var(Beta_boot[i])
    else:
        print("------ LASSO without resampling ------")
        lasso_reg.fit(data,z)
        H = data.T .dot(data)
        Blasso = lasso_reg.coef_
        Beta = np.zeros((len(Blasso),1))
        VarBeta = np.zeros((len(Blasso),1))
        for i in range(len(Blasso)):
            Beta[i] = Blasso[i]
        tmp = np.eye(len(H))
        check = 0
        for j in range(len(Beta)):
            if Beta[j] != 0:
                tmp[j,j] = 1/np.abs(Beta[i])
            else:
                tmp[j,j] = 0
                check = 1
        if check == 0:
            z_model = data.dot(Beta)
            MSE = np.sum( (z_model - z)**2 )/data.shape[0]
            VarBeta = MSE*np.diag(scl.linalg.inv(H + biasL*tmp) .dot(H) .dot(scl.linalg.inv(H + biasL*tmp)))
    return Beta, VarBeta

# initialize seed
np.random.seed(42) # Life, Universe and Everything

# Producing Data set for the Franke function
size = 500
x = np.random.rand(size,1)
y = np.random.rand(size,1)
z = FrankeFunction(x,y) + 0.1*np.random.randn(size,1)

# OLS regression 
max_degree = 5
mse = [0]*max_degree
R2 = [0]*max_degree
Beta = {}
VarBeta = {}
for i in range(max_degree):
	degree = i+1
	Beta[str(i)], VarBeta[str(i)] = LinReg(x,y,z,degree,False)

	poly = PolynomialFeatures(degree=degree)
	data = poly.fit_transform(np.concatenate((x, y), axis=1))
	z_model = data.dot(Beta[str(i)])

	mse[i] = np.sum( (z_model - z)**2 )/size
	R2[i] = 1 - np.sum( (z_model - z)**2 )/np.sum( (z - np.mean(z))**2 )
	var = np.sum( (z_model - np.mean(z_model))**2 )/size
	bias = np.sum( (z - np.mean(z_model))**2 )/size
	print("------ Degree: ",degree)
	print("mse: %g\nR2: %g"%(mse[i], R2[i]))
	print("variance: %g"%var)
	print("bias: %g\n"%bias)
	print("----Parameters with uncertainties:----")
	for j in range(len(Beta[str(i)])):
		print(Beta[str(i)][j]," +/- ",np.sqrt(VarBeta[str(i)][j]))

# Plot parameters
for j in range(max_degree):
	xaxis = np.linspace(1,len(Beta[str(j)]),len(Beta[str(j)]))
	plt.figure()
	plt.plot(xaxis,Beta[str(j)],linestyle=':',marker='o')
	plt.grid()
	title = r"OLS regression - weights $\beta_j$ for "+str(j+1)+". degree polynomial"
	plt.title(title)
	plt.xlabel(r"$j$th parameter")
	plt.ylabel(r"$\beta_j$")
	plt.savefig("franke/OLS/OLS_parameters"+str(j+1))
	#plt.show()

# Plot MSE nd R2 score 
plt.figure()
plt.subplot(211)
plt.title(r"OLS with resampling - $MSE$ and $R^2$ score")
degree = np.linspace(1,5,5)
plt.plot(degree, mse)
axs = plt.gca()
axs.set_ylabel(r"$MSE$")
plt.grid()
plt.subplot(212)
plt.plot(degree, R2)
axs = plt.gca()
axs.set_ylabel(r"$R^2$ score")
axs.set_xlabel(r"poly degree")
plt.grid()
plt.savefig("franke/OLS/OLS_err")
#plt.show()

# OLS regression with resampling
max_degree = 5
mse = [0]*max_degree
R2 = [0]*max_degree
Beta = {}
VarBeta = {}
for i in range(max_degree):
	degree = i+1
	Beta[str(i)], VarBeta[str(i)] = LinReg(x,y,z,degree,True)

	poly = PolynomialFeatures(degree=degree)
	data = poly.fit_transform(np.concatenate((x, y), axis=1))
	z_model = data.dot(Beta[str(i)])

	mse[i] = np.sum( (z_model - z)**2 )/size
	R2[i] = 1 - np.sum( (z_model - z)**2 )/np.sum( (z - np.mean(z))**2 )
	var = np.sum( (z_model - np.mean(z_model))**2 )/size
	bias = np.sum( (z - np.mean(z_model))**2 )/size
	print("------ Degree: ",degree)
	print("mse: %g\nR2: %g"%(mse[i], R2[i]))
	print("variance: %g"%var)
	print("bias: %g\n"%bias)
	print("----Parameters with uncertainties:----")
	for j in range(len(Beta[str(i)])):
		print(Beta[str(i)][j]," +/- ",np.sqrt(VarBeta[str(i)][j]))

# Plot parameters
for j in range(max_degree):
	xaxis = np.linspace(1,len(Beta[str(j)]),len(Beta[str(j)]))
	plt.figure()
	plt.plot(xaxis,Beta[str(j)],linestyle=':',marker='o')
	plt.grid()
	title = r"OLS fit with resampling - weights $\beta_j$ for "+str(j+1)+". degree polynomial"
	plt.title(title)
	plt.xlabel(r"$j$th parameter")
	plt.ylabel(r"$\beta_j$")
	plt.savefig("franke/OLS/OLS_boot_parameters"+str(j+1))
	#plt.show()

# Plot MSE nd R2 score 
plt.figure()
plt.subplot(211)
plt.title(r"OLS with resampling - $MSE$ and $R^2$ score")
degree = np.linspace(1,5,5)
plt.plot(degree, mse)
axs = plt.gca()
axs.set_ylabel(r"$MSE$")
plt.grid()
plt.subplot(212)
plt.plot(degree, R2)
axs = plt.gca()
axs.set_ylabel(r"$R^2$ score")
axs.set_xlabel(r"poly degree")
plt.grid()
plt.savefig("franke/OLS/OLS_boot_err")
#plt.show()

"""
# ------------------------------------------------------------------------------------------------
# ridge regression 
noise = [0.001, 0.01, 0.1, 0.5]
for n in range(len(noise)):
	z = FrankeFunction(x,y) + noise[n]*np.random.randn(size,1)
	parameter = [1e-8, 1e-6, 1e-4, 1e-2]
	max_degree = 5
	mse = [0]*max_degree
	R2 = [0]*max_degree
	Beta = {}
	VarBeta = {}
	print("----- Ridge regression - NOISE FACTOR: ",noise[n])
	for i in range(max_degree):
		degree = i+1
		for k in range(len(parameter)):
			print("lambda: ",parameter[k])
			Beta[str(i)], VarBeta[str(i)] = Ridge(x,y,z,parameter[k],degree,False)

			poly = PolynomialFeatures(degree=degree)
			data = poly.fit_transform(np.concatenate((x, y), axis=1))
			z_model = data.dot(Beta[str(i)])

			mse[i] = np.sum( (z_model - z)**2 )/size
			R2[i] = 1 - np.sum( (z_model - z)**2 )/np.sum( (z - np.mean(z))**2 )
			var = np.sum( (z_model - np.mean(z_model))**2 )/size
			bias = np.sum( (z - np.mean(z_model))**2 )/size
			print("------ Degree: ",degree)
			print("mse: %g\nR2: %g"%(mse[i], R2[i]))
			print("variance: %g"%var)
			print("bias: %g\n"%bias)
			print("----Parameters with uncertainties:----")
			for j in range(len(Beta[str(i)])):
				print(Beta[str(i)][j]," +/- ",np.sqrt(VarBeta[str(i)][j]))


# ridge regression with resampling
noise = [0.001, 0.01, 0.1, 0.5]
for n in range(len(noise)):
	z = FrankeFunction(x,y) + noise[n]*np.random.randn(size,1)
	parameter = [ 1e-8, 1e-6, 1e-4, 1e-2]
	max_degree = 5
	mse = [0]*max_degree
	R2 = [0]*max_degree
	Beta = {}
	VarBeta = {}
	print("----- Ridge regression - NOISE FACTOR: ",noise[n])
	for i in range(max_degree):
		degree = i+1
		for k in range(len(parameter)):
			print("\lambda: ",parameter[k])
			Beta[str(i)], VarBeta[str(i)] = Ridge(x,y,z,parameter[k],degree,True)

			poly = PolynomialFeatures(degree=degree)
			data = poly.fit_transform(np.concatenate((x, y), axis=1))
			z_model = data.dot(Beta[str(i)])

			mse[i] = np.sum( (z_model - z)**2 )/size
			R2[i] = 1 - np.sum( (z_model - z)**2 )/np.sum( (z - np.mean(z))**2 )
			var = np.sum( (z_model - np.mean(z_model))**2 )/size
			bias = np.sum( (z - np.mean(z_model))**2 )/size
			print("------ Degree: ",degree)
			print("mse: %g\nR2: %g"%(mse[i], R2[i]))
			print("variance: %g"%var)
			print("bias: %g\n"%bias)
			print("----Parameters with uncertainties:----")
			for j in range(len(Beta[str(i)])):
				print(Beta[str(i)][j]," +/- ",np.sqrt(VarBeta[str(i)][j]))


# ------------------------------------------------------------------------------------------------

# Lasso regression 
noise = [0.001, 0.01, 0.1, 0.5]
for n in range(len(noise)):
	z = FrankeFunction(x,y) + noise[n]*np.random.randn(size,1)
	parameter = [ 1e-8, 1e-6, 1e-4, 1e-2]
	max_degree = 5
	mse = [0]*max_degree
	R2 = [0]*max_degree
	Beta = {}
	VarBeta = {}
	print("----- Ridge regression - NOISE FACTOR: ",noise[n])
	for i in range(max_degree):
		degree = i+1
		for k in range(len(parameter)):
			print("\lambda: ",parameter[k])
			Beta[str(i)], VarBeta[str(i)] = Lasso(x,y,z,parameter[k],degree,False)

			poly = PolynomialFeatures(degree=degree)
			data = poly.fit_transform(np.concatenate((x, y), axis=1))
			z_model = data.dot(Beta[str(i)])

			mse[i] = np.sum( (z_model - z)**2 )/size
			R2[i] = 1 - np.sum( (z_model - z)**2 )/np.sum( (z - np.mean(z))**2 )
			var = np.sum( (z_model - np.mean(z_model))**2 )/size
			bias = np.sum( (z - np.mean(z_model))**2 )/size
			print("------ Degree: ",degree)
			print("mse: %g\nR2: %g"%(mse[i], R2[i]))
			print("variance: %g"%var)
			print("bias: %g\n"%bias)
			print("----Parameters with uncertainties:----")
			for j in range(len(Beta[str(i)])):
				print(Beta[str(i)][j]," +/- ",np.sqrt(VarBeta[str(i)][j]))

# Lasso regression with resampling
noise = [0.001, 0.01, 0.1, 0.5]
for n in range(len(noise)):
	z = FrankeFunction(x,y) + noise[n]*np.random.randn(size,1)
	parameter = [ 1e-8, 1e-6, 1e-4, 1e-2]
	max_degree = 5
	mse = [0]*max_degree
	R2 = [0]*max_degree
	Beta = {}
	VarBeta = {}
	print("----- Ridge regression - NOISE FACTOR: ",noise[n])
	for i in range(max_degree):
		degree = i+1
		for k in range(len(parameter)):
			print("\lambda: ",parameter[k])
			Beta[str(i)], VarBeta[str(i)] = Lasso(x,y,z,parameter[k],degree,False)

			poly = PolynomialFeatures(degree=degree)
			data = poly.fit_transform(np.concatenate((x, y), axis=1))
			z_model = data.dot(Beta[str(i)])

			mse[i] = np.sum( (z_model - z)**2 )/size
			R2[i] = 1 - np.sum( (z_model - z)**2 )/np.sum( (z - np.mean(z))**2 )
			var = np.sum( (z_model - np.mean(z_model))**2 )/size
			bias = np.sum( (z - np.mean(z_model))**2 )/size
			print("------ Degree: ",degree)
			print("mse: %g\nR2: %g"%(mse[i], R2[i]))
			print("variance: %g"%var)
			print("bias: %g\n"%bias)
			print("----Parameters with uncertainties:----")
			for j in range(len(Beta[str(i)])):
				print(Beta[str(i)][j]," +/- ",np.sqrt(VarBeta[str(i)][j]))
"""