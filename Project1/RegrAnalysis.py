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
def LinearRegressionOLS(x,y,z,degree,resampling):
    poly = PolynomialFeatures(degree=degree)
    data = poly.fit_transform(np.concatenate((x, y), axis=1))
    if resampling==True:
        print("------ OLS with resampling ------")
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
        zlin = data .dot(Beta)
        MSE = mean_squared_error(z,zlin)
    else:
        print("------ OLS without resampling ------")
        U, D, Vt = scl.linalg.svd(data)
        V = Vt.T
        diag = np.zeros([V.shape[1],U.shape[1]])
        diagVar = np.zeros([V.shape[0],V.shape[1]])
        np.fill_diagonal(diag,D**(-1))
        np.fill_diagonal(diagVar,D**(-2))
        Beta = V.dot(diag).dot(U.T).dot(z) 
        H = data.T .dot(data)
        zlin = data .dot(Beta)
        MSE = mean_squared_error(z,zlin)
        VarBeta = MSE*(np.diag(V.dot(diagVar).dot(Vt))[np.newaxis]).T
    R2S = r2_score(z,zlin)
    print("-----Polynomial degree = ", degree,"-----")
    print("MSE %g" % MSE)
    print("RS2 %g" % R2S)
    error = np.mean( np.mean((z - zlin)**2, axis=1, keepdims=True) )
    bias = np.mean( (z - np.mean(zlin, axis=1, keepdims=True))**2 )
    variance = np.mean( np.var(zlin, axis=1, keepdims=True) )
    print("MSE")
    print(MSE)
    print("Error")
    print(error)
    print("Bias")
    print(bias)
    print("Variance")
    print(variance)
    return Beta, VarBeta, MSE, R2S


# Check OLS with SKlearn
def SKLcheckLS(x,y,z,degree):
    poly = PolynomialFeatures(degree=degree)
    data = poly.fit_transform(np.concatenate((x, y), axis=1))
    linreg = LinearRegression(fit_intercept=False)
    linreg.fit(data,z)
    return linreg

# K-fold Function
def k_fold(x, y, z, Pol_deg, method, biasLambda, k):
    n = len(x) 
    poly = PolynomialFeatures(degree=Pol_deg)
    data = poly.fit_transform(np.concatenate((x, y), axis=1))
    len_fold = n//k
    MSE = np.zeros((k,1))
    R2S = np.zeros((k,1))
    MSE_sampling = 0
    R2S_sampling = 0
    Beta_sampling = np.zeros((1,data.shape[1]))
    VarBeta_sampling = np.zeros((1,data.shape[1]))
    x_train= np.zeros((len_fold*(k-1),k))
    y_train = np.zeros((len_fold*(k-1),k))
    z_train = np.zeros((len_fold*(k-1),k))
    x_test = np.zeros((len_fold,k))
    y_test = np.zeros((len_fold,k))
    z_test = np.zeros((len_fold,k))
    z_regr  = np.zeros((len_fold,k))
    z_anal = np.zeros((n,k))
    for j in range(k):
        l = 0
        # Create training and test data
        for i in range(len_fold):
            x_test[i,j] = x[j*len_fold + i]
            y_test[i,j] = y[j*len_fold + i]
            z_test[i,j] = z[j*len_fold + i]
        for i in range(n):
            if (i < j*len_fold) or (i > (j + 1)*len_fold):
                x_train[l,j] = x[i]
                y_train[l,j] = y[i]
                z_train[l,j] = z[i]
                l = l + 1
        # Set up the regression
        if method == "OLS":
            MeanSquare_train, r2score_train, Beta_train, VarBeta_train = LinearRegressionOLS(x_train[:,[j]], y_train[:,[j]], z_train[:,[j]], Pol_deg,"False")
        else: 
            if method == "ridge":
                MeanSquare_train, r2score_train, Beta_train, VarBeta_train = Ridge(x_train[:,[j]], y_train[:,[j]], z_train[:,[j]], biasLambda, Pol_deg, "False")
            else:
               if method == "lasso":
                     MeanSquare_train, r2score_train, Beta_train, VarBeta_train = Lasso(x_train[:,[j]], y_train[:,[j]], z_train[:,[j]], biasLambda, Pol_deg, "False")
               else:
                     print("ERROR: method not recognized")
        # Now I do the validation of the method
        data_test = poly.fit_transform(np.concatenate((x_test[:,[j]], y_test[:,[j]]), axis=1))
        z_regr[:,[j]] = data_test .dot(np.transpose(Beta_train));
        MSE[j] = mean_squared_error(z_test[:,[j]], z_regr[:,[j]])
        R2S[j] = r2_score(z_test[:,[j]],z_regr[:,[j]])
        # Use the parameters obtained to make predictions and calculate MSE and R2score
        MSE_sampling = MSE_sampling + MSE[j]
        R2S_sampling = R2S_sampling + R2S[j]
        Beta_sampling = Beta_sampling + Beta_train
        VarBeta_sampling = VarBeta_sampling + VarBeta_train
        # I set up the data for the bias and the variance
        z_anal[:, [j]] = data .dot(np.transpose(Beta_train));
    # Take the mean value
    MSE_sampling = MSE_sampling/k
    R2S_sampling = R2S_sampling/k
    Beta_sampling = Beta_sampling/k
    VarBeta_sampling = VarBeta_sampling/k
    # Error, Bias, Variance
    error = np.mean( np.mean((z - z_anal)**2, axis=1, keepdims=True) )
    bias = np.mean( (z - np.mean(z_anal, axis=1, keepdims=True))**2 )
    variance = np.mean( np.var(z_anal, axis=1, keepdims=True) )
    print("MSE")
    print(MSE_sampling)
    print("Error")
    print(error)
    print("Bias")
    print(bias)
    print("Variance")
    print(variance)
    return MSE_sampling, R2S_sampling, Beta_sampling, VarBeta_sampling

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
        z_fit = data .dot(Beta)
        MSE = mean_squared_error(z,z_fit)
        R2score = r2_score(z,z_fit)
    else:
        print("------ RIDGE without resampling ------")
        H = data.T .dot(data)
        Beta = scl.linalg.inv(H + biasR*np.eye(H.shape[1])) .dot(data.T) .dot(z) 
        z_fit = data .dot(Beta)
        MSE = mean_squared_error(z,z_fit)
        VarBeta = MSE * np.diag((H + biasR*H.shape[1]) .dot(H) .dot((H + biasR*H.shape[1]).T))
        R2score = r2_score(z,z_fit)
    print("-------",degree,"th degree polynomial","-------")
    print(" MSE: ", MSE)
    print(" R2 score: ", R2score, "\n")
    error = np.mean( np.mean((z - z_fit)**2, axis=1, keepdims=True) )
    bias = np.mean( (z - np.mean(z_fit, axis=1, keepdims=True))**2 )
    variance = np.mean( np.var(z_fit, axis=1, keepdims=True) )
    print("MSE")
    print(MSE)
    print("Error")
    print(error)
    print("Bias")
    print(bias)
    print("Variance")
    print(variance)
    return MSE, R2score, np.transpose(Beta), VarBeta

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
        zlin = data .dot(Beta)
        MSE = mean_squared_error(z,zlin)
        R2score = r2_score(z,zlin)
    else:
        print("------ LASSO without resampling ------")
        lasso_reg.fit(data,z)
        H = data.T .dot(data)
        Blasso = lasso_reg.coef_
        Beta = np.zeros((len(Blasso),1))
        VarBeta = np.zeros((len(Blasso),1))
        for i in range(len(Blasso)):
            Beta[i] = Blasso[i]
        zlin = data .dot(Beta)
        MSE = mean_squared_error(z,zlin)
        R2score = r2_score(z,zlin)
        tmp = np.eye(len(H))
        check = 0
        for j in range(len(Beta)):
            if Beta[j] != 0:
                tmp[j,j] = 1/np.abs(Beta[i])
            else:
                tmp[j,j] = 0
                check = 1
        if check == 0:
            VarBeta = MSE*np.diag(scl.linalg.inv(H + biasL*tmp) .dot(H) .dot(scl.linalg.inv(H + biasL*tmp)))
    print("-------",degree,"th degree polynomial","-------")
    print(" MSE: ", MSE)
    print(" R2 score: ", R2score, "\n")
    error = np.mean( np.mean((z - zlin)**2, axis=1, keepdims=True) )
    bias = np.mean( (z - np.mean(zlin, axis=1, keepdims=True))**2 )
    variance = np.mean( np.var(zlin, axis=1, keepdims=True) )
    print("MSE")
    print(MSE)
    print("Error")
    print(error)
    print("Bias")
    print(bias)
    print("Variance")
    print(variance)
    return MSE, R2score, np.transpose(Beta), VarBeta

# initialize seed
np.random.seed(42) # Life, Universe and Everything

# Producing Data set for the Franke function
size = 500
x = np.random.rand(size,1)
y = np.random.rand(size,1)
z = FrankeFunction(x,y) + 0.1*np.random.randn(size,1)
print(x.shape, y.shape, z.shape)


# Load terrain

# terrain1 = imread('terrain1.tif')
# terrain1 = terrain1[0:100,0:100]
# plt.figure()
# plt.imshow(terrain1, cmap='gray')
# plt.savefig("fittingfunction")
# #plt.show()
# terrain1 = np.reshape(terrain1,-1)
# # Intialize grid
# size = 100
# x=(np.array([i for i in range(size)]*size)[np.newaxis]).T
# x=x/size

# y=[i for i in range(size)]
# y = (np.array(y*size)[np.newaxis]).T
# y=y/size
# z = (np.array(terrain1)[np.newaxis]).T
# print(x.shape,y.shape,z.shape)
 
# Fit with OLS without resampling
BetaOLS = {}
VarOLS = {}
MSElin = [0]*5
R2Slin = [0]*5
for k in range(5):
	degree = k+1
	BetaOLS[str(k)], VarOLS[str(k)], MSElin[k], R2Slin[k] = LinearRegressionOLS(x,y,z,degree, False )
	# Check with scikit-learn
	linreg = SKLcheckLS(x,y,z,degree)
	Check = linreg.coef_

# Plot OLS parameters
for j in range(5):
	xaxis = np.linspace(1,len(BetaOLS[str(j)]),len(BetaOLS[str(j)]))
	plt.figure()
	plt.plot(xaxis,BetaOLS[str(j)],linestyle=':',marker='o')
	plt.grid()
	title = r"OLS fit - weights $\beta_j$ for "+str(j+1)+". degree polynomial"
	plt.title(title)
	plt.xlabel(r"$j$th parameter")
	plt.ylabel(r"$\beta_j$")
	plt.savefig("figures/OLS/OLS_parameters"+str(j+1))
	#plt.show()

# Plot MSE nd R2 score for OLS
plt.figure()
plt.subplot(211)
plt.title(r"OLS without resampling - $MSE$ and $R^2$ score")
degree = np.linspace(1,5,5)
plt.plot(degree, MSElin)
axs = plt.gca()
axs.set_ylabel(r"$MSE$")
plt.grid()
plt.subplot(212)
plt.plot(degree, R2Slin)
axs = plt.gca()
axs.set_ylabel(r"$R^2$ score")
axs.set_xlabel(r"poly degree")
plt.grid()
plt.savefig("figures/OLS/OLS_err")
#plt.show()



# # Print a log to check parameters with scikit-learn
# print("------- Polynomial degree = ",degree, "-------")
# print("My fit parameters")
# print(Blin1)
# print(VarBlin1)
# print("SKlearn fit parameters")
# print(Blin1_check.T)


# Bootstrap resampling on OLS
BetaOLS = {}
VarOLS = {}
MSElin = [0]*5
R2Slin = [0]*5
for k in range(5):
	degree = k+1
	BetaOLS[str(k)], VarOLS[str(k)], MSElin[k], R2Slin[k] = LinearRegressionOLS(x,y,z,degree, True )

# Plot OLS bootstrap parameters
for j in range(5):
	xaxis = np.linspace(1,len(BetaOLS[str(j)]),len(BetaOLS[str(j)]))
	plt.figure()
	plt.plot(xaxis,BetaOLS[str(j)],linestyle=':',marker='o')
	plt.grid()
	title = r"OLS fit with resampling - weights $\beta_j$ for "+str(j+1)+". degree polynomial"
	plt.title(title)
	plt.savefig("figures/OLS/OLS_boot_parameters"+str(j+1))
	plt.xlabel(r"$j$th parameter")
	plt.ylabel(r"$\beta_j$")
	#plt.show()

# Plot MSE nd R2 score for OLS w/ bootstrap
plt.figure()
plt.subplot(211)
plt.title(r"OLS with resampling - $MSE$ and $R^2$ score")
degree = np.linspace(1,5,5)
plt.plot(degree, MSElin)
axs = plt.gca()
axs.set_ylabel(r"$MSE$")
plt.grid()
plt.subplot(212)
plt.plot(degree, R2Slin)
axs = plt.gca()
axs.set_ylabel(r"$R^2$ score")
axs.set_xlabel(r"poly degree")
plt.grid()
plt.savefig("figures/OLS/OLS_boot_err")
#plt.show()


# Plot fitting domain
#plt.figure()
#plt.plot(x,y,linestyle="none",marker='.',markersize=8)
#plt.show()


# Ridge regression
# reinitialize z data to study noise, redundant but easier
#noise = [0.001, 0.01, 0.1, 0.5]
#for n in range(len(noise)):
#z = FrankeFunction(x,y) + noise[n]*np.random.randn(size,1)

parameter = [ 1e-8, 1e-6, 1e-4, 1e-2]#np.logspace(-9,-4,6)
MSE = [0]*5
r2score = [0]*5
BetaRidge = {}
VarRidge = {}

# Plot MSE and R2 score for different lambdas
plt.figure()
for k in range(len(parameter)):
	BetaRidge[str(k)] = {}
	VarRidge[str(k)] = {}
	for j in range(5):
		degree = j+1
		MSE[j], r2score[j], BetaRidge[str(k)][str(j)], VarRidge[str(k)][str(j)] = Ridge(x,y,z,parameter[k], degree, False)
	# Plot ridge errors
	plt.subplot(211)
	plt.title(r"Ridge regression - $MSE$ and $R^2$ score. ")#Noise factor = "+str(noise[n]))
	degree = np.linspace(1,5,5)
	plt.plot(degree, MSE)
	axs = plt.gca()
	axs.set_ylabel(r"$MSE$")
	plt.subplot(212)
	plt.plot(degree, r2score)
	axs = plt.gca()
	axs.set_ylabel(r"$R^2$ score")
	axs.set_xlabel(r"poly degree")
plt.subplot(211)
plt.legend([r"$\lambda=$"+str(parameter[0]),r"$\lambda=$"+str(parameter[1]),r"$\lambda=$"+str(parameter[2]),r"$\lambda=$"+str(parameter[3])])	
plt.grid()
plt.subplot(212)
plt.grid()
plt.savefig("figures/ridge/ridge_err")#+"noise"+str(n+1))
#plt.show()

# Plot parameters for different lambdas
for j in range(5):
	xaxis = np.arange(1,len(BetaRidge["0"][str(j)].T)+1)
	plt.figure()
	for k in range(1,len(parameter)):
		plt.plot(xaxis,BetaRidge[str(k)][str(j)].T, linestyle=':',marker='.',linewidth=0.5)
	plt.plot(xaxis,BetaRidge["0"][str(j)].T,linestyle=':',marker='x')
	plt.grid()
	plt.legend([r"$\lambda=$"+str(parameter[0]),r"$\lambda=$"+str(parameter[1]),r"$\lambda=$"+str(parameter[2]),r"$\lambda=$"+str(parameter[3])])
	plt.title(r"Ridge parameters for different $\lambda$.") #Noise factor = "+str(noise[n]))
	plt.savefig("figures/ridge/ridge_parameters"+str(j+1)) #+"noise"+str(n+1))
	#plt.show()



# Ridge regression with resampling
# reinitialize z data to study noise, redundant but easier
#noise = [0.001, 0.01, 0.1, 0.5]
#for n in range(len(noise)):
#	z = FrankeFunction(x,y) + noise[n]*np.random.randn(size,1)

parameter = [ 1e-8, 1e-6, 1e-4, 1e-2]#np.logspace(-9,-4,6)
MSE = [0]*5
r2score = [0]*5
BetaRidge = {}
VarRidge = {}

# Plot MSE and R2 score for different lambdas
plt.figure()
for k in range(len(parameter)):
	BetaRidge[str(k)] = {}
	VarRidge[str(k)] = {}
	for j in range(5):
		degree = j+1
		MSE[j], r2score[j], BetaRidge[str(k)][str(j)], VarRidge[str(k)][str(j)] = Ridge(x,y,z,parameter[k], degree, True)
	# Plot ridge errors
	plt.subplot(211)
	plt.title(r"Ridge with resampling - $MSE$ and $R^2$ score.")#" Noise factor = "+str(noise[n]))
	degree = np.linspace(1,5,5)
	plt.plot(degree, MSE)
	axs = plt.gca()
	axs.set_ylabel(r"$MSE$")
	plt.subplot(212)
	plt.plot(degree, r2score)
	axs = plt.gca()
	axs.set_ylabel(r"$R^2$ score")
	axs.set_xlabel(r"poly degree")
plt.subplot(211)
plt.legend([r"$\lambda=$"+str(parameter[0]),r"$\lambda=$"+str(parameter[1]),r"$\lambda=$"+str(parameter[2]),r"$\lambda=$"+str(parameter[3])])	
plt.grid()
plt.subplot(212)
plt.grid()
plt.savefig("figures/ridge/ridge_boot_err")#+"noise"+str(n+1))
#plt.show()

# Plot parameters for different lambdas
for j in range(5):
	xaxis = np.arange(1,len(BetaRidge["0"][str(j)].T)+1)
	plt.figure()
	for k in range(1,len(parameter)):
		plt.plot(xaxis,BetaRidge[str(k)][str(j)].T, linestyle=':',marker='.',linewidth=0.5)
	plt.plot(xaxis,BetaRidge["0"][str(j)].T,linestyle=':',marker='x')
	plt.grid()
	plt.legend([r"$\lambda=$"+str(parameter[0]),r"$\lambda=$"+str(parameter[1]),r"$\lambda=$"+str(parameter[2]),r"$\lambda=$"+str(parameter[3])])
	plt.title(r"Ridge parameters for different $\lambda$.")#" Noise factor = "+str(noise[n]))
	plt.savefig("figures/ridge/ridge_boot_parameters"+str(j+1))#+"noise"+str(n+1))
	#plt.show()



# Lasso regression
# reinitialize z data to study noise, redundant but easier
#noise = [0.001, 0.01, 0.1, 0.5]
#for n in range(len(noise)):
#	z = FrankeFunction(x,y) + noise[n]*np.random.randn(size,1)

parameter = [ 1e-8, 1e-6, 1e-4, 1e-2]#np.logspace(-9,-4,6)
MSE = [0]*5
r2score = [0]*5
BetaLasso = {}
VarLasso = {}

# Plot MSE and R2 score for different lambdas
plt.figure()
for k in range(len(parameter)):
	BetaLasso[str(k)] = {}
	VarLasso[str(k)] = {}
	for j in range(5):
		degree = j+1
		MSE[j], r2score[j], BetaLasso[str(k)][str(j)], VarLasso[str(k)][str(j)] = Lasso(x,y,z,parameter[k], degree, False)
	# Plot ridge errors
	plt.subplot(211)
	plt.title(r"Lasso regression - $MSE$ and $R^2$ score.")#" Noise factor = "+str(noise[n]))
	degree = np.linspace(1,5,5)
	plt.plot(degree, MSE)
	axs = plt.gca()
	axs.set_ylabel(r"$MSE$")
	plt.subplot(212)
	plt.plot(degree, r2score)
	axs = plt.gca()
	axs.set_ylabel(r"$R^2$ score")
	axs.set_xlabel(r"poly degree")
plt.subplot(211)
plt.legend([r"$\lambda=$"+str(parameter[0]),r"$\lambda=$"+str(parameter[1]),r"$\lambda=$"+str(parameter[2]),r"$\lambda=$"+str(parameter[3])])	
plt.grid()
plt.subplot(212)
plt.grid()
plt.savefig("figures/lasso/lasso_err")#+"noise"+str(n+1))
#plt.show()
# Plot parameters for different lambdas
for j in range(5):
	xaxis = np.arange(1,len(BetaLasso["0"][str(j)].T)+1)
	plt.figure()
	for k in range(1,len(parameter)):
		plt.plot(xaxis,BetaLasso[str(k)][str(j)].T, linestyle=':',marker='.',linewidth=0.5)
	plt.plot(xaxis,BetaLasso["0"][str(j)].T,linestyle=':',marker='x')
	plt.grid()
	plt.legend([r"$\lambda=$"+str(parameter[0]),r"$\lambda=$"+str(parameter[1]),r"$\lambda=$"+str(parameter[2]),r"$\lambda=$"+str(parameter[3])])
	plt.title(r"Lasso parameters for different $\lambda$.")#" Noise factor = "+str(noise[n]))
	plt.savefig("figures/lasso/lasso_parameters"+str(j+1))#+"noise"+str(n+1))
	#plt.show()


# Lasso regression with resampling
# reinitialize z data to study noise, redundant but easier
#noise = [0.001, 0.01, 0.1, 0.5]
#for n in range(len(noise)):
#	z = FrankeFunction(x,y) + noise[n]*np.random.randn(size,1)

parameter = [0, 1e-8, 1e-6, 1e-4, 1e-2]#np.logspace(-9,-4,6)
MSE = [0]*5
r2score = [0]*5
BetaLasso = {}
VarLasso = {}

# Plot MSE and R2 score for different lambdas
plt.figure()
for k in range(len(parameter)):
	BetaLasso[str(k)] = {}
	VarLasso[str(k)] = {}
	for j in range(5):
		degree = j+1
		MSE[j], r2score[j], BetaLasso[str(k)][str(j)], VarLasso[str(k)][str(j)] = Lasso(x,y,z,parameter[k], degree, True)
	# Plot ridge errors
	plt.subplot(211)
	plt.title(r"Lasso with resampling - $MSE$ and $R^2$ score.")#" Noise factor = "+str(noise[n]))
	degree = np.linspace(1,5,5)
	plt.plot(degree, MSE)
	axs = plt.gca()
	axs.set_ylabel(r"$MSE$")
	plt.subplot(212)
	plt.plot(degree, r2score)
	axs = plt.gca()
	axs.set_ylabel(r"$R^2$ score")
	axs.set_xlabel(r"poly degree")
plt.subplot(211)
plt.legend([r"$\lambda=$"+str(parameter[0]),r"$\lambda=$"+str(parameter[1]),r"$\lambda=$"+str(parameter[2]),r"$\lambda=$"+str(parameter[3])])	
plt.grid()
plt.subplot(212)
plt.grid()
plt.savefig("figures/lasso/lasso_boot_err")#+"noise"+str(n+1))
#plt.show()

# Plot parameters for different lambdas
for j in range(5):
	xaxis = np.arange(1,len(BetaLasso["0"][str(j)].T)+1)
	plt.figure()
	for k in range(1,len(parameter)):
		plt.plot(xaxis,BetaLasso[str(k)][str(j)].T, linestyle=':',marker='.',linewidth=0.5)
	plt.plot(xaxis,BetaLasso["0"][str(j)].T,linestyle=':',marker='x')
	plt.grid()
	plt.legend([r"$\lambda=$"+str(parameter[0]),r"$\lambda=$"+str(parameter[1]),r"$\lambda=$"+str(parameter[2]),r"$\lambda=$"+str(parameter[3])])
	plt.title(r"Lasso parameters for different $\lambda$.")#" Noise factor = "+str(noise[n]))
	plt.savefig("figures/lasso/lasso_boot_parameters"+str(j+1))#+"noise"+str(n+1))
	#plt.show()
