import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import linear_model
import random
import numpy as np
import scipy as scl

# Define functions

# Franke Function
def FrankeFunction(x,y): 
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1)) 
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2)) 
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4

# OLS linear regression
def LinearRegressionOLS(x,y,z,degree):
    poly = PolynomialFeatures(degree=degree)
    data = poly.fit_transform(np.concatenate((x, y), axis=1))
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
    print(VarBeta)
    R2S = r2_score(z,zlin)
    print("-----Polynomial degree = ", degree,"-----")
    print("MSE %g" % MSE)
    print("RS2 %g" % R2S)
    return Beta, VarBeta, MSE, R2S


# Least squares regression    
def LSregression(x, y, z, degree, resampling):
    poly = PolynomialFeatures(degree=degree)
    data = poly.fit_transform(np.concatenate((x, y), axis=1))
    if resampling == True:
        k=1000
        MSE = 0
        R2score = 0
        for i in range(k):
            data = random.choices(pool, pool.shape[1])
            H = data.T .dot(data)
            Beta = scl.linalg.inv(H) .dot(data.T) .dot(z)
            z_fit = data .dot(Beta)
            MSE += mean_squared_error(z,z_fit)
            VarBeta = np.diag(H .dot(MSE * np.eye(H.shape[1])))
            R2score += r2_score(z,z_fit)
        MSE = MSE/k
        R2score = R2score/k
    else:
        H = data.T .dot(data)
        Beta = scl.linalg.inv(H) .dot(data.T) .dot(z)
        z_fit = data .dot(Beta)
        MSE = mean_squared_error(z,z_fit)
        VarBeta = np.diag(H .dot(MSE * np.eye(H.shape[1])))
        R2score = r2_score(z,z_fit)
    print("-------",degree,"th degree polynomial","-------")
    print(" MSE: ", MSE)
    print(" R2 score: ", R2score, "\n")
    return MSE, R2score, np.transpose(Beta), VarBeta


# Ridge regression    
def Ridge(x, y, z, biasR, degree, resampling):
    poly = PolynomialFeatures(degree=degree)
    data = poly.fit_transform(np.concatenate((x, y), axis=1))
    if resampling == True:
        k=1000
        MSE = 0
        R2score = 0
        for i in range(k):
            data = random.choices(pool, pool.shape[1])
            H = data.T .dot(data)
            Beta = scl.linalg.inv(H + biasR*H.shape[1]) .dot(data.T) .dot(z) 
            z_fit = data .dot(Beta)
            MSE += mean_squared_error(z,z_fit)
            VarBeta = MSE * np.diag((H + biasR*H.shape[1]) .dot(H) .dot((H + biasR*H.shape[1]).T))
            R2score += r2_score(z,z_fit)
        MSE = MSE/k
        R2score = R2score/k
    else:
        H = data.T .dot(data)
        Beta = scl.linalg.inv(H + biasR*H.shape[1]) .dot(data.T) .dot(z) 
        z_fit = data .dot(Beta)
        MSE = mean_squared_error(z,z_fit)
        VarBeta = MSE * np.diag((H + biasR*H.shape[1]) .dot(H) .dot((H + biasR*H.shape[1]).T))
        R2score = r2_score(z,z_fit)
    print("-------",degree,"th degree polynomial","-------")
    print(" MSE: ", MSE)
    print(" R2 score: ", R2score, "\n")
    return MSE, R2score, np.transpose(Beta), VarBeta

def Lasso(x, y, z, biasL, degree, resampling):
    poly = PolynomialFeatures(degree=degree)
    data = poly.fit_transform(np.concatenate((x, y), axis=1))
    lasso_reg = linear_model.Lasso(alpha=biasL, fit_intercept=False)
    if resampling == True:
        k=1000
        MSE = 0
        R2score = 0
        for i in range(k):
            data = random.choices(pool, pool.shape[1])
            lasso_reg.fit(data,z)
            H = data.T .dot(data)
            Blasso = lasso_reg.coef_
            Beta = np.zeros((len(Blasso),1))
            VarBeta = np.zeros((len(Blasso),1))
            for i in range(len(Blasso)):
                Beta[i] = Blasso[i]
            z_fit = data .dot(Beta)
            MSE_tmp = mean_squared_error(z,z_fit)
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
            MSE += MSE_tmp
            R2score += r2_score(z,z_fit)
        MSE = MSE/k
        R2score = R2score/k
    else:
        lasso_reg.fit(data,z)
        H = data.T .dot(data)
        Blasso = lasso_reg.coef_
        Beta = np.zeros((len(Blasso),1))
        VarBeta = np.zeros((len(Blasso),1))
        for i in range(len(Blasso)):
            Beta[i] = Blasso[i]
        z_fit = data .dot(Beta)
        MSE = mean_squared_error(z,z_fit)
        R2score = r2_score(z,z_fit)
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
    print("-------",degree,"th degree polynomial","-------")
    print(" MSE: ", MSE)
    print(" R2 score: ", R2score, "\n")
    return MSE, R2score, np.transpose(Beta), VarBeta
# K-fold Function
#def bootstrap():
#    x = np.random.shuffle(x)


def k_fold(x, y, z, Pol_deg, method, biasLambda, k):
    n = len(x) 
    poly = PolynomialFeatures(degree=degree)
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
            MeanSquare_train, r2score_train, Beta_train, VarBeta_train = LSregression(x_train[:,[j]], y_train[:,[j]], z_train[:,[j]], Pol_deg,"False")
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


# Create Data set from Franke function
size = 100
x = np.random.rand(size,1)
y = np.random.rand(size,1)
z = FrankeFunction(x,y) + 0.5*np.random.randn(size,1)


# Start doing regression with OLS up to 5th degree
print("OLS REGRESSION RESULTS")
fitLS = {}
MeanSquareErrorsLS = [0]*5
r2scoresLS = [0]*5
for k in range(5):
    degree=k+1
    name = "poly" + str(k+1)
    fitLS[name] = {}
    MeanSquareErrorsLS[k], r2scoresLS[k], fitLS[name]["Beta"], fitLS[name]["VarBeta"] = Lasso(x, y, z, 0.01, degree,"True")

fitLS_boot = {}
MeanSquareErrorsLS_boot = [0]*5
r2scoresLS_boot = [0]*5
for k in range(5):
    degree=k+1
    name = "poly" + str(k+1)
    fitLS_boot[name] = {}
    MeanSquareErrorsLS_boot[k], r2scoresLS_boot[k], fitLS_boot[name]["Beta"], fitLS_boot[name]["VarBeta"] = k_fold(x, y, z, degree, "OLS", 0.01, 10)

xaxis = np.linspace(1,5,5)
plt.subplot(211)
plt.plot(xaxis,MeanSquareErrorsLS)
plt.subplot(212)
plt.plot(xaxis,MeanSquareErrorsLS_boot)
plt.show()

"""
plt.subplot(211)
plt.plot(degree, MeanSquareErrors,linestyle='--',linewidth=0.5,marker='o', color='r')
plt.grid(True, which="both", linestyle='--')
plt.title(r"OLS - Mean square error and $R^2$ score")
plt.ylabel(r"$MSE$")
plt.subplot(212)
plt.plot(degree, r2scores,linestyle='--',linewidth=0.5,marker='o', color='b')
plt.grid(True, which="both", linestyle='--')
plt.xlabel("polynomial degree")
plt.ylabel(r"$R^2$")
plt.show()


# Use shrinkage methods (k-fold) on data set
fitLS_kfold = {}
MeanSquareErrorsLS_kfold = [0]*5
r2scoresLS_kfold = [0]*5
for k in range(5):
    degree = k+1
    name = "poly" + str(k+1)
    fitLS_kfold[name]={}
    MeanSquareErrorsLS_kfold[k], r2scoresLS_kfold[k], fitLS_kfold[name]["Beta"], fitLS_kfold[name]["VarBeta"] = k_fold(x,y,z,degree,"OLS",0,5)
print(MeanSquareErrorsLS)
print(xaxis)
plt.subplot(211)
plt.plot(xaxis, MeanSquareErrorsLS)
plt.subplot(212)
plt.plot(xaxis, MeanSquareErrorsLS_kfold)
plt.show()
    



# Regression using ridge method
cycles = 10
bias = np.linspace(0,1,cycles)
print("RIDGE REGRESSION RESULTS")
fit_Ridge = {}
parameters_Ridge = {}
for i in range(cycles):
    print("\lambda = ", bias[i])
    for k in range(5):
        degree=k+1
        name = "poly" + str(k+1)
        fit_Ridge[name], parameters[name] = Ridge(x, y, bias[i],degree)

# Using shrinkage on ridge regression
print("RIDGE REGRESSION RESULTS")
fit_Ridge = {}
parameters_Ridge = {}
for i in range(cycles):
    print("\lambda = ", bias[i])
    for k in range(5):
        degree=k+1
        name = "poly" + str(k+1)
        fit_Ridge[name] = k_fold(x,y,z,degree,"ridge",bias[i],5)
"""