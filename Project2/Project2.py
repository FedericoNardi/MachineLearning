import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
import scipy as scl

# Random seed
np.random.seed(42)

# Function that creates a dimxdim lattice. It can be used as dim times a 1-D lattice of length dim
def Lattice(dim):
    State = np.random.choice([-1, 1], size=(dim,dim))
    return State

#Function that computes the energy of a 1-D lattice
def Data_Ising_E1(State, Length, J):
    E = np.zeros((Length,1))
    for j in range(Length):
        for i in range(Length):
            E[j] = E[j] + State[i - 1, j]*State[i, j]
    return E

# OLS linear regression
def LinearRegression(x,y,resampling):
    if resampling==True:
        print("------ OLS with resampling ------")
        sample_steps = 1000
        VarBeta = np.zeros([x.shape[1],1])
        Beta_boot = np.zeros([x.shape[1],sample_steps])
        Beta = np.zeros([x.shape[1],1])
        pool = np.zeros(x.shape)
        y_sample = np.zeros([x.shape[0],1])   
        ylinsample = np.zeros([y.shape[0],sample_steps])         
        for i in range(sample_steps):
            for k in range(x.shape[0]):
                index = np.random.randint(x.shape[0])
                pool[k,:] = x[index,:]
                y_sample[k] = y[index]
            U, D, Vt = scl.linalg.svd(pool)
            V = Vt.T
            Sigma = np.zeros([V.shape[1],U.shape[1]])
            np.fill_diagonal(Sigma,D**(-1))
            # Parameters of the regression
            Beta_sample = V .dot(Sigma) .dot(U.T) .dot(y_sample)
            tmp = x .dot(Beta_sample)
            ylinsample.T[i] = tmp[:,0]
            Beta_boot.T[i] = Beta_sample.T
        for i in range(x.shape[1]):
            Beta[i] = np.mean(Beta_boot[i])
            VarBeta[i] = np.var(Beta_boot[i])
        ylin = x .dot(Beta)
        MSE = mean_squared_error(y,ylin)
        R2S = r2_score(y,ylin)
        error_tmp = np.zeros([y.shape[0],1])
        ymean = np.zeros([y.shape[0],1])
        variance_tmp = np.zeros([y.shape[0],1])
        for i in range(sample_steps):
            error_tmp = error_tmp + (y - ylinsample[:,i])**2
        error_tmp = error_tmp/sample_steps
        error = np.sum(error_tmp)/y.shape[0]
        for i in range(sample_steps):
            ymean = ymean + ylinsample[:,i]
        ymean = ymean/sample_steps
        bias = np.sum((y - ymean)**2)/y.shape[0]
        for i in range(sample_steps):
            variance_tmp = variance_tmp + (ylinsample[:,i] - ymean)**2
        variance_tmp = variance_tmp/sample_steps
        variance = np.sum(variance_tmp)/y.shape[0]
        print("Beta:")
        print(Beta)
        print("Beta variance:")
        print(VarBeta)
        print("Model:")
        print(ylin)
        print("Errors of the model")
        print("MSE: ", MSE, "\n")
        print("R2 score: ", R2S, "\n")
        print("Error", error, "\n")
        print("Bias", bias, "\n")
        print("Variance", variance, "\n")
    else:
        print("------ OLS without resampling ------")
        # Single Value decomposition to be always able of making inverse
        U, D, Vt = scl.linalg.svd(x)
        V = Vt.T
        Sigma = np.zeros([V.shape[1],U.shape[1]])
        SigmaVar = np.zeros([V.shape[0],V.shape[1]])
        np.fill_diagonal(Sigma,D**(-1))
        np.fill_diagonal(SigmaVar,D**(-2))
        # Parameters of the regression
        Beta = V .dot(Sigma) .dot(U.T) .dot(y)
        # I calculate the error of the model
        # First I calculate the output of the model
        ylin = x .dot(Beta)
        # Error on the parameters
        MSE = mean_squared_error(y,ylin)
        VarBeta = MSE*(np.diag(V .dot(SigmaVar) .dot(V.T))[np.newaxis]).T
        R2S = r2_score(y,ylin)
        print("Beta:")
        print(Beta)
        print("Beta variance:")
        print(VarBeta)
        print("Model:")
        print(ylin)
        print("Errors of the model")
        print("MSE: ", MSE, "\n")
        print("R2 score: ", R2S, "\n")
    return Beta, VarBeta

def RidgeRegression(x, y, biasR, resampling):
    if resampling==True:
        print("------ RIDGE with resampling ------")
        sample_steps = 1000
        VarBeta = np.zeros([x.shape[1],1])
        Beta_boot = np.zeros([x.shape[1],sample_steps])
        Beta = np.zeros([x.shape[1],1])
        pool = np.zeros(x.shape)
        y_sample = np.zeros([x.shape[0],1])
        ylinsample = np.zeros([y.shape[0],sample_steps])
        for i in range(sample_steps):
            for k in range(x.shape[0]):
                index = np.random.randint(x.shape[0])
                pool[k,:] = x[index,:]
                y_sample[k] = y[index]
            H = x.T .dot(x)
            Beta_sample = scl.linalg.inv(H + biasR*np.eye(H.shape[0])) .dot(x.T) .dot(y_sample)
            tmp = x .dot(Beta_sample)
            ylinsample.T[i] = tmp[:,0]
            Beta_boot.T[i] = Beta_sample.T
        for i in range(x.shape[1]):
            Beta[i] = np.mean(Beta_boot[i]) 
            VarBeta[i] = np.var(Beta_boot[i])
        ylin = x .dot(Beta)
        MSE = mean_squared_error(y,ylin)
        R2S = r2_score(y,ylin)
        error_tmp = np.zeros([y.shape[0],1])
        ymean = np.zeros([y.shape[0],1])
        variance_tmp = np.zeros([y.shape[0],1])
        for i in range(sample_steps):
            error_tmp = error_tmp + (y - ylinsample[:,i])**2
        error_tmp = error_tmp/sample_steps
        error = np.sum(error_tmp)/y.shape[0]
        for i in range(sample_steps):
            ymean = ymean + ylinsample[:,i]
        ymean = ymean/sample_steps
        bias = np.sum((y - ymean)**2)/y.shape[0]
        for i in range(sample_steps):
            variance_tmp = variance_tmp + (ylinsample[:,i] - ymean)**2
        variance_tmp = variance_tmp/sample_steps
        variance = np.sum(variance_tmp)/y.shape[0]
        print("Beta:")
        print(Beta)
        print("Beta variance:")
        print(VarBeta)
        print("Model:")
        print(ylin)
        print("Errors of the model")
        print("MSE: ", MSE, "\n")
        print("R2 score: ", R2S, "\n")
        print("Error", error, "\n")
        print("Bias", bias, "\n")
        print("Variance", variance, "\n")
    else:
        print("------ RIDGE without resampling ------")
        H = x.T .dot(x)
        Beta = scl.linalg.inv(H + biasR*np.eye(H.shape[1])) .dot(x.T) .dot(y) 
        ylin = x .dot(Beta)
        MSE = mean_squared_error(y,ylin)
        R2S = r2_score(y,ylin)
        VarBeta = MSE * np.diag((H + biasR*H.shape[1]) .dot(H) .dot((H + biasR*H.shape[1]).T))
        print("Beta:")
        print(Beta)
        print("Beta variance:")
        print(VarBeta)
        print("Model:")
        print(ylin)
        print("Errors of the model")
        print("MSE: ", MSE, "\n")
        print("R2 score: ", R2S, "\n")
    return Beta, VarBeta

def LassoRegression(x, y, biasL, resampling):
    Lasso_reg = linear_model.Lasso(alpha=biasL, fit_intercept=False)
    if resampling==True:
        print("------ LASSO with resampling ------")
        sample_steps = 1000
        VarBeta = np.zeros([x.shape[1],1])
        Beta_boot = np.zeros([x.shape[1],sample_steps])
        Beta = np.zeros([x.shape[1],1])
        pool = np.zeros(x.shape)
        y_sample = np.zeros([x.shape[0],1])
        ylinsample = np.zeros([y.shape[0],sample_steps])
        for i in range(sample_steps):
            for k in range(x.shape[0]):
                index = np.random.randint(x.shape[0])
                pool[k,:] = x[index,:]
                y_sample[k] = y[index]
            Lasso_reg.fit(pool,y_sample)
            Beta_sample = Lasso_reg.coef_
            Beta_boot.T[i] = Beta_sample.T
        for i in range(x.shape[1]):
            Beta[i] = np.mean(Beta_boot[i]) 
            VarBeta[i] = np.var(Beta_boot[i])
            tmp = x .dot(Beta_sample)
            ylinsample.T[i] = tmp
        ylin = x .dot(Beta)
        MSE = mean_squared_error(y,ylin)
        R2S = r2_score(y,ylin)
        error_tmp = np.zeros([y.shape[0],1])
        ymean = np.zeros([y.shape[0],1])
        variance_tmp = np.zeros([y.shape[0],1])
        for i in range(sample_steps):
            error_tmp = error_tmp + (y - ylinsample[:,i])**2
        error_tmp = error_tmp/sample_steps
        error = np.sum(error_tmp)/y.shape[0]
        for i in range(sample_steps):
            ymean = ymean + ylinsample[:,i]
        ymean = ymean/sample_steps
        bias = np.sum((y - ymean)**2)/y.shape[0]
        for i in range(sample_steps):
            variance_tmp = variance_tmp + (ylinsample[:,i] - ymean)**2
        variance_tmp = variance_tmp/sample_steps
        variance = np.sum(variance_tmp)/y.shape[0]
        print("Beta:")
        print(Beta)
        print("Beta variance:")
        print(VarBeta)
        print("Model:")
        print(ylin)
        print("Errors of the model")
        print("MSE: ", MSE, "\n")
        print("R2 score: ", R2S, "\n")
        print("Error", error, "\n")
        print("Bias", bias, "\n")
        print("Variance", variance, "\n")
    else:
        print("------ LASSO without resampling ------")
        Lasso_reg.fit(x,y)
        H = x.T .dot(x)
        BLasso = Lasso_reg.coef_
        Beta = np.zeros((len(BLasso),1))
        VarBeta = np.zeros((len(BLasso),1))
        for i in range(len(BLasso)):
            Beta[i] = BLasso[i]
        ylin = x .dot(Beta)
        MSE = mean_squared_error(y,ylin)
        tmp = np.eye(len(H))
        check = 0
        VarBeta = np.zeros((len(BLasso),1))
        for j in range(len(Beta)):
            if Beta[j] != 0:
                tmp[j,j] = 1/np.abs(Beta[j])
            else:
                tmp[j,j] = 0
                check = 1
        if check == 0:
            VarBeta = MSE*np.diag(scl.linalg.inv(H + biasL*tmp) .dot(H) .dot(scl.linalg.inv(H + biasL*tmp)))
        R2S = r2_score(y,ylin)
        print("Beta:")
        print(Beta)
        print("Beta variance:")
        print(VarBeta)
        print("Model:")
        print(ylin)
        print("Errors of the model")
        print("MSE: ", MSE, "\n")
        print("R2 score: ", R2S, "\n")
    return Beta, VarBeta
        
# I initialize the data
L = 2;
Data = Lattice(L)
E_Data = Data_Ising_E1(Data,L,1)
print("The data are :")
print(Data)
print("The energies are :")
print(E_Data)

# Now we start the data analysis
# I set up the arrays for the regression
Data_Reg = np.zeros((L,L*L))
for i in range(L):
    for j in range(L):
        for k in range(L):
            Data_Reg[k,i*L+j] = Data[i,k]*Data[j,k]
print("The data for the regression are:")
print(Data_Reg)

# I set up the linear regression
BetaLin, VarBetaLin  = LinearRegression(Data_Reg, E_Data, False)
BetaRidBoot, VarBetaRidBoot = RidgeRegression(Data_Reg, E_Data, 0.01, True)
BetaRidNoBoot, VarBetaRidNoBoot = RidgeRegression(Data_Reg, E_Data, 0.01, False)
BetaLasBoot, VarBetaLasBoot = LassoRegression(Data_Reg, E_Data, 0.01, True)
BetaLasNoBoot, VarBetaLasNoBoot = LassoRegression(Data_Reg, E_Data, 0.01, False)

