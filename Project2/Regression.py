import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
import scipy as scl
import matplotlib.pyplot as plt 
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Random seed
np.random.seed(42)

# Function that creates a dimxdim lattice. It can be used as dim times a 1-D lattice of length dim
def Lattice(dim1, dim2):
    State = np.random.choice([-1, 1], size=(dim1, dim2))
    return State

#Function that computes the energy of a 1-D lattice
def Data_Ising_E1(State, Length, Conf):
    E = np.zeros((Conf,1))
    for j in range(Conf):
        for i in range(Length):
            E[j] = E[j] - State[i - 1, j]*State[i, j]
    return E

# OLS linear regression
def LinearRegression(x,y,resampling):
    if resampling==True:
        print("------ OLS with resampling ------")
        sample_steps = 100
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
        error = 0
        bias = 0
        variance = 0
        print("Errors of the model")
        print("MSE: ", MSE, "\n")
        print("R2 score: ", R2S, "\n")
    return Beta, VarBeta, MSE, R2S, error, bias, variance

def RidgeRegression(x, y, biasR, resampling):
    if resampling==True:
        print("------ RIDGE with resampling ------")
        sample_steps = 100
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
        error = 0
        bias = 0
        variance = 0
        print("Errors of the model")
        print("MSE: ", MSE, "\n")
        print("R2 score: ", R2S, "\n")
    return Beta, VarBeta, MSE, R2S, error, bias, variance

def LassoRegression(x, y, biasL, resampling):
    Lasso_reg = linear_model.Lasso(alpha=biasL, fit_intercept=False)
    if resampling==True:
        print("------ LASSO with resampling ------")
        sample_steps = 100
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
            tmp = x .dot(Beta_sample)
            ylinsample.T[i] = tmp
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
        error = 0
        bias = 0
        variance = 0
        print("Errors of the model")
        print("MSE: ", MSE, "\n")
        print("R2 score: ", R2S, "\n")
    return Beta, VarBeta, MSE, R2S, error, bias, variance
        
# I initialize the data
L = 40
conf = 200
Data = Lattice(L, conf)
E_Data = Data_Ising_E1(Data,L, conf)
print('Data')
print(Data)
print('E')
print(E_Data)
# Now we start the data analysis
# I set up the arrays for the regression
Data_Reg = np.zeros((conf,L*L))
for i in range(L):
    for j in range(L):
        for k in range(conf):
            Data_Reg[k,i*L+j] = Data[i,k]*Data[j,k]

# I set up the linear regression
BetaLin, VarBetaLin, MSELin, R2Lin, errorLin, biasLin, varLin  = LinearRegression(Data_Reg, E_Data, False)
BetaLinBoot, VarBetaLinBoot, MSELinBoot, R2LinBoot, errorLinBoot, biasLinBoot, varLinBoot  = LinearRegression(Data_Reg, E_Data, True)
BetaRid, VarBetaRid, MSERid, R2Rid, errorRid, biasRid, varRid  = RidgeRegression(Data_Reg, E_Data, 0.1, False)
BetaLas, VarBetaLas, MSELas, R2Las, errorLas, biasLas, varLas  = LassoRegression(Data_Reg, E_Data, 0.1, False)
BetaRidBoot, VarBetaRidBoot, MSERidBoot, R2RidBoot, errorRidBoot, biasRidBoot, varRidBoot  = RidgeRegression(Data_Reg, E_Data, 0.1, True)
BetaLasBoot, VarBetaLasBoot, MSELasBoot, R2LasBoot, errorLasBoot, biasLasBoot, varLasBoot  = LassoRegression(Data_Reg, E_Data, 0.1, True)

# Show search results
import seaborn as sns

sns.set()

BetaExpected = np.zeros([L, L])
BetaLinPlot = np.zeros([L, L])
BetaRidPlot = np.zeros([L, L])
BetaLasPlot = np.zeros([L, L])
BetaRidBootPlot = np.zeros([L, L])
BetaLasBootPlot = np.zeros([L, L])
for i in range(L):
    BetaExpected[i,i-1] = -1
    BetaExpected[i-1,i] = -1
    
BetaLinPlot = np.array(BetaLin).reshape((L,L))
BetaLinBootPlot = np.array(BetaLinBoot).reshape((L,L))
BetaRidPlot = np.array(BetaRid).reshape((L,L))
BetaLasPlot = np.array(BetaLas).reshape((L,L))
BetaRidBootPlot = np.array(BetaRidBoot).reshape((L,L))
BetaLasBootPlot = np.array(BetaLasBoot).reshape((L,L))

fig, ax = plt.subplots(figsize = (5, 5))
sns.heatmap(BetaExpected, annot=False, ax=ax, cmap="viridis")
ax.set_title("Expected J",fontsize=16)
ax.set_ylabel("$x_1$",fontsize=13)
ax.set_xlabel("$x_2$",fontsize=13)
plt.savefig("Reg_Expected_Beta.jpg")
plt.show()

fig, ax = plt.subplots(figsize = (5, 5))
sns.heatmap(BetaLinPlot, annot=False, ax=ax, cmap="viridis")
ax.set_title("J from linear regression",fontsize=16)
ax.set_ylabel("$x_1$",fontsize=13)
ax.set_xlabel("$x_2$",fontsize=13)
plt.savefig("Reg_Beta_Lin.jpg")
plt.show()

fig, ax = plt.subplots(figsize = (5, 5))
sns.heatmap(BetaLinBootPlot, annot=False, ax=ax, cmap="viridis")
ax.set_title("J from linear regression with bootstrap",fontsize=16)
ax.set_ylabel("$x_1$",fontsize=13)
ax.set_xlabel("$x_2$",fontsize=13)
plt.savefig("Reg_Beta_Boot_Lin.jpg")
plt.show()

fig, ax = plt.subplots(figsize = (5, 5))
sns.heatmap(BetaRidPlot, annot=False, ax=ax, cmap="viridis")
ax.set_title("J from Ridge regression",fontsize=16)
ax.set_ylabel("$x_1$",fontsize=13)
ax.set_xlabel("$x_2$",fontsize=13)
plt.savefig("Reg_Beta_Ridge1.jpg")
plt.show()

fig, ax = plt.subplots(figsize = (5, 5))
sns.heatmap(BetaLasPlot, annot=False, ax=ax, cmap="viridis")
ax.set_title("J from Lasso regression",fontsize=16)
ax.set_ylabel("$x_1$",fontsize=13)
ax.set_xlabel("$x_2$",fontsize=13)
plt.savefig("Reg_Beta_Lasso1.jpg")
plt.show()

fig, ax = plt.subplots(figsize = (5, 5))
sns.heatmap(BetaRidBootPlot, annot=False, ax=ax, cmap="viridis")
ax.set_title("J from Ridge regression with bootstrap",fontsize=16)
ax.set_ylabel("$x_1$",fontsize=13)
ax.set_xlabel("$x_2$",fontsize=13)
plt.savefig("Reg_Beta_Boot_Ridge1.jpg")
plt.show()

fig, ax = plt.subplots(figsize = (5, 5))
sns.heatmap(BetaLasBootPlot, annot=False, ax=ax, cmap="viridis")
ax.set_title("J from Lasso regression with bootstrap",fontsize=16)
ax.set_ylabel("$x_1$",fontsize=13)
ax.set_xlabel("$x_2$",fontsize=13)
plt.savefig("Reg_Beta_Bott_Lasso1.jpg")
plt.show()

BetaRid, VarBetaRid, MSERid, R2Rid, errorRid, biasRid, varRid  = RidgeRegression(Data_Reg, E_Data, 0.00001, False)
BetaLas, VarBetaLas, MSELas, R2Las, errorLas, biasLas, varLas  = LassoRegression(Data_Reg, E_Data, 0.00001, False)
BetaRidBoot, VarBetaRidBoot, MSERidBoot, R2RidBoot, errorRidBoot, biasRidBoot, varRidBoot  = RidgeRegression(Data_Reg, E_Data, 0.00001, True)
BetaLasBoot, VarBetaLasBoot, MSELasBoot, R2LasBoot, errorLasBoot, biasLasBoot, varLasBoot  = LassoRegression(Data_Reg, E_Data, 0.00001, True)

BetaRidPlot = np.array(BetaRid).reshape((L,L))
BetaLasPlot = np.array(BetaLas).reshape((L,L))
BetaRidBootPlot = np.array(BetaRidBoot).reshape((L,L))
BetaLasBootPlot = np.array(BetaLasBoot).reshape((L,L))

fig, ax = plt.subplots(figsize = (5, 5))
sns.heatmap(BetaRidPlot, annot=False, ax=ax, cmap="viridis")
ax.set_title("J from Ridge regression",fontsize=16)
ax.set_ylabel("$x_1$",fontsize=13)
ax.set_xlabel("$x_2$",fontsize=13)
plt.savefig("Reg_Beta_Ridge2.jpg")
plt.show()

fig, ax = plt.subplots(figsize = (5, 5))
sns.heatmap(BetaLasPlot, annot=False, ax=ax, cmap="viridis")
ax.set_title("J from Lasso regression",fontsize=16)
ax.set_ylabel("$x_1$",fontsize=13)
ax.set_xlabel("$x_2$",fontsize=13)
plt.savefig("Reg_Beta_Lasso2.jpg")
plt.show()

fig, ax = plt.subplots(figsize = (5, 5))
sns.heatmap(BetaRidBootPlot, annot=False, ax=ax, cmap="viridis")
ax.set_title("J from Ridge regression with bootstrap",fontsize=16)
ax.set_ylabel("$x_1$",fontsize=13)
ax.set_xlabel("$x_2$",fontsize=13)
plt.savefig("Reg_Beta_Boot_Ridge2.jpg")
plt.show()

fig, ax = plt.subplots(figsize = (5, 5))
sns.heatmap(BetaLasBootPlot, annot=False, ax=ax, cmap="viridis")
ax.set_title("J from Lasso regression with bootstrap",fontsize=16)
ax.set_ylabel("$x_1$",fontsize=13)
ax.set_xlabel("$x_2$",fontsize=13)
plt.savefig("Reg_Beta_Bott_Lasso2.jpg")
plt.show()

BetaRid, VarBetaRid, MSERid, R2Rid, errorRid, biasRid, varRid  = RidgeRegression(Data_Reg, E_Data, 0.00001, False)
BetaLas, VarBetaLas, MSELas, R2Las, errorLas, biasLas, varLas  = LassoRegression(Data_Reg, E_Data, 1000, False)
BetaRidBoot, VarBetaRidBoot, MSERidBoot, R2RidBoot, errorRidBoot, biasRidBoot, varRidBoot  = RidgeRegression(Data_Reg, E_Data, 1000, True)
BetaLasBoot, VarBetaLasBoot, MSELasBoot, R2LasBoot, errorLasBoot, biasLasBoot, varLasBoot  = LassoRegression(Data_Reg, E_Data, 1000, True)

BetaRidPlot = np.array(BetaRid).reshape((L,L))
BetaLasPlot = np.array(BetaLas).reshape((L,L))
BetaRidBootPlot = np.array(BetaRidBoot).reshape((L,L))
BetaLasBootPlot = np.array(BetaLasBoot).reshape((L,L))

fig, ax = plt.subplots(figsize = (5, 5))
sns.heatmap(BetaRidPlot, annot=False, ax=ax, cmap="viridis")
ax.set_title("J from Ridge regression",fontsize=16)
ax.set_ylabel("$x_1$",fontsize=13)
ax.set_xlabel("$x_2$",fontsize=13)
plt.savefig("Reg_Beta_Ridge3.jpg")
plt.show()

fig, ax = plt.subplots(figsize = (5, 5))
sns.heatmap(BetaLasPlot, annot=False, ax=ax, cmap="viridis")
ax.set_title("J from Lasso regression",fontsize=16)
ax.set_ylabel("$x_1$",fontsize=13)
ax.set_xlabel("$x_2$",fontsize=13)
plt.savefig("Reg_Beta_Lasso3.jpg")
plt.show()

fig, ax = plt.subplots(figsize = (5, 5))
sns.heatmap(BetaRidBootPlot, annot=False, ax=ax, cmap="viridis")
ax.set_title("J from Ridge regression with bootstrap",fontsize=16)
ax.set_ylabel("$x_1$",fontsize=13)
ax.set_xlabel("$x_2$",fontsize=13)
plt.savefig("Reg_Beta_Boot_Ridge3.jpg")
plt.show()

fig, ax = plt.subplots(figsize = (5, 5))
sns.heatmap(BetaLasBootPlot, annot=False, ax=ax, cmap="viridis")
ax.set_title("J from Lasso regression with bootstrap",fontsize=16)
ax.set_ylabel("$x_1$",fontsize=13)
ax.set_xlabel("$x_2$",fontsize=13)
plt.savefig("Reg_Beta_Bott_Lasso3.jpg")
plt.show()

# define regularisation parameter
lmbdas=np.logspace(-5,5,11)
lmbdas = lmbdas.T

BetaRidBoot = np.zeros([Data_Reg.shape[1],1])
BetaRidNoBoot = np.zeros([Data_Reg.shape[1],1])
BetaLasBoot = np.zeros([Data_Reg.shape[1],1])
BetaLasNoBoot = np.zeros([Data_Reg.shape[1],1])

VarBetaRidBoot = np.zeros([Data_Reg.shape[1],1])
VarBetaRidNoBoot = np.zeros([Data_Reg.shape[1],1])
VarBetaLasBoot = np.zeros([Data_Reg.shape[1],1])
VarBetaLasNoBoot = np.zeros([Data_Reg.shape[1],1])

MSERidBoot = np.zeros([lmbdas.shape[0],1])
MSERidNoBoot = np.zeros([lmbdas.shape[0],1])
MSELasBoot = np.zeros([lmbdas.shape[0],1])
MSELasNoBoot = np.zeros([lmbdas.shape[0],1])

R2SRidBoot = np.zeros([lmbdas.shape[0],1])
R2SRidNoBoot = np.zeros([lmbdas.shape[0],1])
R2SLasBoot = np.zeros([lmbdas.shape[0],1])
R2SLasNoBoot = np.zeros([lmbdas.shape[0],1])

errorRidBoot = np.zeros([lmbdas.shape[0],1])
errorRidNoBoot = np.zeros([lmbdas.shape[0],1])
errorLasBoot = np.zeros([lmbdas.shape[0],1])
errorLasNoBoot = np.zeros([lmbdas.shape[0],1])

biasRidBoot = np.zeros([lmbdas.shape[0],1])
biasRidNoBoot = np.zeros([lmbdas.shape[0],1])
biasLasBoot = np.zeros([lmbdas.shape[0],1])
biasLasNoBoot = np.zeros([lmbdas.shape[0],1])

varRidBoot = np.zeros([lmbdas.shape[0],1])
varRidNoBoot = np.zeros([lmbdas.shape[0],1])
varLasBoot = np.zeros([lmbdas.shape[0],1])
varLasNoBoot = np.zeros([lmbdas.shape[0],1])


# loop over regularisation strength
for i in range(lmbdas.shape[0]):
    BetaRidBoot, VarBetaRidBoot, MSERidBoot[i], R2SRidBoot[i], errorRidBoot[i], biasRidBoot[i], varRidBoot[i]  = RidgeRegression(Data_Reg, E_Data, lmbdas[i], True)
    BetaRidNoBoot, VarBetaRidNoBoot, MSERidNoBoot[i], R2SRidNoBoot[i], errorRidNoBoot[i], biasRidNoBoot[i], varRidNoBoot[i] = RidgeRegression(Data_Reg, E_Data, lmbdas[i], False)
    BetaLasBoot, VarBetaLasBoot, MSELasBoot[i], R2SLasBoot[i], errorLasBoot[i], biasLasBoot[i], varLasBoot[i] = LassoRegression(Data_Reg, E_Data, lmbdas[i], True)
    BetaLasNoBoot, VarBetaLasNoBoot, MSELasNoBoot[i], R2SLasNoBoot[i], errorLasNoBoot[i], biasLasNoBoot[i], varLasNoBoot[i] = LassoRegression(Data_Reg, E_Data, lmbdas[i], False)


# plot accuracy against regularisation strength
plt.semilogx(lmbdas,MSERidBoot,'*-b',label='MSE Ridge Bootstrap')
plt.semilogx(lmbdas,MSERidNoBoot,'*-r',label='MSE Ridge No Bootstrap')
plt.semilogx(lmbdas,MSELasBoot,'*-g',label='MSE Lasso Bootstrap')
plt.semilogx(lmbdas,MSELasNoBoot,'*-y',label='MSE Lasso No Bootstrap')

plt.xlabel('$\\lambda$')
plt.ylabel('$\\mathrm{MSE}$')

plt.grid()
plt.legend()
plt.savefig("Reg_MSE.jpg")
plt.show()

plt.semilogx(lmbdas,R2SRidBoot,'*-b',label='R2S Ridge Bootstrap')
plt.semilogx(lmbdas,R2SRidNoBoot,'*-r',label='R2S Ridge No Bootstrap')
plt.semilogx(lmbdas,R2SLasBoot,'*-g',label='R2S Lasso Bootstrap')
plt.semilogx(lmbdas,R2SLasNoBoot,'*-y',label='R2S Lasso No Bootstrap')

plt.xlabel('$\\lambda$')
plt.ylabel('$\\mathrm{R^2}$')

plt.grid()
plt.legend()
plt.savefig("Reg_R2S.jpg")
plt.show()

plt.semilogx(lmbdas,errorRidBoot,'*-b',label='Error Ridge')
plt.semilogx(lmbdas,biasRidBoot,'*-r',label='Bias Ridge')
plt.semilogx(lmbdas,varRidBoot,'*-g',label='Variance Ridge')
plt.semilogx(lmbdas,errorLasBoot,'*-y',label='Error Lasso')
plt.semilogx(lmbdas,biasLasBoot,'*-c',label='Bias Lasso')
plt.semilogx(lmbdas,varLasNoBoot,'*-m',label='Variance Lasso')


plt.xlabel('$\\lambda$')
plt.ylabel('$Error$')

plt.grid()
plt.legend()
plt.savefig("Reg_Errors.jpg")
plt.show()


