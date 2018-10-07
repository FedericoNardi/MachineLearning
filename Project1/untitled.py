

" Code for the project 1 of the course Data analysis and Machine learning."
" I import some packages"
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import PolynomialFeatures
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import linear_model
from sklearn.linear_model import LinearRegression as SKregression
import numpy as np
import scipy as scl
from statistics import mean


# I create the functions

# Franke Function
def FrankeFunction(x,y): 
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1)) 
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2)) 
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4

9# OLS linear regression
def LinearRegressionOLS(x,y,z,degree,resampling):
    poly = PolynomialFeatures(degree=degree)
    data = poly.fit_transform(np.concatenate((x, y), axis=1))
    if resampling==True:
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
    return Beta, VarBeta, MSE, R2S



    # poly = PolynomialFeatures(degree=degree)
    # DataSet = poly.fit_transform(np.concatenate((x,y), axis=1))
    # if resampling == "True":
    #     MeanSquareError = 0
    #     rSquareScore = 0
    #     sample_steps = 10
    #     VarBeta = np.zeros([DataSet.shape[1],1])
    #     Beta_boot = np.zeros([DataSet.shape[1],sample_steps])
    #     Beta = np.zeros([DataSet.shape[1],1])
    #     pool = np.zeros(DataSet.shape)
    #     z_sample = np.zeros([DataSet.shape[0],1])
    #     for i in range(sample_steps):
    #         for k in range(DataSet.shape[0]):
    #             index = np.random.randint(DataSet.shape[0])
    #             pool[k] = DataSet[index]
    #             z_sample[k] = z[index]  
    #         H = (pool.T).dot(pool)
    #         Beta_sample = scl.linalg.inv(H).dot(pool.T).dot(z_sample)
    #         Beta_boot.T[i] = Beta_sample.T
    #     for i in range(DataSet.shape[1]):
    #         Beta[i] = np.mean(Beta_boot[i]) 
    #         VarBeta[i] = np.var(Beta_boot[i])
    #         print(Beta_boot[1:4])
    #         foo 
    #     z_fit = DataSet.dot(Beta)
    #     MeanSquareError_sample = MSE(z_fit, model)  
    #     rSquareScore = r2score(z_fit, model)

# Check OLS with SKlearn
def SKLcheckLS(x,y,z,degree):
    poly = PolynomialFeatures(degree=degree)
    data = poly.fit_transform(np.concatenate((x, y), axis=1))
    linreg = SKregression(fit_intercept=False)
    linreg.fit(data,z)
    return linreg

# K-fold Function
def k_fold(x, y, z, Pol_deg, spec_regre, k):
    n = len(x) 
    len_fold = n//k
    MSE = np.zeros((k,1))
    R2S = np.zeros((k,1))
    MSE_sampling = 0
    R2S_sampling = 0
    for j in range(k):
        x_train= np.zeros((80,1))
        y_train = np.zeros((80,1))
        z_train = np.zeros((80,1))
        x_test = np.zeros((20,1))
        y_test = np.zeros((20,1))
        z_test = np.zeros((20,1))
        l = 0
        # I create the training and test data
        for i in range(len_fold):
            x_test[i] = x[j*len_fold + i]
            y_test[i] = y[j*len_fold + i]
            z_test[i] = z[j*len_fold + i]
        for i in range(n):
            if (i < j*len_fold) or (i > (j + 1)*len_fold):
                x_train[l] = x[i]
                y_train[l] = y[i]
                z_train[l] = z[i]
                l = l + 1
        # I set up the regression
        poly = PolynomialFeatures(degree= Pol_deg)
        data_train = poly.fit_transform(np.concatenate((x_train, y_train), axis=1))
        data_test = poly.fit_transform(np.concatenate((x_test, y_test), axis=1))
        spec_regre.fit(data_train, z_train)
        # Now I use the parameters I obtained to make predictions and I calculate the MSE and R2score
        MSE[j] = mean_squared_error(spec_regre.predict(data_test),z_test)
        R2S[j] = r2_score(spec_regre.predict(data_test),z_test)
        MSE_sampling = MSE_sampling + MSE[j]
        R2S_sampling = R2S_sampling + R2S[j]
    # Now I take the mean value
    MSE_sampling = MSE_sampling/k
    R2S_sampling = R2S_sampling/k
    return MSE_sampling, R2S_sampling



# Initialize random seed    
#np.random.seed(42) #Life, Universe and Everything

# Generate data
size = 100
x = np.array([[i] for i in range(size)]*size)
x=x/size
y = []
for i in range(size):
    y += [i]*size
y = (np.array(y)[np.newaxis]).T
y=y/size

#y = np.random.rand(size,1)
z = FrankeFunction(x,y) + 0.1*np.random.randn(size**2,1)


xplot = np.arange(0, 1, 0.01)
yplot = np.arange(0, 1, 0.01)

# I start doing a linear regression
# Linear regression with degree 1
print("Results of the linear regression")



# Fith with OLS
degree = 5

Blin1, VarBlin1, MSElin1, R2Slin1 = LinearRegressionOLS(x,y,z,degree, True )

# Fit with SKL to check
linreg1 = SKLcheckLS(x,y,z,degree)

# Print a log
Blin1_check = linreg1.coef_
print("------- Polynomial degree = ",degree, "-------")
print("My fit parameters")
print(Blin1)
print(VarBlin1)
print("SKlearn fit parameters")
print(Blin1_check.T)

foo 

# Let's k-fold with the linear.
# Polyonomial degree = 1
MSElin1Fold, R2Slin1Fold = k_fold(x, y, z, 1, linreg1, 5)
print("Polynomial degree = 1")
print("K-fold MSE")
print(MSElin1Fold)
print("K-fold R2S")
print(R2Slin1Fold)
# Polyonomial degree = 2
MSElin2Fold, R2Slin2Fold = k_fold(x, y, z, 2, linreg2, 5)
print("Polynomial degree = 2")
print("K-fold MSE")
print(MSElin2Fold)
print("K-fold R2S")
print(R2Slin2Fold)
# Polyonomial degree = 3
MSElin3Fold, R2Slin3Fold = k_fold(x, y, z, 3, linreg3, 5)
print("Polynomial degree = 3")
print("K-fold MSE")
print(MSElin3Fold)
print("K-fold R2S")
print(R2Slin3Fold)
# Polyonomial degree = 4
MSElin4Fold, R2Slin4Fold = k_fold(x, y, z, 4, linreg4, 5)
print("Polynomial degree = 4")
print("K-fold MSE")
print(MSElin4Fold)
print("K-fold R2S")
print(R2Slin4Fold)
# Polyonomial degree = 5
MSElin5Fold, R2Slin5Fold = k_fold(x, y, z, 5, linreg5, 5)
print("Polynomial degree = 5")
print("K-fold MSE")
print(MSElin5Fold)
print("K-fold R2S")
print(R2Slin5Fold)


# Ridge regression with lambda = 0.1
lamda = 0.1
print("Results of the Ridge regression")
# Ridge regression with degree 1
Brid1 = scl.linalg.inv(H1 + lamda *np.eye(3)) .dot(data_Lin1.T) .dot(z)
zrid1 = data_Lin1 .dot(Brid1)
VarZrid1 = mean_squared_error(z, zrid1)
VarBrid2 = H1 .dot(VarZrid1 * np.eye(3))
MSErid1 = mean_squared_error(z,zrid1)
R2Srid1 = r2_score(z,zrid1)
print("Polynomial degree = 1")
print("MSE %g" % MSErid1)
print("RS2 %g" % R2Srid1)
# Ridge regression with degree 2
Brid2 = scl.linalg.inv(H2 + lamda *np.eye(6)) .dot(data_Lin2.T) .dot(z)
zrid2 = data_Lin2 .dot(Brid2)
VarZrid2 = mean_squared_error(z, zrid2)
VarBrid2 = H2 .dot(VarZrid2 * np.eye(6))
MSErid2 = mean_squared_error(z,zrid2)
R2Srid2 = r2_score(z,zrid2)
print("Polynomial degree = 2")
print("MSE %g" % MSErid2)
print("RS2 %g" % R2Srid2)             
# Ridge regression with degree 3
Brid3 = scl.linalg.inv(H3 + lamda *np.eye(10)) .dot(data_Lin3.T) .dot(z)
zrid3 = data_Lin3 .dot(Brid3)
VarZrid3 = mean_squared_error(z, zrid3)
VarBrid3 = H3 .dot(VarZrid3 * np.eye(10))
MSErid3 = mean_squared_error(z,zrid3)
R2Srid3 = r2_score(z,zrid3)
print("Polynomial degree = 3")
print("MSE %g" % MSErid3)
print("RS2 %g" % R2Srid3)                   
# Ridge regression with degree 4
Brid4 = scl.linalg.inv(H4 + lamda *np.eye(15)) .dot(data_Lin4.T) .dot(z)
zrid4 = data_Lin4 .dot(Brid4)
VarZrid4 = mean_squared_error(z, zrid4)
VarBrid4 = H4 .dot(VarZrid4 * np.eye(15))
MSErid4 = mean_squared_error(z,zrid1)
R2Srid4 = r2_score(z,zrid1)
print("Polynomial degree = 4")
print("MSE %g" % MSErid4)
print("RS2 %g" % R2Srid4)  
# Ridge regression with degree 5
Brid5 = scl.linalg.inv(H5 + lamda *np.eye(21)) .dot(data_Lin5.T) .dot(z)
zrid5 = data_Lin5 .dot(Brid5)
VarZrid5 = mean_squared_error(z, zrid5)
VarBrid5 = H5 .dot(VarZrid5 * np.eye(21))  
MSErid5 = mean_squared_error(z,zrid5)
R2Srid5 = r2_score(z,zrid5)
print("Polynomial degree = 5")
print("MSE %g" % MSErid5)
print("RS2 %g" % R2Srid5)

# Check with Sklearn of the Ridge regression
ridge1 = linear_model.RidgeCV(alphas=[0.1], fit_intercept=False)
ridge2 = linear_model.RidgeCV(alphas=[0.1], fit_intercept=False)
ridge3 = linear_model.RidgeCV(alphas=[0.1], fit_intercept=False)
ridge4 = linear_model.RidgeCV(alphas=[0.1], fit_intercept=False)
ridge5 = linear_model.RidgeCV(alphas=[0.1], fit_intercept=False)
print("Check the results with Sklearn")
# Check plynomial degree = 1
ridge1.fit(data_Lin1,z)
Bridge1_check = ridge1.coef_
print("Polynomial degree = 1")
print("My fit parameters")
print(Brid1)
print("Skelearn fit parameters")
print(Bridge1_check.T)
# Check plynomial degree = 2
ridge2.fit(data_Lin2,z)
Bridge2_check = ridge2.coef_
print("Polynomial degree = 2")
print("My fit parameters")
print(Brid2)
print("Skelearn fit parameters")
print(Bridge2_check.T)
# Check plynomial degree = 3
ridge3.fit(data_Lin3,z)
Bridge3_check = ridge3.coef_
print("Polynomial degree = 3")
print("My fit parameters")
print(Brid3)
print("Skelearn fit parameters")
print(Bridge3_check.T)
# Check plynomial degree = 4
ridge4.fit(data_Lin4,z)
Bridge4_check = ridge4.coef_
print("Polynomial degree = 4")
print("My fit parameters")
print(Brid4)
print("Skelearn fit parameters")
print(Bridge4_check.T)
# Check plynomial degree = 5
ridge5.fit(data_Lin5,z)
Bridge5_check = ridge5.coef_
print("Polynomial degree = 5")
print("My fit parameters")
print(Brid5)
print("Skelearn fit parameters")
print(Bridge5_check.T)

# Let's k-fold with the Ridge.
# Polyonomial degree = 1
MSErid1Fold, R2Srid1Fold = k_fold(x, y, z, 1, ridge1, 5)
print("Polynomial degree = 1")
print("K-fold MSE")
print(MSErid1Fold)
print("K-fold R2S")
print(R2Srid1Fold)
# Polyonomial degree = 2
MSErid2Fold, R2Srid2Fold = k_fold(x, y, z, 2, ridge2, 5)
print("Polynomial degree = 2")
print("K-fold MSE")
print(MSErid2Fold)
print("K-fold R2S")
print(R2Srid2Fold)
# Polyonomial degree = 3
MSErid3Fold, R2Srid3Fold = k_fold(x, y, z, 3, ridge3, 5)
print("Polynomial degree = 3")
print("K-fold MSE")
print(MSErid3Fold)
print("K-fold R2S")
print(R2Srid3Fold)
# Polyonomial degree = 4
MSErid4Fold, R2Srid4Fold = k_fold(x, y, z, 4, ridge4, 5)
print("Polynomial degree = 4")
print("K-fold MSE")
print(MSErid4Fold)
print("K-fold R2S")
print(R2Srid4Fold)
# Polyonomial degree = 5
MSErid5Fold, R2Srid5Fold = k_fold(x, y, z, 5, ridge5, 5)
print("Polynomial degree = 5")
print("K-fold MSE")
print(MSErid5Fold)
print("K-fold R2S")
print(R2Srid5Fold)


# Lasso regression 
lasso1 = linear_model.Lasso(alpha=0.01, fit_intercept=False)
lasso2 = linear_model.Lasso(alpha=0.01, fit_intercept=False)
lasso3 = linear_model.Lasso(alpha=0.01, fit_intercept=False)
lasso4 = linear_model.Lasso(alpha=0.01, fit_intercept=False)
lasso5 = linear_model.Lasso(alpha=0.01, fit_intercept=False)
print("Results of the Lasso regression")
# Ridge regression with degree 1
lasso1.fit(data_Lin1,z)
Blasso1 = lasso1.coef_
print("Polynomial degree = 1")
print("Skelearn fit parameters")
print(Blasso1.T)
# Ridge regression with degree 2
lasso2.fit(data_Lin2,z)
Blasso2 = lasso2.coef_
print("Polynomial degree = 2")
print("Skelearn fit parameters")
print(Blasso2.T)
# Ridge regression with degree 3
lasso3.fit(data_Lin3,z)
Blasso3 = lasso3.coef_
print("Polynomial degree = 3")
print("Skelearn fit parameters")
print(Blasso3.T)
# Ridge regression with degree 4
lasso4.fit(data_Lin4,z)
Blasso4 = lasso4.coef_
print("Polynomial degree = 4")
print("Skelearn fit parameters")
print(Blasso4.T)
# Ridge regression with degree 5
lasso5.fit(data_Lin5,z)
Blasso5 = lasso5.coef_
print("Polynomial degree = 5")
print("Skelearn fit parameters")
print(Blasso5.T)

# Let's k-fold with the Lasso.
# Polyonomial degree = 1
MSElasso1Fold, R2Slasso1Fold = k_fold(x, y, z, 1, lasso1, 5)
print("Polynomial degree = 1")
print("K-fold MSE")
print(MSElasso1Fold)
print("K-fold R2S")
print(R2Slasso1Fold)
# Polyonomial degree = 2
MSElasso2Fold, R2Slasso2Fold = k_fold(x, y, z, 2, lasso2, 5)
print("Polynomial degree = 2")
print("K-fold MSE")
print(MSElasso2Fold)
print("K-fold R2S")
print(R2Slasso2Fold)
# Polyonomial degree = 3
MSElasso3Fold, R2Slasso3Fold = k_fold(x, y, z, 3, lasso3, 5)
print("Polynomial degree = 3")
print("K-fold MSE")
print(MSElasso3Fold)
print("K-fold R2S")
print(R2Slasso3Fold)
# Polyonomial degree = 4
MSElasso4Fold, R2Slasso4Fold = k_fold(x, y, z, 4, lasso4, 5)
print("Polynomial degree = 4")
print("K-fold MSE")
print(MSElasso4Fold)
print("K-fold R2S")
print(R2Slasso4Fold)
# Polyonomial degree = 5
MSElasso5Fold, R2Slasso5Fold = k_fold(x, y, z, 5, lasso5, 5)
print("Polynomial degree = 5")
print("K-fold MSE")
print(MSElasso5Fold)
print("K-fold R2S")
print(R2Slasso5Fold)

# Plots
x_surf, y_surf = np.meshgrid(x,y)
z_surf = FrankeFunction(x_surf,y_surf) + 0.1*np.random.randn(100,1)
# Plot the surface
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
FraPlo = ax.scatter(x_surf, y_surf, z_surf)
# Customize the z axis 
ax.set_zlim( -0.10, 1.40)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
plt.show()