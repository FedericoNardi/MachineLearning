import numpy as np
import pickle
import scipy as scl
from sklearn.model_selection import train_test_split
import warnings
from scipy.special import expit

def PickRandom(x,y,dim):
    X_train = np.zeros([x.shape[0],x.shape[1]])
    X_train = np.zeros([y.shape[0],y.shape[1]])
    for i in range(dim):
        index = np.random.randint(x.shape[0])
        X_train[i] = x[index]


def Logistic_Newton(X,y,max_iters,tolerance):
    #Initialize beta-parameters
    X_copy = X
    y_copy = y
    beta    = np.zeros(X_copy.shape[1])
    Hessian = np.zeros([X_copy.shape[0], X_copy.shape[0]])
    X_hessian = X_copy.T
    norm    = 100
    #first step
    z = np.dot(X_copy, beta)
    p = expit(z)
    gradient = -np.dot(X.T,y-p)
    for i in range(X_copy.shape[1]):
        for j in range(X_copy.shape[0]):
            X_hessian[i,j] = X_hessian[i,j]*p[j]*(1 - p[j])
    Hessian = np.dot(X_hessian, X_copy)
    Hessian_inv = np.linalg.pinv(Hessian)
    
    beta -= np.dot(Hessian, gradient)
    
    for k in range(1,max_iters):
        
        z = np.dot(X, beta)
        p = expit(z)

        gradient      = -np.dot(X_copy.T,y-p)
        
        for i in range(X_copy.shape[1]):
            for j in range(X_copy.shape[0]):
                X_hessian[i,j] = X_hessian[i,j]*p[j]*(1 - p[j])
        Hessian = np.dot(X_hessian, X_copy)
        Hessian_inv = np.linalg.pinv(Hessian)
            

        beta         -= np.dot(Hessian_inv, gradient)
    
        norm          = np.linalg.norm(gradient)

        if(norm < tolerance):
            print("Newton method converged to given precision in %d iterations" % k)
            break

#    return beta, norm    
    
def Logistic_GradDesc(X,y,eta,max_iters,tolerance):
    
    #Initialize beta-parameters
    beta    = np.zeros(X.shape[1])
    norm    = 100
    #first step
    z = np.dot(X, beta)
    p = expit(z)
    gradient = -np.dot(X.T,y-p)
    beta    -= eta*gradient
    norm     = np.linalg.norm(gradient)
    
    for k in range(1,max_iters):
        
        z = np.dot(X, beta)
        p = expit(z)

        gradient      = -np.dot(X.T,y-p)

        beta         -= eta*gradient
    
        norm          = np.linalg.norm(gradient)

        if(norm < tolerance):
            print("Gradient Descent converged to given precision in %d iterations" % k)
            break

    return beta, norm

def Logistic_SteepGradDesc(X,y,eta,max_iters,tolerance):
    
    #Initialize beta-parameters
    beta    = np.zeros(X.shape[1])
    beta_prev = beta.copy()
    norm    = 100
    eta_k   = 0 
    #first step
    z = np.dot(X, beta)
    p = expit(z)
    gradient = -np.dot(X.T,y-p)
    beta    -= eta*gradient
    norm     = np.linalg.norm(gradient)
    
    gradient_prev = gradient.copy()
    
    for k in range(1,max_iters):
        
        z = np.dot(X, beta)
        p = expit(z)

        gradient_prev = gradient.copy()
        gradient      = -np.dot(X.T,y-p)
        
        eta_k         = np.dot((beta - beta_prev),gradient-gradient_prev) / np.linalg.norm(gradient-gradient_prev)**2

        beta_prev     = beta.copy()
        beta         -= eta_k*gradient
    
        norm          = np.linalg.norm(gradient)

        if(norm < tolerance):
            print("Gradient Descent converged to given precision in %d iterations" % k)
            break
    return beta, norm

def Logistic_StocGradDesc(X,y,eta,max_iters,tolerance):
    X_copy = X
    y_copy = y
    X_batch = np.zeros(X.shape)
    y_batch = np.zeros(y.shape)
    # Number of minibathces
    m = 100
    #Element of the minibatches
    n = X.shape[0]/m
    #Initialize beta-parameters
    beta    = np.zeros(X.shape[1])
    norm    = 100
    #first step
    z = np.dot(X, beta)
    p = expit(z)
    gradient = -np.dot(X.T,y-p)
    beta    -= eta*gradient
    norm     = np.linalg.norm(gradient)
    
    #I create the minibatches
    for i in range(X.shape[0]):
        random_index = np.random.randint(X.shape[0]-i)
        X_batch[i,:] = X_copy[random_index,:]
        y_batch[i] = y_copy[random_index]
        del X_copy[random_index,:]
        del y_copy[random_index]
    for k in range(1,max_iters):
        for i in range(m):
            # I pick a minibatch at random
            s = np.random.randint(m)
            z = np.dot(X_batch[s*n:(s+1)*n,:], beta)
            p = expit(z)

            gradient      = -np.dot(X_batch[s*n:(s+1)*n,:].T,y_batch[s*n:(s+1)*n]-p)

            beta         -= eta*gradient
    
            norm          = np.linalg.norm(gradient)
            
        gradient      = -np.dot(X.T,y-p)
        norm          = np.linalg.norm(gradient)
        if(norm < tolerance):
            print("Gradient Descent converged to given precision in %d iterations" % k)
            break
    return beta, norm

def LogisticFit(x,y,Beta):
    ymodel = np.zeros([len(y),1])
    P = expit(np.dot(X, Beta))
    for i in range(len(y)):
        if P[i] < 0.5:
            ymodel[i] = 0
        else:
            ymodel[i] = 1
    return ymodel

def AccuracyTest(y1,y2):
    Accuracy = 0;
    for i in range(len(y1)):
        if y1[i] == y2[i]:
            Accuracy = Accuracy + 1
    Accuracy = Accuracy/len(y1)
    return Accuracy

#Comment this to turn on warnings
warnings.filterwarnings('ignore')

np.random.seed(42) # shuffle random seed generator

# Ising model parameters
L=40 # linear system size
J=-1.0 # Ising interaction
T=np.linspace(0.25,4.0,16) # set of temperatures
T_c=2.26 # Onsager critical temperature in the TD limit
###### define ML parameters
num_classes=2
train_to_test_ratio=0.01 # training samples

# path to data directory
path_to_data="C:\Anaconda\Programes/IsingData/"

# load data
file_name = "Ising2DFM_reSample_L40_T=All.pkl" # this file contains 16*10000 samples taken in T=np.arange(0.25,4.0001,0.25)
data = pickle.load(open(path_to_data+file_name,'rb')) # pickle reads the file and returns the Python object (1D array, compressed bits)
data = np.unpackbits(data).reshape(-1, 1600) # Decompress array and reshape for convenience
data=data.astype('int')
data[np.where(data==0)]=-1 # map 0 state to -1 (Ising variable can take values +/-1)

file_name = "Ising2DFM_reSample_L40_T=All_labels.pkl" # this file contains 16*10000 samples taken in T=np.arange(0.25,4.0001,0.25)
labels = pickle.load(open(path_to_data+file_name,'rb')) # pickle reads the file and returns the Python object (here just a 1D array with the binary labels)

# divide data into ordered, critical and disordered
X_ordered=data[:70000,:]
Y_ordered=labels[:70000]

X_critical=data[70000:100000,:]
Y_critical=labels[70000:100000]

X_disordered=data[100000:,:]
Y_disordered=labels[100000:]

del data,labels

# define training and test data sets
X=np.concatenate((X_ordered,X_disordered))
Y=np.concatenate((Y_ordered,Y_disordered))

# pick random data points from ordered and disordered states 
# to create the training and test sets
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,train_size=train_to_test_ratio)

# full data set
X=np.concatenate((X_critical,X))
Y=np.concatenate((Y_critical,Y))

print('X_train shape:', X_train.shape)
print('Y_train shape:', Y_train.shape)
print()
print(X_train.shape[0], 'train samples')
print(X_critical.shape[0], 'critical samples')
print(X_test.shape[0], 'test samples')

###### apply logistic regression
from sklearn import linear_model

# define regularisation parameter
lmbdas=np.logspace(-5,5,11)

# preallocate data
train_accuracy=np.zeros(lmbdas.shape,np.float64)
test_accuracy=np.zeros(lmbdas.shape,np.float64)
critical_accuracy=np.zeros(lmbdas.shape,np.float64)

train_accuracy_N=np.zeros(lmbdas.shape,np.float64)
test_accuracy_N=np.zeros(lmbdas.shape,np.float64)
critical_accuracy_N=np.zeros(lmbdas.shape,np.float64)

train_accuracy_GD=np.zeros(lmbdas.shape,np.float64)
test_accuracy_GD=np.zeros(lmbdas.shape,np.float64)
critical_accuracy_GD=np.zeros(lmbdas.shape,np.float64)

train_accuracy_SG=np.zeros(lmbdas.shape,np.float64)
test_accuracy_SG=np.zeros(lmbdas.shape,np.float64)
critical_accuracy_SG=np.zeros(lmbdas.shape,np.float64)

train_accuracy_SCG=np.zeros(lmbdas.shape,np.float64)
test_accuracy_SCG=np.zeros(lmbdas.shape,np.float64)
critical_accuracy_SCG=np.zeros(lmbdas.shape,np.float64)

train_accuracy_SGD=np.zeros(lmbdas.shape,np.float64)
test_accuracy_SGD=np.zeros(lmbdas.shape,np.float64)
critical_accuracy_SGD=np.zeros(lmbdas.shape,np.float64)

print('Logistic Stuff')


#    # Logistic Newton
#    # fit training data
#    X_train_copy = X_train
#    Y_train_copy = Y_train
#    beta_N, norm_N = Logistic_Newton(X_train_copy,Y_train_copy, 1000, 1e-4)
#    # check accuracy
#    Ymodel_train_N = LogisticFit(X_train,Y_train,beta_N)
#    Ymodel_test_N = LogisticFit(X_test,Y_test,beta_N)
#    Ymodel_critical_N = LogisticFit(X_critical,Y_critical,beta_N)
#    train_accuracy_N = AccuracyTest(Y_train,Ymodel_train_N)
#    test_accuracy_N = AccuracyTest(Y_test,Ymodel_test_N)
#    critical_accuracy_N = AccuracyTest(Y_critical,Ymodel_critical_N)
#    print('Newton')
#    print('accuracy: train, test, critical')
#    print('liblin: %0.4f, %0.4f, %0.4f' %(train_accuracy_N,test_accuracy_N,critical_accuracy_N) )


# Logistic Gradient descent
# fit training data
beta_GD, norm_GD = Logistic_GradDesc(X_train,Y_train, 0.1, 1000, 1e-4)
# check accuracy
Ymodel_train_GD = LogisticFit(X_train,Y_train,beta_GD)
Ymodel_test_GD = LogisticFit(X_test,Y_test,beta_GD)
Ymodel_critical_GD = LogisticFit(X_critical,Y_critical,beta_GD)
train_accuracy_GD = AccuracyTest(Y_train,Ymodel_train_GD)
test_accuracy_GD = AccuracyTest(Y_test,Ymodel_test_GD)
critical_accuracy_GD = AccuracyTest(Y_critical,Ymodel_critical_GD)
print('Gradient descent')
print('accuracy: train, test, critical')
print('liblin: %0.4f, %0.4f, %0.4f' %(train_accuracy_GD,test_accuracy_GD,critical_accuracy_GD) )


# Logistic Steepest Gradient descent
# fit training data
beta_SG, norm_SG = Logistic_SteepGradDesc(X_train,Y_train, 0.1, 1000, 1e-4)
# check accuracy
Ymodel_train_SG = LogisticFit(X_train,Y_train,beta_SG)
Ymodel_test_SG = LogisticFit(X_test,Y_test,beta_SG)
Ymodel_critical_SG = LogisticFit(X_critical,Y_critical,beta_SG)
train_accuracy_SG = AccuracyTest(Y_train,Ymodel_train_SG)
test_accuracy_SG = AccuracyTest(Y_test,Ymodel_test_SG)
critical_accuracy_SG = AccuracyTest(Y_critical,Ymodel_critical_SG)
print('Steepest Gradient descent')
print('accuracy: train, test, critical')
print('liblin: %0.4f, %0.4f, %0.4f' %(train_accuracy_SG,test_accuracy_SG,critical_accuracy_SG) )


# Logistic Steepest Gradient descent
# fit training data
beta_SCG, norm_SCG = Logistic_SteepGradDesc(X_train,Y_train, 0.1, 1000, 1e-4)
# check accuracy
Ymodel_train_SCG = LogisticFit(X_train,Y_train,beta_SCG)
Ymodel_test_SCG = LogisticFit(X_test,Y_test,beta_SCG)
Ymodel_critical_SCG = LogisticFit(X_critical,Y_critical,beta_SCG)
train_accuracy_SCG = AccuracyTest(Y_train,Ymodel_train_SCG)
test_accuracy_SCG = AccuracyTest(Y_test,Ymodel_test_SCG)
critical_accuracy_SCG = AccuracyTest(Y_critical,Ymodel_critical_SCG)
print('Stochastic Gradient descent')
print('accuracy: train, test, critical')
print('liblin: %0.4f, %0.4f, %0.4f' %(train_accuracy_SCG,test_accuracy_SCG,critical_accuracy_SCG) )
    
print('Sklearn')
# loop over regularisation strength
for i,lmbda in enumerate(lmbdas):
    # define logistic regressor
    logreg=linear_model.LogisticRegression(C=1.0/lmbda,random_state=1,verbose=0,max_iter=1E3,tol=1E-5)
    
    # fit training data
    logreg.fit(X_train, Y_train)

    # check accuracy
    train_accuracy[i]=logreg.score(X_train,Y_train)
    test_accuracy[i]=logreg.score(X_test,Y_test)
    critical_accuracy[i]=logreg.score(X_critical,Y_critical)
    
    print('accuracy: train, test, critical')
    print('liblin: %0.4f, %0.4f, %0.4f' %(train_accuracy[i],test_accuracy[i],critical_accuracy[i]) )
    
    # define SGD-based logistic regression
    logreg_SGD = linear_model.SGDClassifier(loss='log', penalty='l2', alpha=lmbda, max_iter=100, 
                                           shuffle=True, random_state=1, learning_rate='optimal')

    # fit training data
    logreg_SGD.fit(X_train,Y_train)

    # check accuracy
    train_accuracy_SGD[i]=logreg_SGD.score(X_train,Y_train)
    test_accuracy_SGD[i]=logreg_SGD.score(X_test,Y_test)
    critical_accuracy_SGD[i]=logreg_SGD.score(X_critical,Y_critical)
    print('SGD: %0.4f, %0.4f, %0.4f' %(train_accuracy_SGD[i],test_accuracy_SGD[i],critical_accuracy_SGD[i]) )

    print('finished computing %i/11 iterations' %(i+1))