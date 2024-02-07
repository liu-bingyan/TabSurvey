import numpy as np
# Generate uniform and normal distributed data

def generate_artifact_data(n_rows = 50000):
    X1 = np.random.rand(n_rows, 10)
    X2 = np.random.normal(0, 1, (n_rows, 10))  
    X = np.concatenate((X1, X2), axis=1)
    y1 = multiclass_compose(X1)
    y2 = multiclass_compose(X2)
    y = y1 + y2
    return X, y

def multiclass_compose(X):
    y0 = np.ones(X.shape[0])
    #linear group
    y1 = X[:,1]
    y2 = 0.1 * X[:,2]
    y3 = 0.01 * X[:,3]

    #non-linear group
    y4 = X[:,4]**2
    y5 = np.tanh(2*X[:,5]-1)
    y6 = np.sin(2* np.pi * X[:,6])

    #indicator group
    y7 = (X[:,7]>0.75) # w should be big
    y8 = ((X[:,8]>0.5) & (X[:,8]<0.75) ) # at least two bins are active, with big w, and one is positive, one is negative

    #mixture group
    X9 = X[:,9]
    y9_0 = (X[:,9]<0.2)
    y9_1 = (X[:,9]>0.2) & (X[:,9]<0.4)
    y9_2 = (X[:,9]>0.4) & (X[:,9]<0.6) 
    y9_3 = (X[:,9]>0.7) & (X[:,9]<0.75) 
    y9_4 = (X[:,9]>0.8) & (X[:,9]<0.81) 
    y9_5 = (X[:,9]>0.95) 
    
    y9 = y9_0+ np.sin(2*np.pi* X9 * y9_1)+ np.tanh(y9_2) + y9_3*X9 + y9_4 + y9_5

    #sum all together : should I use a non-linear function?
    #y = y0+y1+y2+y3+y4+y5+y6+y7+y8+y9

    y = y7

    return y