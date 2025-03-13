import numpy as np
from collections import Counter
from tqdm import tqdm
import math
from joblib import Memory
from numba import njit
# Create a memory object to cache results
memory = Memory("cache_directory", verbose=0)

class gauss_kernel():
    def __init__(self, sigma=5, kernel = None):
        self.sigma = sigma
        self.kernel = kernel

    def __call__(self,X, Y):
        if self.kernel is not None:
            x = self.kernel(X,X)
            y = self.kernel(X,Y).T
        else:
            x = X
            y = Y
        x = x.astype(np.float64)
        y = y.astype(np.float64)
        sigma = self.sigma
        #compute the norm of x squared
        x_norm = np.sum(x**2, axis=-1)
        #compute the norm of y squared
        y_norm = np.sum(y**2, axis=-1)
        #compute the dot product between x and y
        dot_product = np.dot(x, y.T)
        #compute the kernel matrix
        K = np.exp(-(x_norm[:, None] + y_norm[None, :] - 2 * dot_product) / (2 * sigma**2))
        return K

@memory.cache
def get_weights(x0,y0,k, keys):
    kernel = spectrum_kernel_vec(k,keys)
    vec0 = kernel.to_vectors(x0[y0==0])
    vec1 = kernel.to_vectors(x0[y0==1])
    weights = np.zeros(len(keys))
    for i in tqdm(range(len(keys))):
        weights[i]=np.abs(np.sum(vec0[:,i][:,None]@vec0[:,i][None,:])+
                    np.sum(vec1[:,i][:,None]@vec1[:,i][None,:])-
                    2*np.sum(vec0[:,i][:,None]@vec1[:,i][None,:]))
    return weights


class sqrt_kernel():
    def __init__(self, kernel):
        self.kernel = kernel
    def __call__(self,X,Y):
        K = self.kernel(X,Y)
        return K
class sum_kernel():
    def __init__(self,kernels):
        self.kernels = kernels
    def __call__(self,X,Y):
        return np.sum([k.kernel(X,Y) for k in self.kernels], axis=0)/len(self.kernels)

class constant_kernel():
    def __init(self):
        pass
    def __call__(self,X,Y):
        return np.ones((X.shape[0],Y.shape[0]))
#spectrum kernel
class linear_kernel():
    def __init__(self,kernel=None):
        self.kernel = kernel

    def __call__(self,X,Y):
        """ compute the kernel between x and y """
        if self.kernel is not None:
            Xk = self.kernel(X,X)
            Yk = self.kernel(X,Y)
            return Xk@Yk
        return X@Y.T

dictionnaryconvert={'C':0,'S':1,'T':2, 'A':3, 'G':4, 'P':5}
def stats(X, nb_spectr=4):
    stats = {}
    for i in range(len(X)):
        x = X[i]
        subsequencesx = [x[k:k+nb_spectr] for k in range(len(x)-nb_spectr+1)]
        for k in range(len(x)-nb_spectr+1):
            sub = x[k:k+nb_spectr]
            if sub not in stats.keys():
                stats[sub]=0
            stats[sub]+=1
    return stats

def to_vectors(X, nb_spectr = 4):
    keys = sorted(stats(X,nb_spectr=nb_spectr).keys())
    keystoind = {keys[i]:i for i in range(len(keys))}
    nX = np.zeros((X.shape[0], len(keys)))
    for i in tqdm(range(len(X))):
        x = X[i]
        for k in range(len(x)-nb_spectr+1):
            sub = x[k:k+nb_spectr]
            nX[i,keystoind[sub]]+=1
    return nX

        

def convert(x):
    return np.array([dictionnaryconvert[c] for c in x], dtype = np.int16)

    


@memory.cache
def spectrumkernel(X,Y,nb_spectr):
    XeqY = X.shape==Y.shape and (X==Y).all()
    K = np.zeros((X.shape[0], Y.shape[0]))
    for i in tqdm(range(X.shape[0])):
        x = X[i]
        subsequencesx = [x[k:k+nb_spectr] for k in range(len(x)-nb_spectr+1)]
        dictsubx = Counter(subsequencesx)
        for j in range(i if XeqY else 0,Y.shape[0]):
            y = Y[j]
            subsequencesy = [y[k:k+nb_spectr] for k in range(len(y)-nb_spectr+1)]
            dictsuby = Counter(subsequencesy)
            Kij = sum(dictsubx[sub]*dictsuby[sub] for sub in dictsubx.keys()&dictsuby.keys())
            Kii = sum(dictsubx[sub]*dictsubx[sub] for sub in dictsubx.keys())
            Kjj = sum(dictsuby[sub]*dictsuby[sub] for sub in dictsuby.keys())
            K[i,j] = Kij/math.sqrt(Kii*Kjj) +1
            if XeqY:
                K[j,i] = Kij/math.sqrt(Kii*Kjj) +1
    return K

class spectrum_kernel_vec():
    def __init__(self, nb_spectr, keys, weights=None):
        self.nb_spectr = nb_spectr
        self.keystoind = {keys[i]:i for i in range(len(keys))}
        self.weights = weights
    
    def __call__(self,X,Y):
        weights = self.weights
        nX = self.to_vectors(X)
        nY = self.to_vectors(Y)
        if weights is not None:
            nX = nX*weights[None,:]
            nY = nY*weights[None,:]
        nX = nX/np.sqrt(np.diag(nX@nX.T))[:,None]
        nY = nY/np.sqrt(np.diag(nY@nY.T))[:,None]
        return nX@nY.T+1

    def to_vectors(self,X):
        nX = np.zeros((X.shape[0], len(self.keystoind)))
        for i in range(len(X)):
            x = X[i]
            for k in range(len(x)-self.nb_spectr+1):
                sub = x[k:k+self.nb_spectr]
                nX[i,self.keystoind[sub]]+=1
        return nX




class spectrum_kernel():
    def __init__(self, nb_spectr):
        self.nb_spectr = nb_spectr


    def __call__(self,X,Y):
        """ compute the kernel between x and y """
        return spectrumkernel(X,Y, self.nb_spectr)

@memory.cache
def swkernel(X,Y,e,S):
    """ compute the kernel between x and y """
    XeqY = X.shape==Y.shape and (X==Y).all()
    K = np.zeros((X.shape[0], Y.shape[0]))
    for i in tqdm(range(X.shape[0])):
        x = convert(X[i])
        for j in range(i if XeqY else 0,Y.shape[0]):
            y = convert(Y[j])
            Kij = swscore(x,y, e, S)
            Kii = swscore(x,x, e, S)
            Kjj = swscore(y,y, e, S)
            K[i,j] = Kij/math.sqrt(Kii*Kjj) +1
            if XeqY:
                K[j,i] = Kij/math.sqrt(Kii*Kjj) +1
    return K


@njit
def swscore(x,y,g,S):
    n=len(x)
    m = len(y)
    H = np.zeros((n+1,m+1), dtype = np.int32)
    for i in range(1,n+1):
        for j in range(1,m+1):
            H[i,j]= max(H[i-1,j-1]+S[x[i-1],y[j-1]], H[i-1,j]-g, H[i,j-1]-g, 0)
    return np.max(H)

@njit
def lascore(x, y, beta, e, d, S):
    n = len(x)
    m = len(y)
    M = np.zeros((n+1,m+1), dtype = np.float64)
    X = np.zeros((n+1,m+1), dtype = np.float64)
    Y = np.zeros((n+1,m+1),dtype = np.float64)
    X2 = np.zeros((n+1,m+1),dtype = np.float64)
    Y2 = np.zeros((n+1,m+1), dtype = np.float64)
    for i in range(1,n+1):
        for j in range(1,m+1):
            M[i,j] = math.exp(beta*(S[x[i-1],y[j-1]]))*(1+X[i-1,j-1]+Y[i-1,j-1]+M[i-1,j-1])
            X[i,j] = math.exp(beta*d)*M[i-1,j]+math.exp(beta*e)*X[i-1,j]
            Y[i,j] = math.exp(beta*d)*(M[i,j-1]+X[i,j-1])+math.exp(beta*e)*Y[i,j-1]
            X2[i,j] = M[i-1,j]+X2[i-1,j]
            Y2[i,j] = M[i,j-1]+X2[i,j-1]+Y2[i,j-1]

    return 1+X2[n,m]+Y2[n,m]+M[n,m]
    
class SWkernel():
    def __init__(self, e=0.2):
        self.e = e
        self.dictionnaryconvert={'C':0,'S':1,'T':2, 'A':3, 'G':4, 'P':5}
        self.S = -np.ones((6,6))+2*np.eye(6)

    def convert(self,x):
        return np.array([self.dictionnaryconvert[c] for c in x], dtype = np.int16)

    def __call__(self,X,Y):
        """ compute the kernel between x and y """
        return swkernel(X,Y,self.e, self.S)

@memory.cache
def lakernel(X,Y, beta, e, d, S):
        """ compute the kernel between x and y """
        XeqY = X.shape==Y.shape and (X==Y).all()
        K = np.zeros((X.shape[0], Y.shape[0]))
        for i in tqdm(range(X.shape[0])):
            x = convert(X[i])
            for j in range(i if XeqY else 0,Y.shape[0]):
                y = convert(Y[j])
                Kij = math.log(lascore(x,y, beta, e, d, S))/beta
                Kii = math.log(lascore(x,x, beta, e, d, S))/beta
                Kjj = math.log(lascore(y,y,beta, e,d, S))/beta
                K[i,j] = Kij/math.sqrt(Kii*Kjj) +1 # normalization
                if XeqY:
                    K[j,i] = Kij/math.sqrt(Kii*Kjj) +1
        
        return K

class LAkernel():
    def __init__(self,beta=0.2, d=1, e=0.2):
        self.beta = beta
        self.d = d
        self.e = e
        self.dictionnaryconvert={'C':0,'S':1,'T':2, 'A':3, 'G':4, 'P':5}
        self.S = -np.ones((6,6))+2*np.eye(6)

    def convert(self,x):
        return np.array([self.dictionnaryconvert[c] for c in x], dtype = np.int16)

    def __call__(self,X,Y):
        """ compute the kernel between x and y """
        return lakernel(X,Y,self.beta,  self.e,self.d, self.S)
