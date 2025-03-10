import numpy as np
from collections import Counter
from tqdm import tqdm
import math

#spectrum kernel
class linear_kernel():
    def __init__(self,kernel=None):
        self.kernel = kernel

    def kernel(self,X,Y):
        """ compute the kernel between x and y """
        if self.kernel is not None:
            Xk = self.kernel(X,X)
            Yk = self.kernel(X,Y)
            return Xk@Yk.T
        return X@Y.T

class spectrum_kernel():
    def __init__(self, nb_spectr):
        self.nb_spectr = nb_spectr


    def kernel(self,X,Y):
        """ compute the kernel between x and y """
        XeqY = X.shape==Y.shape and (X==Y).all()
        K = np.zeros((X.shape[0], Y.shape[0]))
        for i in tqdm(range(X.shape[0])):
            x = X[i]
            print(len(x))
            subsequencesx = [x[k:k+self.nb_spectr] for k in range(len(x)-self.nb_spectr+1)]
            dictsubx = Counter(subsequencesx)
            for j in range(i if XeqY else 0,Y.shape[0]):
                y = Y[j]
                subsequencesy = [y[k:k+self.nb_spectr] for k in range(len(y)-self.nb_spectr+1)]
                dictsuby = Counter(subsequencesy)
                Kij = sum(dictsubx[sub]*dictsuby[sub] for sub in dictsubx.keys()&dictsuby.keys())
                Kii = sum(dictsubx[sub]*dictsubx[sub] for sub in dictsubx.keys())
                Kjj = sum(dictsuby[sub]*dictsuby[sub] for sub in dictsuby.keys())
                K[i,j] = Kij/math.sqrt(Kii*Kjj) +1
                if XeqY:
                    K[j,i] = Kij/math.sqrt(Kii*Kjj) +1

        return K

class LAkernel():
    def __init__(self,beta):
        self.beta = beta

    def kernel(self,X,Y):
        """ compute the kernel between x and y """
        XeqY = X.shape==Y.shape and (X==Y).all()
        K = np.zeros((X.shape[0], Y.shape[0]))
        for i in tqdm(range(X.shape[0])):
            x = X[i]
            for j in range(i if XeqY else 0,Y.shape[0]):
                y = Y[j]
                Kij = 1
                Kii = 1
                Kjj = 1
                K[i,j] = Kij/math.sqrt(Kii*Kjj) +1
                if XeqY:
                    K[j,i] = Kij/math.sqrt(Kii*Kjj) +1

        return K


