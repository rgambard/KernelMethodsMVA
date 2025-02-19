import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from collections import Counter




def multispectrum_kernel(x,y):
    return spectrum_kernel(x,y,n_spectr=1)+4*spectrum_kernel(x,y,n_spectr = 2)+9*spectrum_kernel(x,y,n_spectr=3) + 16* spectrum_kernel(x,y,n_spectr = 4)
#spectrum kernel
def spectrum_kernel(x,y, n_spectr = 2):
    """ compute the kernel between x and y """
    n = len(x)
    m = len(y)
    subsequencesx = [x[i:i+n_spectr] for i in range(len(x)-n_spectr+1)]
    subsequencesy = [y[i:i+n_spectr] for i in range(len(x)-n_spectr+1)]
    i = 0
    j = 0
    nb_same_substrings = 0
    dictsubx = Counter(subsequencesx)
    dictsuby = Counter(subsequencesy)
    nb_same_substrings = sum(dictsubx[sub]*dictsuby[sub] for sub in dictsubx.keys()&dictsuby.keys())
    return nb_same_substrings



def kernel_matrix(X, Y, kernel_fct):
    """ compute the kernel beetween all elements in X and all elements in Y """
    n = X.shape[0]
    m = Y.shape[0]
    kernel_mat = np.zeros((n,m))

    for i in tqdm(range(n), desc="computing kernel matrix"):
        for j in range(m):
            kernel_mat[i,j] = kernel_fct(X[i], Y[j])
    return kernel_mat



#we load the data
xin = pd.read_csv('data/Xtr2.csv')
yin = pd.read_csv('data/Ytr2.csv')
data = pd.merge(xin, yin, on='Id', how='inner')
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

# we split the dataset to get a validation set
n_train = 1600
n_test = 400
X_train, X_test = data['seq'].values[:n_train], data['seq'].values[n_train:n_train+n_test]
Y_train, Y_test = data['Bound'].values[:n_train], data['Bound'].values[n_train:n_train+n_test]


# we compute the Gram matrix for train and test
K_train = kernel_matrix(X_train, X_train, spectrum_kernel)
K_test = kernel_matrix(X_test, X_train, spectrum_kernel)

print("built kernel matrices")

# we solve the logistic regression problem
model = LogisticRegression(solver='lbfgs', max_iter=100000, verbose=1, C = 0.00001)
model.fit(K_train, Y_train)
print(f"Accuracy: {model.score(K_test, Y_test):.2f}")
