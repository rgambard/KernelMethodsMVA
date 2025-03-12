import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from kernels import spectrum_kernel, linear_kernel, LAkernel, sum_kernel, constant_kernel, sqrt_kernel, stats, to_vectors, spectrum_kernel_vec, swscore, convert, get_weights
from ksvc import KernelSVC, KernelSVC_cvx, improvingKernelSVC




def read_X_mat100(filename='data/Xtr0_mat100.csv'):
    data = []
    with open(filename, 'r') as file:
        for row in file:
            row_s = row.split(' ')
            data.append(np.array(row_s, dtype=float))

    return np.array(data)

def read_Y(filename='data/Ytr2.csv'):
    data = []
    with open(filename, 'r') as file:
        for row in file:
            row_s = row[-2]
            if row_s == 'd':
                pass
            else:
                data.append(int(row_s))
    return np.array(data)

data_type = "seq"

if data_type=="seq":
#we load the data
    data = pd.read_csv('data/Xtr2.csv')
    seqs = data['seq'].values
    x0 = seqs

    k = 4

    y0 = read_Y('data/Ytr2.csv')
    statsx = stats(x0,k)
    keys = sorted(statsx.keys())
    print("nb keys", len(keys))
    kernel = spectrum_kernel_vec(k,keys)
    #weights = get_weights(x0,y0,k,keys)

    #kernel.weights = np.sqrt(weights)/g0

    #kernel = spectrum_kernel(5).kernel
    #kernel2 = linear_kernel(spectrum_kernel(4).kernel).kernel
    #kernel3 = constant_kernel().kernel
    #kernel = sum_kernel([kernel1, kernel2]).kernel

else:
    x0 = read_X_mat100('data/Xtr0_mat100.csv')
    x0 = x0/x0.std()
    kernel = linear_kernel()


train_ratio = 10
val_ratio = 50

y0 = read_Y('data/Ytr2.csv')
x = np.concatenate([x0], axis=0)
y = np.concatenate([y0], axis=0)
#y = np.concatenate((y,y))
y = 2*y-1 #met en -1 et 1

c = list(zip(x, y))
np.random.seed(13)
np.random.shuffle(c)
X, Y = zip(*c)
x_train = np.array(X[:train_ratio*len(X)//100])
y_train = np.array(Y[:train_ratio*len(Y)//100])

x_val = np.array(X[-val_ratio*len(X)//100:])
y_val = np.array(Y[-val_ratio*len(Y)//100:])

print("training on ", x_train.shape[0]," data points ","val on ", x_val.shape[0], "data points")



svm_cvx=False
svm = False
isvm = True
klr =False
if svm_cvx :
# we solve the logistic regression problem
    print("SVM_cvx")
    model = KernelSVC_cvx(1,kernel)
    model.fit(x_train, y_train)
    print(f"Accuracy: {model.score(x_val, y_val):.2f}")
    print(f"Accuracy on train: {model.score(x_train, y_train):.2f}")
 
if svm :
# we solve the logistic regression problem
    print("SVM")
    model = KernelSVC(0.4,kernel, keys)
    model.fit(x_train, y_train)
    print(f"Accuracy: {model.score(x_val, y_val):.2f}")
    print(f"Accuracy on train: {model.score(x_train, y_train):.2f}")
if isvm :
# we solve the logistic regression problem
    print("SVM")
    model = improvingKernelSVC(1000,kernel, keys)
    model.fit(x_train, y_train, x_val, y_val)
    print(f"Accuracy: {model.score(x_val, y_val):.2f}")
    print(f"Accuracy on train: {model.score(x_train, y_train):.2f}")

if klr:
    print("logistic regression")
    model = LogisticRegression(solver='lbfgs', max_iter=10000000, verbose=3, C = 1)
    #model.fit(kernel(x_train,x_train), y_train)
    model.fit(x_train, y_train)
    print(f"Accuracy: {model.score(x_val, y_val):.2f}")
    print(f"Accuracy on train: {model.score(x_train, y_train):.2f}")
    #print(f"Accuracy: {model.score(kernel(x_val, x_train), y_val):.2f}")
    #print(f"Accuracy on train: {model.score(kernel(x_train, x_train), y_train):.2f}")

