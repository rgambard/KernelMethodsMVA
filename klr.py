import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from kernels import spectrum_kernel, linear_kernel
from ksvc import KernelSVC, KernelSVC_cvx




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
    x0 = pd.read_csv('data/Xtr2.csv')
    x0 = x0['seq'].values
    kernel = spectrum_kernel(5).kernel

else:
    x0 = read_X_mat100('data/Xtr0_mat100.csv')
    x0 = x0/x0.std()
    kernel = linear_kernel().kernel


train_ratio = 20
val_ratio = 50

y0 = read_Y('data/Ytr2.csv')
x = x0
x = x
y = y0
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


kernel = linear_kernel(kernel=kernel).kernel

svm_cvx=False
svm =True
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
    model = KernelSVC(0.5,kernel)
    model.fit(x_train, y_train)
    print(f"Accuracy: {model.score(x_val, y_val):.2f}")
    print(f"Accuracy on train: {model.score(x_train, y_train):.2f}")
if klr:
    print("logistic regression")
    model = LogisticRegression(solver='lbfgs', max_iter=10000000, verbose=3, C = 1)
    model.fit(kernel(x_train,x_train), y_train)
    print(f"Accuracy: {model.score(kernel(x_val, x_train), y_val):.2f}")
    print(f"Accuracy on train: {model.score(kernel(x_train, x_train), y_train):.2f}")

