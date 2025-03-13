import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from kernels import spectrum_kernel, linear_kernel, LAkernel, sum_kernel, constant_kernel, sqrt_kernel, stats, to_vectors, spectrum_kernel_vec, swscore, convert, get_weights
from ksvc import KernelSVC, KernelSVC_cvx, improvingKernelSVC, GDKernelSVC




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

k=5 

data = pd.read_csv('data/Xtr0.csv')
seqs = data['seq'].values
x0 = seqs
data = pd.read_csv('data/Xte0.csv')
seqs = data['seq'].values
xt0 = seqs



y0 = read_Y('data/Ytr0.csv')

data = pd.read_csv('data/Xtr1.csv')
seqs = data['seq'].values
x1 = seqs
data = pd.read_csv('data/Xte1.csv')
seqs = data['seq'].values
xt1 = seqs



y1 = read_Y('data/Ytr1.csv')
data = pd.read_csv('data/Xtr2.csv')
seqs = data['seq'].values
x2 = seqs
data = pd.read_csv('data/Xte2.csv')
seqs = data['seq'].values
xt2 = seqs



y2 = read_Y('data/Ytr2.csv')

def SVM(x,y,k):
    x = np.array(x)
    y = np.array(y)
    statsx = stats(x,k)
    keys = sorted(statsx.keys())
    print("nb keys", len(keys))

    kernel = spectrum_kernel_vec(k,keys)
    model = GDKernelSVC(0.4,kernel)
    y = 2*y-1
    model.fit(x,y)
    return model

model1 = SVM(x0,y0,k)
model2 = SVM(x1,y1,k)
model3 = SVM(x2,y2,k)

p0 = model1.predict(xt0)
p1 = model2.predict(xt1)
p2 = model3.predict(xt3)

prediction = np.concatenate((p0,p1,p2))
#save prediction
def save_prediction(prediction, filename='Yte0.csv'):
    with open(filename, 'w') as file:
        file.write('Id,Bound\n')
        for i in range(len(prediction)):
            file.write(str(i)+','+str(int(prediction[i]>0))+'\n')

save_prediction(prediction, 'Yte_roms.csv')
