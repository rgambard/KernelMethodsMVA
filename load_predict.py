import numpy as np
import matplotlib.pyplot as plt

def read_X_mat100(filename='data/Xtr0_mat100.csv'):
    data = []
    with open(filename, 'r') as file:
        for row in file:
            row_s = row.split(' ')
            data.append(np.array(row_s, dtype=float))

    return np.array(data)

def read_Y(filename='data/Ytr0.csv'):
    data = []
    with open(filename, 'r') as file:
        for row in file:
            row_s = row[-2]
            if row_s == 'd':
                pass
            else:
                data.append(int(row_s))
    return np.array(data)

#load model
alpha = np.load('models/alpha90.npy')
print(alpha)
support_vectors = np.load('models/support_vectors90.npy')

def gauss_kernel(x, y, sigma=0.02):
    #compute the norm of x squared
    x_norm = np.sum(x**2, axis=-1)
    #compute the norm of y squared
    y_norm = np.sum(y**2, axis=-1)
    #compute the dot product between x and y
    dot_product = np.dot(x, y.T)
    #compute the kernel matrix
    K = np.exp(-(x_norm[:, None] + y_norm[None, :] - 2 * dot_product) / (2 * sigma**2))
    return K

def predict(alpha,support_vectors,kernel,x):
    return np.dot(alpha,kernel(support_vectors,x))

#load data
xmin = 0
xmax = 1000
x0 = read_X_mat100('data/Xte0_mat100.csv')#[xmin:xmax]
x1 = read_X_mat100('data/Xte1_mat100.csv')#[xmin:xmax]
x2 = read_X_mat100('data/Xte2_mat100.csv')#[xmin:xmax]

x = np.concatenate((x0,x1,x2))
prediction = predict(alpha,support_vectors,gauss_kernel,x)
print(np.mean(prediction>0))

#save prediction
def save_prediction(prediction, filename='Yte0.csv'):
    with open(filename, 'w') as file:
        file.write('Id,Bound\n')
        for i in range(len(prediction)):
            file.write(str(i)+','+str(int(prediction[i]>0))+'\n')

save_prediction(prediction, 'Yte.csv')
