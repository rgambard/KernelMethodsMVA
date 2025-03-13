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
alpha_x0 = np.load('models/alphaLR_x0.npy')
print(alpha_x0)
x_train_x0 = np.load('models/x_train_x0.npy')

alpha_x1 = np.load('models/alphaLR_x1.npy')
print(alpha_x1)
x_train_x1 = np.load('models/x_train_x1.npy')

alpha_x2 = np.load('models/alphaLR_x2.npy')
print(alpha_x2)
x_train_x2 = np.load('models/x_train_x2.npy')

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

prediction0 = predict(alpha_x0,x_train_x0,gauss_kernel,x0)
prediction1 = predict(alpha_x1,x_train_x1,gauss_kernel,x1)
prediction2 = predict(alpha_x2,x_train_x2,gauss_kernel,x2)
print(np.mean(prediction0>0))
print(np.mean(prediction1>0))
print(np.mean(prediction2>0))

prediction = np.concatenate((prediction0,prediction1,prediction2))

#save prediction
def save_prediction(prediction, filename='Yte.csv'):
    with open(filename, 'w') as file:
        file.write('Id,Bound\n')
        for i in range(len(prediction)):
            file.write(str(i)+','+str(int(prediction[i]>0))+'\n')

save_prediction(prediction, 'Yte.csv')
