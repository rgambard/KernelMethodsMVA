import numpy as np
import matplotlib.pyplot as plt
import tqdm

np.random.seed(10)
def read_X_mat100(filename='Xtr0_mat100.csv'):
    data = []
    with open(filename, 'r') as file:
        for row in file:
            row_s = row.split(' ')
            data.append(np.array(row_s, dtype=float))

    return np.array(data)

def read_Y(filename='Ytr0.csv'):
    data = []
    with open(filename, 'r') as file:
        for row in file:
            row_s = row[-2]
            if row_s == 'd':
                pass
            else:
                data.append(int(row_s))
    return np.array(data)

train_ratio = 90
val_ratio = 10
test_ratio = 100-train_ratio-val_ratio

x0 = read_X_mat100()#[xmin:xmax]
y0 = read_Y()#[xmin:xmax]

x1 = read_X_mat100('Xtr1_mat100.csv')#[xmin:xmax]
y1 = read_Y('Ytr1.csv')#[xmin:xmax]

x2 = read_X_mat100('Xtr2_mat100.csv')#[xmin:xmax]
y2 = read_Y('Ytr2.csv')#[xmin:xmax]

x = x2#np.concatenate((x0,x1,x2))
y = y2#np.concatenate((y0,y1,y2))
y = 2*y-1 #met en -1 et 1

c = list(zip(x, y))
np.random.shuffle(c)
X, Y = zip(*c)

x_train = np.array(X[:train_ratio*len(X)//100])
y_train = np.array(Y[:train_ratio*len(Y)//100])


x_val = np.array(X[train_ratio*len(X)//100:(train_ratio+val_ratio)*len(X)//100])
y_val = np.array(Y[train_ratio*len(Y)//100:(train_ratio+val_ratio)*len(Y)//100])

def param_set(ni=x_train.shape[0]):
    if ni>0:
        step_size = 0.000001
        lambd = 1/100000
        n_steps = 100
        return step_size,lambd,n_steps
    else:
        step_size = 1/(ni**2)
        lambd = 1/(ni**0.5)
        n_steps = 2*ni
    return step_size,lambd,n_steps

def gauss_kernel(x, y, sigma=0.05):
    #compute the norm of x squared
    x_norm = np.sum(x**2, axis=-1)
    #compute the norm of y squared
    y_norm = np.sum(y**2, axis=-1)
    #compute the dot product between x and y
    dot_product = np.dot(x, y.T)
    #compute the kernel matrix
    K = np.exp(-(x_norm[:, None] + y_norm[None, :] - 2 * dot_product) / (2 * sigma**2))
    return K

def euc_kernel(x, y):
    return np.dot(x,y.T)

def SVM_loss(mu,x,y,K,lambd=1/2000):
    smu = np.sum(mu)
    d_y = np.diag(y)
    yKy = np.dot(d_y,np.dot(K,d_y))
    data_attachment = np.dot(mu.T,np.dot(yKy,mu))/(4*lambd)
    return smu - data_attachment

def SVM_solver(x,y,K,lambd,step_size=0.000001,n_steps=1000):
    n = x.shape[0]
    d_y = np.diag(y)
    yKy = np.dot(d_y,np.dot(K,d_y))
    mu = np.zeros(n)
    losses = []
    
    for i in tqdm.tqdm(range(n_steps)):
        mu = mu + step_size *(1 - np.dot(yKy,mu)/(2*lambd))
        mu[mu<0] = 0
        mu[mu>1/n] = 1/n
        loss = SVM_loss(mu,x,y,K,lambd)
        losses.append(loss)
        if i>3 and loss<=losses[-3]:
            print(f'Convergence reached {i}')
            break

    support_vectors = x[mu>0]
    non_support_vectors = x[mu==0]
    print(len(support_vectors))
    mu_support = mu[mu>0]
    y_support = y[mu>0]
    diag_y_support = np.diag(y_support)
    alpha = np.dot(diag_y_support,mu_support)/(2*lambd)
    return alpha,support_vectors,mu,losses,non_support_vectors

#alpha,support_vectors,mu,losses,non_sup = SVM_solver(x_val,y_val,K_val,lambd,step_size,n_steps)
def predict(alpha,support_vectors,kernel,x):
    return np.dot(alpha,kernel(support_vectors,x))

ni = x_train.shape[0]
step_size,lambd,n_steps = param_set()
print(step_size,lambd,n_steps)
K = gauss_kernel(x_train,x_train)
alpha,support_vectors,mu,losses,non_sup = SVM_solver(x_train,y_train,K,lambd,step_size,n_steps)
plt.plot(losses)
plt.show()

prediction_train = predict(alpha,support_vectors,gauss_kernel,x_train)
print(np.mean(prediction_train*y_train>0))

prediction_val = predict(alpha,support_vectors,gauss_kernel,x_val)
print(np.mean(prediction_val*y_val>0))
print(np.mean(prediction_val*y_val==0))

#save the model
np.save('alpha90_x2.npy',alpha)
np.save('support_vectors90_x2.npy',support_vectors)