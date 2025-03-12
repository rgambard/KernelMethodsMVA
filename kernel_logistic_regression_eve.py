import numpy as np
import matplotlib.pyplot as plt
import tqdm

np.random.seed(13)
def read_X_mat100(filename='Xtr0_mat100.csv'):
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

x = read_X_mat100()
y = read_Y()
y = 2*y-1 #met en -1 et 1

#ni = 100

#x = np.random.randn(ni,2)#np.concatenate((np.random.randn(ni,1), 0*np.random.randn(ni,1)), axis=1)
#y = x[:,0]>0
#y = 2*y-1
#y = (3*x[:,0]+np.random.randn(ni))
#x = np.eye(2)
#y = np.array([0, 1])
n = x.shape[0]
#plt.plot(x[:,0],y, 'o')
#plt.show()

def gauss_kernel(x, y,sigma=0.02):
    return np.exp(-(np.dot((x-y).T,(x-y)))/(2*sigma**2))

def K_gauss_kernel(x, y, sigma=0.02): 
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
    return np.dot(x.T,y)

def poly_kernel(x, y):
    return (np.dot(x.T,y)+1)**2

def kernel_funct(kernel,x):
    return lambda y: kernel(x, y)

def compute_K_matrix(kernel,x):
    K = np.zeros((x.shape[0], x.shape[0]))
    for i in range(x.shape[0]):
        for j in range(i,x.shape[0]):
            K[i,j] = kernel(x[i], x[j])
            K[j,i] = K[i,j]
    return K

def solve_WKRR(K, y,W=np.eye(n),lamb=1): #résolution du problème de minimisation
    n = K.shape[0]
    alpha = np.linalg.solve(np.dot(W,K) + n*lamb*np.eye(K.shape[0]), np.dot(W,y))
    return alpha

def give_f(alpha,kernel,x): #fonction f obtenue
    f = lambda z: np.sum([alpha[i]*kernel_funct(kernel,x[i])(z) for i in range(len(alpha))], axis=0)
    return f    

def compute_error(y_pred, y_true): #erreur
    return np.mean(((y_pred-y_true)/2)**2)

def predict(alpha,kernel,x_train,x):
    return np.dot(alpha,kernel(x_train,x)) #prédiction

def logistic_funct(t): #fonction logistique
    return 1/(1+np.exp(-t))

def WLR_step(alpha_0,K,y,lambd=1,eps=10**(-10)): #résolution du problème de minimisation avec la fonction logistique
    M = np.dot(K,alpha_0)
    P = np.zeros(n)
    W = np.zeros(n)
    z = np.zeros(n)
    for i,mi in enumerate(M):
        P[i] = -logistic_funct(-y[i]*mi)
        W[i] = logistic_funct(y[i]*mi)*logistic_funct(-y[i]*mi)
        #z[i] = mi + y[i]/(logistic_funct(y[i]*mi)+eps)
        z[i] = mi - P[i]*y[i]/W[i]
    W = np.diag(W)
    return solve_WKRR(K, z, W, lambd)
    
def WLR(K,y,alpha0,lambd=1,min_steps = 0,max_steps = 100,eps=(10**(-12))): #fonction qui résout le problème de minimisation avec la fonction logistique
    alpha = alpha0.copy()
    for _ in range(min_steps): #Un certain nombre de pas pour initialiser
        alpha = WLR_step(alpha,K,y,lambd)
    for i in range(min_steps,max_steps): #Un certain nombre de pas pour converger
        alpha_new = WLR_step(alpha,K,y,lambd)
        if np.linalg.norm(alpha_new-alpha)/np.linalg.norm(alpha_new)<eps:
            print(i)
            #si l'ecart est petit, on s'arrête
            break
        alpha = alpha_new
    return alpha

train_ratio = 99
val_ratio = 1
test_ratio = 100-train_ratio-val_ratio

x0 = read_X_mat100()#[xmin:xmax]
y0 = read_Y()#[xmin:xmax]

x1 = read_X_mat100('Xtr1_mat100.csv')#[xmin:xmax]
y1 = read_Y('Ytr1.csv')#[xmin:xmax]

x2 = read_X_mat100('Xtr2_mat100.csv')#[xmin:xmax]
y2 = read_Y('Ytr2.csv')#[xmin:xmax]

x = x1#np.concatenate((x0,x1,x2))
y = y1#np.concatenate((y0,y1,y2))
y = 2*y-1 #met en -1 et 1

c = list(zip(x, y))
np.random.shuffle(c)
X, Y = zip(*c)

x_train = np.array(X[:train_ratio*len(X)//100])
y_train = np.array(Y[:train_ratio*len(Y)//100])

x_val = np.array(X[train_ratio*len(X)//100:(train_ratio+val_ratio)*len(X)//100])
y_val = np.array(Y[train_ratio*len(Y)//100:(train_ratio+val_ratio)*len(Y)//100])

K = K_gauss_kernel(x_train, x_train)
print('kernel done')
n = x_train.shape[0]

alphaLR = WLR(K, y_train,np.ones(n),lambd=0.00001/n)
print(alphaLR)

print('alpha done')
#flr = give_f(alphaLR,gauss_kernel, x)
flr = lambda x: predict(alphaLR, K_gauss_kernel,x_train, x)
print('f done')

pred_train = flr(x_train)#np.array([flr(x_train[i]) for i in tqdm.tqdm(range(len(x_train)))])
pred_train = 2*(pred_train>0)-1


pred_val = flr(x_val)#np.array([flr(x_val[i]) for i in tqdm.tqdm(range(len(x_val)))])
#pred_val = 2*(pred_val>0)-1

print(np.mean(pred_train*y_train>0))
print(np.mean(pred_val*y_val>0))
print(np.mean(pred_train==0))
print(np.mean(pred_val==0))

#save model
np.save('alphaLR_x1.npy', alphaLR)
np.save('x_train_x1.npy', x_train)
