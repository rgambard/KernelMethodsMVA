import numpy as np
from tqdm import tqdm
import scipy.optimize as optimize
import cvxpy as cp
class KernelRegression:
    
    def __init__(self,C, kernel, epsilon = 1e-3):
        self.C= C
        self.kernel = kernel        
        self.epsilon = epsilon
       
    
    def fit(self, X, y):
       #### You might define here any variable needed for the rest of the code
        N = len(y)
        K =self.kernel(X,X)
        def loss(alpha):
            loss = 1/2*alpha.T@K@alpha-self.C*alpha@K@y
            print(loss)
            return loss
        def grad_loss(alpha):
            return K@alpha-self.C*K@y

        constraints = ()

        optRes = optimize.minimize(fun=lambda alpha: loss(alpha),
                                   x0=np.zeros(N), 
                                   method='SLSQP', 
                                   tol = self.epsilon,
                                   options={"disp":True, "maxiter":1000},
                                   jac=lambda alpha: grad_loss(alpha), 
                                   constraints=constraints)
        self.alpha = optRes.x

        self.support = X
        


    ### Implementation of the separting function $f$ 
    def separating_function(self,x):
        # Input : matrix x of shape N data points times d dimension
        # Output: vector of size N
        K = self.kernel(self.support,x)
        
        return np.sum(self.alpha[:,None]*K,axis=0)
    
    
    def predict(self, X):
        """ Predict y values in {-1, 1} """
        d = self.separating_function(X)
        return 2 * (d> 0) - 1

    def score(self, X, Y):
        predictions = self.predict(X)
        correct = (predictions == Y)
        return np.sum(correct)/X.shape[0]

class KernelSVC_cvx:
    
    def __init__(self,C, kernel, epsilon = 1e-3):
        self.C= C
        self.kernel = kernel        
        self.epsilon = epsilon
       
    
    def fit(self, X, y):
       #### You might define here any variable needed for the rest of the code
        N = len(y)
        K =self.kernel(X,X)
        def loss(alphaeps):
            alpha = alphaeps[:N]
            eps = alphaeps[N:]
            loss = 1/2*alpha.T@K@alpha+self.C*np.sum(eps)
            print(loss)
            return loss
        def grad_loss(alphaeps):
            alpha = alphaeps[:N]
            eps = alphaeps[N:]
            return np.concatenate([K@alpha, self.C*np.ones((N,))],axis=0)
        def fun_ineq(alphaeps):
            alpha = alphaeps[:N]
            eps = alphaeps[N:]
            return np.concatenate([y*np.sum(np.diag(alpha)@K, axis=1) -1 +eps, eps], axis=0)
        def ineq_jaq(alphaeps):
            alpha = alphaeps[:N]
            eps = alphaeps[N:]
            return np.concatenate([np.concatenate([y[:,None]*K , np.eye(N)],axis=1),
                                    np.concatenate([np.zeros((N,N)),np.eye(N)],axis=1)],
                                                   axis=0)

        constraints = (#{'type': 'eq',  'fun': fun_eq, 'jac': jac_eq},
                       {'type': 'ineq', 
                        'fun': fun_ineq , 
                        'jac':ineq_jaq})

        optRes = optimize.minimize(fun=lambda alpha: loss(alpha),
                                   x0=np.zeros(2*N), 
                                   method='SLSQP', 
                                   tol = self.epsilon,
                                   options={"disp":True, "maxiter":1000},
                                   jac=lambda alpha: grad_loss(alpha), 
                                   constraints=constraints)
        self.alpha = optRes.x[:N]

        self.support = X
        


    ### Implementation of the separting function $f$ 
    def separating_function(self,x):
        # Input : matrix x of shape N data points times d dimension
        # Output: vector of size N
        K = self.kernel(self.support,x)
        
        return np.sum(self.alpha[:,None]*K,axis=0)
    
    
    def predict(self, X):
        """ Predict y values in {-1, 1} """
        d = self.separating_function(X)
        return 2 * (d> 0) - 1

    def score(self, X, Y):
        predictions = self.predict(X)
        correct = (predictions == Y)
        return np.sum(correct)/X.shape[0]
class GDKernelSVC:
    
    def __init__(self, C, kernel, epsilon = 1e-5):
        self.type = 'non-linear'
        self.C = C                               
        self.kernel = kernel        
        self.alpha = None
        self.support = None # support vectors
        self.epsilon = epsilon
        self.norm_f = None
        self.lambd = 1e-3
       
    
    def fit(self, X, y):
       #### You might define here any variable needed for the rest of the code
        N = len(y)
        K = self.kernel(X,X)
        
        # Lagrange dual problem
        def loss(alpha):
            loss =  1/2*(alpha*y).T@K@(alpha*y) - np.sum(alpha)#'''--------------dual loss ------------------ '''
            print(loss)
            return loss

        # Partial derivate of Ld on alpha
        def grad_loss(alpha):
            grad =  np.diag(y)@K@np.diag(y)@alpha - np.ones(N)# '''----------------partial derivative of the dual loss wrt alpha -----------------'''
            return grad


        # Constraints on alpha of the shape :
        # -  d - C*alpha  = 0
        # -  b - A*alpha >= 0


        alpha = np.zeros(N)
        epsilon = 0.0001
        print("fit")
        oldalpha = alpha.copy()
        for i in range(10000):
            alpha = alpha-self.lambd*grad_loss(alpha)
            alpha = np.clip(alpha,0,self.C)
            
            if np.sqrt(np.sum((alpha-oldalpha)**2))<epsilon:
                break
            if i%2==0:
                print(loss(alpha))
                print("train val", np.mean((y*((y*alpha)@K))>0))
            oldalpha = alpha.copy()
            

        self.alpha = alpha*y

        print("train val", np.mean(y*(self.alpha@K)>0))
        ## Assign the required attributes
        print("support", np.sum(np.abs(self.alpha)>1e-3))

        self.support = X#[np.abs(self.alpha)>1e-8] #'''------------------- A matrix with each row corresponding to support vectors ------------------'''
        #maxb =  np.min(1-self.separating_function(X)[(alpha>1e-8) & (y==1)], initial=100000)#''' -----------------offset of the classifier------------------ '''
        #minb =  np.max(-1-self.separating_function(X)[(alpha>1e-8) & (y==-1)], initial = -100000)
        #print(minb, maxb)
        self.b = 0#(minb+maxb)/2
        self.norm_f = 0# '''------------------------RKHS norm of the function f ------------------------------'''


    ### Implementation of the separting function $f$ 
    def separating_function(self,x):
        # Input : matrix x of shape N data points times d dimension
        # Output: vector of size N
        K = self.kernel(self.support,x)
        
        return np.sum((self.alpha[:,None])*K,axis=0)
    
    
    def predict(self, X):
        """ Predict y values in {-1, 1} """
        d = self.separating_function(X)
        return 2 * (d+self.b> 0) - 1

    def score(self, X, Y):
        predictions = self.predict(X)
        correct = (predictions == Y)
        return np.sum(correct)/X.shape[0]


class KernelSVC:
    
    def __init__(self, C, kernel, epsilon = 1e-3):
        self.type = 'non-linear'
        self.C = C                               
        self.kernel = kernel        
        self.alpha = None
        self.support = None # support vectors
        self.epsilon = epsilon
        self.norm_f = None
       
    
    def fit(self, X, y):
       #### You might define here any variable needed for the rest of the code
        N = len(y)
        K = self.kernel(X,X)
        
        # Lagrange dual problem
        def loss(alpha):
            loss =  1/2*(alpha*y).T@K@(alpha*y) - np.sum(alpha)#'''--------------dual loss ------------------ '''
            print(loss)
            return loss

        # Partial derivate of Ld on alpha
        def grad_loss(alpha):
            grad =  np.diag(y)@K@np.diag(y)@alpha - np.ones(N)# '''----------------partial derivative of the dual loss wrt alpha -----------------'''
            return grad


        # Constraints on alpha of the shape :
        # -  d - C*alpha  = 0
        # -  b - A*alpha >= 0

        fun_eq = lambda alpha: np.sum(alpha*y) # '''----------------function defining the equality constraint------------------'''        
        jac_eq = lambda alpha:  y   #'''----------------jacobian wrt alpha of the  equality constraint------------------'''
        fun_ineq = lambda alpha: np.concatenate((-alpha+self.C, alpha), axis=0)  # '''---------------function defining the inequality constraint-------------------'''     
        jac_ineq = lambda alpha: np.concatenate((-np.eye(N),np.eye(N)),axis=0)  # '''---------------jacobian wrt alpha of the  inequality constraint-------------------'''
        
        constraints = (#{'type': 'eq',  'fun': fun_eq, 'jac': jac_eq},
                       {'type': 'ineq', 
                        'fun': fun_ineq , 
                        'jac': jac_ineq})

        optRes = optimize.minimize(fun=lambda alpha: loss(alpha),
                                   x0=np.zeros(N), 
                                   method='SLSQP', 
                                   tol = self.epsilon,
                                   options={"disp":True, 'maxiter':1000},
                                   jac=lambda alpha: grad_loss(alpha), 
                                   constraints=constraints)
        self.alpha = optRes.x*y
        alpha = optRes.x

        ## Assign the required attributes
        print("support", np.sum(np.abs(self.alpha)>1e-3))

        self.support = X#[np.abs(self.alpha)>1e-8] #'''------------------- A matrix with each row corresponding to support vectors ------------------'''
        #maxb =  np.min(1-self.separating_function(X)[(alpha>1e-8) & (y==1)], initial=100000)#''' -----------------offset of the classifier------------------ '''
        #minb =  np.max(-1-self.separating_function(X)[(alpha>1e-8) & (y==-1)], initial = -100000)
        #print(minb, maxb)
        self.b = 0#(minb+maxb)/2
        self.norm_f = 0# '''------------------------RKHS norm of the function f ------------------------------'''
        print("train val", np.mean(y*(self.alpha@K)>0))


    ### Implementation of the separting function $f$ 
    def separating_function(self,x):
        # Input : matrix x of shape N data points times d dimension
        # Output: vector of size N
        K = self.kernel(self.support,x)
        
        return np.sum((self.alpha[:,None])*K,axis=0)
    
    
    def predict(self, X):
        """ Predict y values in {-1, 1} """
        d = self.separating_function(X)
        return 2 * (d+self.b> 0) - 1

    def score(self, X, Y):
        predictions = self.predict(X)
        correct = (predictions == Y)
        return np.sum(correct)/X.shape[0]

class improvingKernelSVC():
    def __init__(self, C, kernel, keys, epsilon = 1e-3):
        self.C = C
        self.kernel = kernel
        self.epsilon = epsilon
        self.keys = keys
        self.weights = np.ones((len(keys),))
        self.lambd = 10
        self.svc = KernelSVC(C, kernel)

    def fit(self,X,Y, Xacc, Yacc):
        print("start training")
        vecX = self.kernel.to_vectors(X)
        self.kernel.weights = self.weights
        self.svc.fit(X,Y)
        old_weights = self.weights.copy()

        for it in range(1000):
            self.kernel.weights = self.weights
            K = self.kernel(X,X)
            #self.svc.fit(X,Y)
            alpha = self.svc.alpha
            answers = Y*(K@alpha)
            support = answers < 1
            derivatives = np.zeros(len(self.keys))
            for i in range(len(self.keys)):
                Ki=vecX[:,i][:,None]@vecX[:,i][None,:]
                derivative_norm = 1/2*alpha@(Ki@alpha)
                derivative_acc = -self.C*np.sum(support*Y*(Ki@alpha))
                derivatives[i] = derivative_norm+derivative_acc
            #print(derivatives)
            self.weights = self.weights-self.lambd*derivatives
            
            if it %20==0:
                Kacc = self.kernel(Xacc,X)
                norm = alpha@K@alpha
                loss = 1/2*norm-self.C*np.sum(np.minimum(Y*(K@alpha),1))
                lossval = np.sum(np.minimum(Yacc*(Kacc@alpha),1))
                accuracy = np.mean(Yacc*(Kacc@alpha)>0)
                accuracytrain = np.mean(Y*(K@alpha)>0)
                print("loss train",loss)
                print("loss train",loss)
                print("accuracy train", accuracytrain)
                print("loss val", lossval)
                print("accuracy val", accuracy)
                print("norm",1/2*alpha@K@alpha)

                # update weights
                self.weights = len(self.keys)*self.weights/np.sum(self.weights)
                print("weights std : ", self.weights.std())
                print("absolute change : ", np.sum(np.abs(old_weights-self.weights)))
                old_weights = self.weights.copy()
                #self.svc.fit(X,Y)

