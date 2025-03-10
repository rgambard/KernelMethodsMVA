import numpy as np
import scipy.optimize as optimize
import cvxpy as cp
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
            return 1/2*alpha.T@K@alpha+self.C*np.sum(eps)
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

        self.support = X[np.abs(self.alpha)>1e-8]
        


    ### Implementation of the separting function $f$ 
    def separating_function(self,x):
        # Input : matrix x of shape N data points times d dimension
        # Output: vector of size N
        K = self.kernel(self.support,x)
        
        return np.sum((self.alpha[np.abs(self.alpha)>1e-8,None])*K,axis=0)
    
    
    def predict(self, X):
        """ Predict y values in {-1, 1} """
        d = self.separating_function(X)
        return 2 * (d> 0) - 1

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

        self.support = X[np.abs(self.alpha)>1e-8] #'''------------------- A matrix with each row corresponding to support vectors ------------------'''
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
        
        return np.sum((self.alpha[np.abs(self.alpha)>1e-8,None])*K,axis=0)
    
    
    def predict(self, X):
        """ Predict y values in {-1, 1} """
        d = self.separating_function(X)
        return 2 * (d+self.b> 0) - 1

    def score(self, X, Y):
        predictions = self.predict(X)
        correct = (predictions == Y)
        return np.sum(correct)/X.shape[0]
