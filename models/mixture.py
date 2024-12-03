import numpy as np
import scipy
from tqdm import tqdm
from scipy.linalg import qr
from math import ceil
import matplotlib.pyplot as plt
from scipy.optimize import minimize

class MixIRLS:
    #
    #   Sequential algorithm for Mixed Linear Regression
    #   Originally written by Pini Zilber & Boaz Nadler / 2023
    #   Modified by Albin Åberg Dahlberg, Nils Staffsgård / 2024
    def __init__(self,
                 K = 2,
                 rho = 2,
                 nu = 1,
                 w_th = 0.1,
                 lasso = None,
                 intercept = True,
                 unknownK = False,
                 wfun = lambda r: 1/(1+r**2),
                 T1 = int(1e3),
                 T2 = int(1e3),
                 tol = 2e-16,
                 corrupt_frac = 0,
                 phase_2 = False,
                 errfun = lambda x: 0,
                 plot = False,
                 verbose = False,
                 ):
        self.beta = None
        self.sigma = None
        self.K = K
        self.rho = rho
        self.nu = nu
        self.w_th = w_th
        self.lasso = lasso

        self.intercept = intercept
        self.unknownK = unknownK

        self.wfun = wfun

        self.T1 = T1
        self.T2 = T2

        self.tol = tol
        self.corrupt_frac = corrupt_frac
        self.errfun = errfun

        self.phase_2 = phase_2
        
        self.plot = plot
        self.verbose = verbose
    
    
    # weighted least squares
    def weighted_ls(self, X, y, w=[]):
        if len(w) == 0:
            w = np.ones(len(y),)
        ws = w


        WX = ws[:, np.newaxis] * X
        if len(y.shape) > 1: # y is a matrix
            wy = ws[:, np.newaxis] * y
        else:
            wy = ws * y

        try:
            # Use numpy's built-in least squares solver
            beta = np.linalg.lstsq(WX, wy, rcond=None)[0]
        except np.linalg.LinAlgError:
            # If that fails, try using the pseudo-inverse
            XtWX = WX.T @ WX
            XtWy = WX.T @ wy
            beta = np.linalg.pinv(XtWX) @ XtWy

        sigma = np.mean((wy - (WX @ beta))**2, axis=0)

        return beta, sigma

    def detect_outliers(self, res, corrupt_frac):
    # input: res of size (n, K) and corrupt_frac between 0 and 1
    # estimates outliers and returns an array with 1 for outlier and 0 for inlier

        n = res.shape[0]
        outlier_indicator = np.zeros((n,), dtype=np.bool_)
        M = np.min(res, axis=1) # take minimal res component per sample
        if corrupt_frac > 0: # mark outliers
            outlier_supp = np.argpartition(M, -round(corrupt_frac*n))[-round(corrupt_frac*n):]
            outlier_indicator[outlier_supp] = 1
    
        return outlier_indicator

    def cluster_by_beta(self, beta, X, y, corrupt_frac):
    # cluster samples into components
    # if robustness > 0, mark outliers with c_hat < 0
        K = beta.shape[1]
        res = np.zeros((len(y), K))
        for k in range(K):
            res[:, k] = abs(X @ beta[:, k] - y)
        outlier_indicator = self.detect_outliers(res, corrupt_frac)
        I = np.argmin(res, axis=1)
        I[outlier_indicator.astype(np.int8)] = -1
        return I

    def OLS(self, X, y, K, c):
        # perform OLS to each component individually according to c
        beta = np.zeros((X.shape[1], K))
        sigma = np.zeros((K,))
        for k in range(K):
            beta[:,k], sigma[k] = self.weighted_ls(X[c==k,:], y[c==k])
        return beta, sigma
    
    def train(self, X, y):
        # add intercept
        if self.intercept:
            X = np.c_[np.ones(len(X),), X]        

        n, d = X.shape
        
        ## set parameters
        K = 0
        if self.unknownK:
            K = 20
        else:
            K = self.K
            
        verbose = self.verbose

        # initialize beta
        beta_init = np.random.randn(d, K)

        # list to store sample indexes vor each K
        S_prim = []
        
        ###########################
        ######### phase I #########
        ###########################
        self.beta = np.zeros((d,K))
        self.sigma = np.zeros((K,))
        supports = np.ones((n,K), dtype=np.bool_) # active samples for each component regression
        w_th = self.w_th
        first_component_w = np.zeros_like(y)
        iter = 0
        k = 0
        while k < K:
            # must also iterate for last component, as some of the active samples might
            # belong to other components, or be data outliers

            # currenet active samples
            curr_X = X[supports[:,k],:]
            curr_y = y[supports[:,k]]
            if self.plot:
                plt.scatter(curr_X[:, 1], curr_y, color='grey', s=1, alpha=0.4)

            # if we repeat due to too low w_th, don't calculate first
            # component again as we can take it as is
            if k==0 and np.any(first_component_w):
                # get weights from the first component
                if verbose:
                    print('use same component 1')
                w = first_component_w
                self.beta[:, 1:] = 0
            else:
                if verbose:
                    print('find component ' + str(k+1))

                # find component
                beta, sigma, w, inner_iter, sprim = self.find_component(curr_X, curr_y, beta_init[:,k])
                iter = iter + inner_iter
                self.beta[:,k] = beta.flatten()
                self.sigma[k] = sigma
                if k==0: # store first component in case we restart MixIRLS
                    first_component_w = w
            
            next_oversampling = max(0, np.count_nonzero(w <= w_th) - self.corrupt_frac * n) / d
            
            if not self.unknownK and (k < K-1) and (next_oversampling < self.rho): # need more active samples
                if verbose:
                    print('w_th ' + str(w_th) + ' is too low! Starting over...')
                w_th = w_th + 0.1
                k = 0
                S_prim = []
                continue
            else: # update index sets
                new_support = supports[:, k].copy()
                new_support[new_support] = w <= w_th
                
                if k < K-1: # not last component
                    supports[:, k+1] = new_support

            if verbose:
                print('MixIRLS. error: ' + '{:.3e}'.format(self.errfun(self.beta)) + ', \tk: ' + str(k+1))

            # If K is unknown, fix K when the next component has too few
            # active samples
            if self.unknownK and (next_oversampling < self.rho):
                K = k+1
                self.beta = self.beta[:, :K]
                self.sigma = self.sigma[:K]
                S_prim.append(sprim)

                if verbose:
                    print('MixIRLS. found K=' + str(K) + ' components, stopping here')
                break

            S_prim.append(sprim)
            k = k + 1

        ###########################
        ######## phase II #########
        ###########################
        iter_phase2 = 0
        if self.phase_2:
            beta_diff = 1
            while (beta_diff > self.tol) and (iter_phase2 < self.T2):
                self.beta_prev = self.beta
                res2 = np.zeros((len(y), K))
                for k in range(K):
                    res2[:, k] = np.abs(X @ self.beta[:, k] - y.flatten())**2

                # caluclate weights (here a component's weight depends on the other
                # components' weight)
                w = 1 / (res2 + 1e-16)
                w = w / np.sum(w + 1e-16, axis=1)[:, np.newaxis]
                highs = np.any(w>=2/3, axis=1)
                w_highs = w[highs,:]
                w_highs[w_highs>=2/3] = 1
                w_highs[w_highs<2/3] = 0
                w[highs,:] = w_highs
                lows = np.any(w<1/K, axis=1)
                w_lows = w[lows,:]
                w_lows[w_lows<1/K] = 0
                w[lows,:] = w_lows
                w = w / np.sum(w + 1e-16, axis=1)[:, np.newaxis]

                # ignore estimated outliers
                outlier_indicator = self.detect_outliers(res2, self.corrupt_frac)
                samples_to_use = ~outlier_indicator
                # calculate new self.beta
                for k in range(K):
                        b, s =  self.weighted_ls(X[samples_to_use,:], y[samples_to_use], w[samples_to_use,k])
                        self.beta[:, k] = b.flatten() 
                        self.sigma[k] = s[0]
                beta_diff = np.linalg.norm(self.beta - self.beta_prev, 'fro') / np.linalg.norm(self.beta, 'fro')
        
                # update iter and report
                iter_phase2 = iter_phase2 + 1
                if verbose and (iter_phase2 % 10 == 0):
                    print('Mix-IRLS. error: ' + '{:.3e}'.format(self.errfun(self.beta)) + ', \tphase2-iter: ' + str(iter_phase2))

        iter_tot = iter + iter_phase2
        return S_prim, supports, iter_tot


    ## auxiliary functions
    def find_component(self, X, y, beta_init=[]):
        # INPUT:
        # wfun = IRLS reweighting function
        # nu = tuning parameter used in IRLS reweighting function
        # rho = minimal oversampling to detect component
        # iterlim_inner = max inner iters
        # beta_init - initialization
        # OUTPUT:
        # beta = regression over large weights
        # w = final weights
        # iter = inner iters done
        
        d = X.shape[1]
        beta, w, iter = self.MixIRLS_inner(X, y, beta_init)
        #I = w > w_th
        I = np.argpartition(w, -ceil(self.rho * d))[-ceil(self.rho*d):]
        I_count = np.count_nonzero(I)
        beta, sigma = self.weighted_ls(X[I,:], y[I])
        if self.plot:
            pred = X[:,:] @ beta
            plt.scatter(X[I,1], y[I], s=20, color='blue', marker='x')
            plt.plot(X[:,1], pred, color='red')
            plt.show()
        if self.verbose:
            print('observed error: ' + str(np.linalg.norm(X[I,:] @ beta - y[I]) / np.linalg.norm(y[I])) + '. active support size: ' + str(I_count))
        return beta, sigma, w, iter, I


    def MixIRLS_inner(self, X, y, beta_init):
        # if beta_init is not supplied or == -1, the OLS is used
        
        n,d = X.shape

        beta = np.zeros((d,))
        Q, R, perm = qr(X, mode='economic', pivoting=True)
        if len(beta_init) == 0:
            beta[perm], _ = self.weighted_ls(R, Q.T @ y)
        else:
            beta = beta_init
        # adjust residuals according to DuMouchel & O'Brien (1989)
        E, _ = self.weighted_ls(R.T, X[:, perm].T)
        E = E.T
        h = np.sum(E * E, axis=1)
        h[h > 1 - 1e-4] = 1 - 1e-4
        adjfactor = 1 / np.sqrt(1-h)
        # IRLS
        for iter in range(self.T1):
            # residuals
            r = adjfactor * (y.flatten() - X @ beta)
            rs = np.sort(np.abs(r))

            # scale
            s = np.median(rs[d:]) / 0.6745 # mad sigma
            s = max(s, 1e-6 * np.std(y)) # lower bound s in case of a good fit
            if s == 0: # perfect fit
                s = 1

            # weights
            w = self.wfun(r / (self.nu * s))

            # beta
            beta_prev = beta.copy()
            beta[perm], _ = self.weighted_ls(X[:,perm], y.flatten(), w)

            # early stop if beta doesn't change
            if np.all(np.abs(beta-beta_prev) <= np.sqrt(1e-16) * np.maximum(np.abs(beta), np.abs(beta_prev))):
                break

        return beta, w, iter

    def predict(self, X):
        if self.intercept:
            X = np.c_[np.ones(len(X),), X]
        return X @ self.beta

    def predict_k(self, X, k):
        if self.intercept:
            X = np.c_[np.ones(len(X),), X]
        return X @ self.beta[:,k]




class MixtureLinearRegression():
    """Implementation of Mixture of Linear Regressions. 
    
    Based on Section 14.5.1 - Pattern Recognition and Machine Learning by Bishop, 2006. 
    Some components usually 'die out' on ToF-ERDA, making them have 0 probability to explain any data point.

    Parameters
    ----------
    K : int
        Number of components

    beta : float
        Noise precision hyperparameter

    iterations : int
        How many iterations the EM algorithms should go 

    bias : bool
        Flag for including or excluding bias term. For ToF-ERDA, should be True

    epsilon : float
        Regularization term for gamma_nk

    lam : float
        Regularization term for W
        
    eta : float
        Regularization term for beta

    random_state : int
        Choosable random seed. If None, model will do completely random initialization
    """    
    def __init__(self, K, beta, iterations, bias=False, epsilon=1e-3, lam=1e-4, eta=1e-6, random_state=None):
        # Number of components
        self.K = K
        # Initialize component variance
        self.beta_k = np.full(self.K, beta)
        # Weights for linreg (D x K), initialize all to 0
        self.W = np.zeros(K)
        # Responsibility, initiated when training (N x K)
        self.gamma_nk = 0
        # Mixture weights (K,)
        self.pi = np.zeros((1, self.K)) + 1/(K)
        # Number of iterations
        self.iterations = iterations
        # Bias flag
        self.bias = bias
        # Regularization terms [ TODO: should be tuned or inferred ]
        self.lam = lam # for weight parameters
        self.epsilon = epsilon # for mixture weights 
        self.eta = eta # for beta
        # Random state
        self.random_state = random_state

    def params(self):
        """ Returns the important parameters of the MLR """
        params = { 'K': self.K,
                  'beta': self.beta_k,
                  'W': self.W,
                  'pi': self.pi,
                  'gamma_nk': self.gamma_nk
                  }
        return params

    def _add_bias(self, X):
        """ Adds a bias/intercept feature to the data. For ToF-ERDA, this is necessary """
        N = X.shape[0]
        return np.concatenate((np.ones((N, 1)), X), axis=1)

    def _e_step(self, X, y):
        """ Calculates the probability of each point belonging each component, gamma_nk 
        
        Side effects: self.gamma_nk 
        """
        # Calculate log likelihoods of Gaussian
        log_probs_k = -(1/2)*(np.log(2*np.pi*(self.beta_k + self.eta)) + (y.reshape(-1,1) - X @ self.W)**2/(self.beta_k+self.eta))

        # Add log of mixing coefficients + regularization
        log_weighted_probs_k = np.log(self.pi + self.epsilon) + log_probs_k

        # Use log-sum-exp trick for numerical stability
        log_norm = scipy.special.logsumexp(log_weighted_probs_k, axis=1, keepdims=True)

        # log likelihood for thresholding
        log_likelihood = np.sum(log_norm)
        
        # Get responsibilities (gamma_nk)
        self.gamma_nk = np.exp(log_weighted_probs_k - log_norm)
        return log_likelihood

    def _m_step(self, X, y):
        """ Updates the parameters to maximize the log likelihood
        
        Side effects: self.W and self.beta_k
        """
        # Update mixture weights
        self.pi = self.gamma_nk.mean(axis=0) 

        for k in range(self.K):
            # Update W matrix (without diag(gamma_nk[:,k]))
            self.W[:, k] = np.linalg.solve((self.gamma_nk[:,k] + self.lam) * X.T @ X, (self.gamma_nk[:,k] + self.lam) * X.T @ y).reshape(-1)

        # Update beta
        self.beta_k = np.mean(self.gamma_nk * (y - (X @ self.W))**2, axis=0)

    def _expectation_maximization(self, X, y, thres=1e-4):
        """ Executes expectation maximization.
        
        Runs for self.iterations iterations or until the difference in log likelihood reaches a threshold """
        prev_log_likelihood = -np.inf
        
        for i in tqdm(range(self.iterations)):
            # E-step
            log_likelihood = self._e_step(X, y)
            # M-step
            self._m_step(X, y)            
            # Check for convergence
            if abs(log_likelihood - prev_log_likelihood) < thres:
                print(f"Converged after {i+1} iterations.")
                return

            prev_log_likelihood = log_likelihood

        print(f"Stopped after {self.iterations} iterations.")

    def train(self, X, y):
        """ Trains the MLR on data """
        if self.bias:
            # Add bias feature
            X = self._add_bias(X)
        # Initialize variables
        _, D = X.shape
        # Initialize regression weights
        if self.random_state:
            random_state = np.random.RandomState(self.random_state)
            self.W = -random_state.uniform(low=y.min(), high=y.max(), size=(D, self.K))
        else: 
            self.W = -np.random.uniform(low=y.min(), high=y.max(), size=(D, self.K))
        # Update mixture parameters via EM algorithm
        self._expectation_maximization(X, y)

    def step(self, X, y):
        """ Do one iteration of EM, can be used to plot model iteratively """
        if self.bias:
            X = self._add_bias(X)
        # if W uninitialized
        if (self.W == 0).all():
            _, D = X.shape
            if self.random_state:
                random_state = np.random.RandomState(self.random_state)
                self.W = random_state.uniform(low=y.min(), high=y.max(), size=(D, self.K))
            else: 
                self.W = np.random.uniform(low=y.min(), high=y.max(), size=(D, self.K))
        # E-step
        _ = self._e_step(X, y)
        # M-step
        self._m_step(X, y)     

    def predict(self, X):
        """ Predicts data deterministically
        
        If areas of uncertainty want to be plotted, use params() function to extract the variances beta
        """
        if self.bias:
            X = self._add_bias(X)
        # Prediction will be the deterministic means of the Linear Regressions
        return X @ self.W
