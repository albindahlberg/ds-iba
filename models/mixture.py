import numpy as np
import scipy
from tqdm import tqdm
from scipy.linalg import qr
from math import ceil
import matplotlib.pyplot as plt



class MixIRLS:
    """
    Sequential algorithm for Mixed Linear Regression.

    Originally written by Pini Zilber & Boaz Nadler / 2023.
    Modified by Albin Åberg Dahlberg, Nils Staffsgård, Johan Hedenström, Folke Hilding / 2024.

    Modifications include:
        - Making the algorithm a class
        - Refinement step for weights after inner IRLS to follow regression trajectory
        - Change the requirement for S'
        - Estimated variance of components 
        - Constrained lasso regression instead of least squares at the end of algorithm
        - Intercept option
        - Plotting of first features (non-intercept) option
        - Removal of phase 2

    Parameters
    ----------
    K : int, optional (default=2)
        The etamber of clusters (mixture components) to fit.

    rho : float, optional (default=2)
        Oversampling parameter for the IRLS algorithm.

    eta : float, optional (default=1)
        Parameter for the IRLS algorithm.

    w_th : float, optional (default=0.95)
        Threshold for weight refinement after the inner IRLS step.

    alpha : float or None, optional (default=None)
        Regularization parameter for the constrained lasso regression. If None, no regularization is applied.

    intercept : bool, optional (default=True)
        Whether to include an intercept in the regression models.

    unknownK : bool, optional (default=False)
        If True, the algorithm dynamically estimates the etamber of clusters (mixture components).

    wfun : callable, optional (default=lambda x: 1/(1+x**2))
        The weight function used in the IRLS algorithm.

    T1 : int, optional (default=1000)
        Maximum etamber of iterations for the inner IRLS loop.

    corrupt_frac : float, optional (default=0)
        Fraction of data assumed to be corrupted (outliers).

    plot : bool, optional (default=False)
        Whether to generate plots of the first features.

    verbose : bool, optional (default=False)
        If True, the algorithm prints detailed progress updates.
    """

    def __init__(self,
                 K=2,
                 rho=2,
                 eta=1,
                 w_th=0.95,
                 intercept=True,
                 unknownK=False,
                 wfun=lambda x: 1/(1+x**2),
                 T1=int(1e3),
                 corrupt_frac=0,
                 plot=False,
                 verbose=False):
        self.K = K
        self.rho = rho
        self.eta = eta
        self.w_th = w_th
        self.intercept = intercept
        self.unknownK = unknownK
        self.wfun = wfun
        self.T1 = T1
        self.corrupt_frac = corrupt_frac
        self.plot = plot
        self.verbose = verbose
        self.supports = np.array([])
        self.Sprim = []


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

        # initialize beta randomly
        beta_init = np.random.randn(d, K)


        # where to store parameters
        self.beta = np.zeros((d,K))
        self.sigma = np.zeros((K,))
        # active samples for each component regression
        supports = np.ones((n,K), dtype=np.bool_)
        self.supports = supports
    
        # store importande index bits for each component
        Sprim = []
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
                if self.verbose:
                    print('use same component 1')
                w = first_component_w
                self.beta[:, 1:] = 0
            else:
                if self.verbose:
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
                if self.verbose:
                    print('w_th ' + str(w_th) + ' is too low! Starting over...')
                w_th = w_th + 0.1
                k = 0
                Sprim = []
                continue
            else: # update index sets
                new_support = supports[:, k].copy()
                new_support[new_support] = np.invert(sprim)

                if k < K-1: # not last component
                    supports[:, k+1] = new_support

            # If K is unknown, fix K when the next component has too few
            # active samples
            if self.unknownK and (next_oversampling < self.rho):
                K = k+1
                self.beta = self.beta[:, :K]
                self.sigma = self.sigma[:K]
                Sprim.append(sprim)

                if self.verbose:
                    print('MixIRLS. found K=' + str(K) + ' components, stopping here')
                break

            Sprim.append(sprim)
            k = k + 1

        self.support = supports
        self.Sprim = Sprim
        return Sprim, supports, iter

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

        # weighted least squares
        beta = np.linalg.lstsq(WX, wy, rcond=None)[0]

        # estimated variance
        sigma = np.mean((y - (X @ beta))**2, axis=0)
        return beta, sigma

    def find_component(self, X, y, beta_init=[]):
        beta, w, iter = self.MixIRLS_inner(X, y, beta_init)

        ##### Refinement step, weights along gaussian likelihood        
        residuals = y - X @ beta
        # Update weights as gaussian
        variance = np.var(residuals)
        w = 1/np.sqrt(2 * np.pi * variance) * np.exp(-residuals**2 / (2 * variance))
        # Rescale theshold w.r.t. maximum weight
        #threshold = self.w_th * np.max(w)
        # Compute the percentile threshold
        percentile_value = np.percentile(w, 100*self.w_th)  # self.percentile is the desired percentile (0-100)

        # Select points with w >= percentile value
        I = w > percentile_value
        
        # Select points with w >= threshold
        #I = w >= self.w_th
        I_count = np.count_nonzero(I)

        beta, sigma = self.weighted_ls(X[I, :], y[I])

        if self.plot:
            pred = X @ beta
            plt.plot(X[:, 1], pred, color='red')
            plt.scatter(X[I, 1], y[I], s=1, color='blue')
            plt.show()
            
            # Sort the values of w
            w_sorted = np.sort(w)

            # Compute the cumulative probabilities
            cdf = np.arange(1, len(w_sorted) + 1) / len(w_sorted)

            # Plot the empirical CDF
            plt.plot(w_sorted, cdf, label='CDF of w')
            plt.axvline(x=percentile_value, color='red', linestyle='-', label=f'{100*self.w_th}th percentile = {percentile_value:.4f}')
            plt.title('Empirical CDF of w')
            plt.xlabel('w')
            plt.ylabel('CDF')
            plt.legend()
            plt.show()
        if self.verbose:
            print(f'Observed error: {np.linalg.norm(X[I, :] @ beta - y[I]) / np.linalg.norm(y[I])}. '
                f'Active support size: {I_count}')

        return beta, sigma, w, iter, I

    def MixIRLS_inner(self, X, y, beta_init):

        n,d = X.shape

        beta = np.zeros((d,))
        Q, R, perm = qr(X, mode='economic', pivoting=True)
        if len(beta_init) == 0:
            # if beta_init is not supplied or == -1, the OLS is used
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
            w = self.wfun(r / (self.eta * s))

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

    def get_component_indeces(self, k):
        """ Yields the points of a specified component """
        i = self.supports[:,k] # bit array of points that were considered
        j = self.Sprim[k]  # bit array of used points
        return i, j

    def get_component_points(self, X, y, k):
        """ Yields the points of a specified component """
        i, j = self.get_component_indeces(k)
        return X[i][j], y[i][j]




class MixtureLinearRegression():
    """Implementation of Mixture of Linear Regressions. 
    
    Based on Section 14.5.1 - Pattern Recognition and Machine Learning by Bishop, 2006. 
    Some components usually 'die out' on ToF-ERDA, making them have 0 probability to explain any data point.

    Parameters
    ----------
    K : int
        etamber of components

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
        # etamber of components
        self.K = K
        # Initialize component variance
        self.beta_k = np.full(self.K, beta)
        # Weights for linreg (D x K), initialize all to 0
        self.W = np.zeros(K)
        # Responsibility, initiated when training (N x K)
        self.gamma_nk = 0
        # Mixture weights (K,)
        self.pi = np.zeros((1, self.K)) + 1/(K)
        # etamber of iterations
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

        # Use log-sum-exp trick for etamerical stability
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
