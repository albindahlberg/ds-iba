import numpy as np
import scipy
from tqdm import tqdm

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
