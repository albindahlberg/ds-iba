import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import minimize
from sklearn.cluster import KMeans
from utils import lists  

def load_data(file_path):
    header, events, timing = lists.lstRead(file_path)
    coin = [True, True, False, False, False, False, False, False]
    zdrop = True
    data = np.array(lists.getCoins(events, coin, zdrop))
    X = np.array(data[0]).astype(int)
    y = data[1].astype(int).reshape((-1, 1))
    if X.shape[0] != y.shape[0]:
        raise ValueError("Mismatched sizes: X and y must be the same length")
    return X, y

def Phi(x, exponents=[1, -1/2, 1/2]):
    phi = np.empty((x.shape[0], len(exponents)))
    for i, b in enumerate(exponents):
        if b < 0:
            phi[:, i] = (1 / np.power(x, -b).flatten())
        else:
            phi[:, i] = np.power(x, b).flatten()
    return phi

class AffineCouplingLayer:
    def __init__(self, input_dim):
        self.input_dim = input_dim
        self.split_dim = input_dim // 2
        self.scale = np.random.randn(self.split_dim)
        self.shift = np.random.randn(self.split_dim)
        self.additional_scale = np.random.randn(self.split_dim)

    def forward(self, x):
        x1, x2 = x[:, :self.split_dim], x[:, self.split_dim:]
        s, t = np.tanh(self.scale * x1 + self.additional_scale * np.sin(x1)), self.shift * x1
        y1, y2 = x1, x2 * np.exp(s) + t
        return np.concatenate([y1, y2], axis=1), s

    def inverse(self, y):
        y1, y2 = y[:, :self.split_dim], y[:, self.split_dim:]
        s, t = np.tanh(self.scale * y1 + self.additional_scale * np.sin(y1)), self.shift * y1
        x1, x2 = y1, (y2 - t) * np.exp(-s)
        return np.concatenate([x1, x2], axis=1)

class NormalizingFlow:
    def __init__(self, num_layers, input_dim):
        self.layers = [AffineCouplingLayer(input_dim) for _ in range(num_layers)]
    
    def forward(self, x):
        log_det_jacobian = 0
        for layer in self.layers:
            x, s = layer.forward(x)
            log_det_jacobian += np.sum(s, axis=1)
        return x, log_det_jacobian
    
    def inverse(self, y):
        for layer in reversed(self.layers):
            y = layer.inverse(y)
        return y

def train_normalizing_flow(phi, flow):
    initial_params = np.hstack([np.hstack([layer.scale, layer.shift]) for layer in flow.layers])

    def negative_log_likelihood(params, x, flow):
        param_index = 0
        for layer in flow.layers:
            scale_size = layer.scale.size
            shift_size = layer.shift.size
            layer.scale = params[param_index:param_index + scale_size]
            layer.shift = params[param_index + scale_size: param_index + scale_size + shift_size]
            param_index += scale_size + shift_size
        z, log_det_jacobian = flow.forward(x)
        base_log_prob = np.sum(norm.logpdf(z), axis=1)
        return -np.mean(base_log_prob + log_det_jacobian)

    result = minimize(negative_log_likelihood, initial_params, args=(phi, flow), method='Nelder-Mead')
    optimized_params = result.x

    param_index = 0
    for layer in flow.layers:
        scale_size = layer.scale.size
        shift_size = layer.shift.size
        layer.scale = optimized_params[param_index:param_index + scale_size]
        layer.shift = optimized_params[param_index + scale_size: param_index + scale_size + shift_size]
        param_index += scale_size + shift_size

    print("Optimization success:", result.success)
    print("Final negative log likelihood:", result.fun)
    return flow

def calculate_log_probability(flow, x_point):
    x_point = x_point.reshape(1, -1)
    z, log_det_jacobian = flow.forward(x_point)
    base_log_prob = np.sum(norm.logpdf(z), axis=1)
    log_prob = base_log_prob + log_det_jacobian
    return log_prob[0]

def segment_curves(flow, phi, X, y, num_clusters=5):
    z_transformed, _ = flow.forward(phi)
    kmeans = KMeans(n_clusters=num_clusters, random_state=0)
    cluster_labels = kmeans.fit_predict(z_transformed)

    segmented_data = [phi[cluster_labels == i] for i in range(num_clusters)]
    plt.figure(figsize=(10, 8))
    plt.scatter(X, y, s=0.05, alpha=0.2, color="lightgray", label="Original Data")
    colors = plt.cm.viridis(np.linspace(0, 1, num_clusters))

    for i, segment in enumerate(segmented_data):
        segment_original = flow.inverse(segment)
        plt.scatter(segment_original[:, 0], segment_original[:, 1], s=0.5, color=colors[i], label=f"Cluster {i+1}")

    plt.xlabel("Energy (channel)")
    plt.ylabel("ToF (channel)")
    plt.legend()
    plt.show()

def generate_sample(flow, latent_dim):
    z_sample = np.random.normal(size=(1, latent_dim))  
    x_generated = flow.inverse(z_sample)  
    return x_generated.flatten()

def main():
    FILE_PATH = '../data/tof_erda/raw/I_36MeV_SH2-12_S18.lst'
    X, y = load_data(FILE_PATH)
    phi = Phi(X, exponents=[1, -1/2, 1/2])
    flow = NormalizingFlow(num_layers=3, input_dim=phi.shape[1])
    flow = train_normalizing_flow(phi, flow)
    segment_curves(flow, phi, X, y, num_clusters=5)
    
    x_generated = generate_sample(flow, latent_dim=phi.shape[1])
    log_prob = calculate_log_probability(flow, x_generated)
    probability = np.exp(log_prob)
    
    print("Generated Sample:", x_generated)
    print("Log-Probability of generated sample:", log_prob)
    print("Probability of generated sample:", probability)

if __name__ == "__main__":
    main()
