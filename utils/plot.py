import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def make_palette(K, cmap=plt.cm.rainbow):
    """ Creates an evenly spaced color map with K colors """
    indices = np.linspace(0, 1, K)
    return cmap(indices)

def plot_predictions(x, y, y_pred, model, size=1, fill=False, cmap=plt.cm.rainbow):
    """ Plots predictions over the true data points.
    
    If fill, the beta parameters of the MLR will be used to visualize the component variances.
    
    Parameters:
    ----------
    x : X (Energy)
    y : y (ToF)
    y_pred : predicted ToF
    size : size of sample points in plot
    fill : flag for plotting variance
    cmap : color map to use
    """
    p = model.params()
    colors = make_palette(p['K'], cmap=cmap)
    
    sort_index = np.argsort(x.flatten())
    x = x[sort_index]
    y = y[sort_index]
    y_pred = y_pred[sort_index]
    beta = p['beta']
    
    plt.scatter(x, y, c='grey', s=size, alpha=0.1)
    plt.ylabel('ToF (channel)')
    plt.xlabel('Energy (channel)')
    for k in tqdm(range(p['K'])):
        color = colors[k]
        plt.plot(x, y_pred[:,k], color=color, alpha=0.8, label=f'Component {k}')
        if fill:
            # Fill two standard deviations (95% interval)
            plt.fill_between(x, 
                            y_pred[:, k] - 2*np.sqrt(beta[k]), 
                            y_pred[:, k] + 2*np.sqrt(beta[k]), 
                            color=color,
                            alpha=0.3)
        plt.legend(loc='upper right')
    plt.show()