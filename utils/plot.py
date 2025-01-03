# Albin Åberg Dahlberg
import numpy as np
import matplotlib.pyplot as plt

def plot_bananas(x, y, title=None,alpha=0.3, s=0.1):
    """ Plots bananas of ToF-ERDA experiments """
    plt.figure(figsize=(10,8))
    plt.scatter(x, y, alpha=alpha, s=s)
    plt.xlabel('ToF (channel)')
    plt.ylabel('energy (channel)')
    plt.title(title)
    plt.show()

def make_palette(K, cmap=plt.cm.hsv):
    """ Creates an evenly spaced color map with K colors """
    indices = np.linspace(0, 1, K+1)
    return cmap(indices)
