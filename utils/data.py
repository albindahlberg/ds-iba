# Albin Åberg Dahlberg
import numpy as np
from . import lists

def load_lst_file(file):
    header, events, timing = lists.lstRead(file)
    coin = [True, True, False, False, False, False, False, False]
    zdrop = True
    data = np.array(lists.getCoins(events,coin,zdrop))
    return data

def Phi(x, exponents=[-1/2]):
    """ Feature engineering for ToF data
    
    If:
        X - Energy (outData[0])
        y = ToF (outData[1])
    
    Result: 
        phi = 1/X^(-1/2)
        y = y

    This captures a linear relationship between the variables
    """
    phi = np.empty((x.shape[0], len(exponents)))
    for i, b in enumerate(exponents):
        if b < 0:
            phi[:,i] = (1/np.power(x, -b).flatten())
        else:
            phi[:,i] = np.power(x, b).flatten()
    return phi

def load_tof(file, unique=False):
    """ Reads the data from a ToF experiment
    
    Parameters
    ----------
    file : string
        Name of .lst file to read.

    unique : bool, optional
        Flag for removing duplicate samples
    
    Returns
    ----------
    X : Energy (channels)
    y : ToF (channels)
    """
    data = load_lst_file(file)
    # X = Energy, use for plotting
    X = np.array(data[0]).astype(int)
    # y = ToF
    y = np.array(data[1]).astype(int)
    
    if unique:
        data = np.vstack([X,y]).T
        data = np.unique(data, axis=0)
        X, y = data[:,0], data[:,1]

    return X, y
    