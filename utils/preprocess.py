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

def load_tof(file):
    """ Reads the data from a ToF experiment
    
    Returns:
    --------
    X : Energy (channels)
    y : ToF (channels)
    phi : feature engineered energy for linear relationship
        phi = 1/sqrt(X)
    """
    data = load_lst_file(file)
    # X = Energy, use for plotting
    X = np.array(data[0]).astype(int)
    # y = ToF
    y = data[1].astype(int).reshape((-1,1))
    # Feature for training linear model
    phi = Phi(X)
    return X, y, phi
    