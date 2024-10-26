import numpy as np
import matplotlib.pyplot as plt

load_data = np.load('data_project2.npz')
data = load_data['arr_0']

def get_data(idx):
    """
    Get one training instance from the data set at the index idx. 
    
    The data is assumed to be in an array `data`.
    
    Args:
        idx (int): An integer specifying which of the training example to fetch
        
    Returns:
        x (array): An array of shape (time_steps, 3) which specifies the input to
                   the neural network. The first column is the time and the second
                   and third columns specify the (x, y) coordinates of the second
                   particle. Note that the first particle is always assumed to be
                   at (1, 0) and the third particle can be inferred from the first
                   and second particle's position.
                   
        y (array): An array of shape (time_steps, 4) which specifies the output that 
                   is expected from the neural network.
                   
                   The first two columns specify the (x, y) coordinates of the first
                   particles and the next two columns give the coordinates of the 
                   second particle for the specified time (length of the columns).
                   The third particles position can be inferred from the first
                   and second particle's position.
    """
    #format : [t, x_1, y_1, x_2, y_2, v_{x,1}, v_{y,1}, v_{x,2}, v_{y,2}]
    data_instance = data[idx]
    t = data_instance[:,0]
    x1 = data_instance[:,1]
    y1 = data_instance[:,2]
    x2 = data_instance[:,3]
    y2 = data_instance[:,4]

    x = np.column_stack((t,x2,y2))
    y = np.column_stack((x1,y1,x2,y2))

    return x,y
    #raise NotImplementedError

def get_trajectories(pred):
    """
    Gets the trajectories from a predicted output pred.
    
    Args:
        pred (array): An array of shape (N, 4) where N is the number of time
                      steps. The four columns give the positions of the particles
                      1 and 2 for all the time steps.
    Returns:
        p1, p2, p3 (tuple of arrays): Three arrays of dimensions (N, 2) where N is the number 
                             of time steps and the two columns for each array give 
                             the positions of the three particles (p1, p2, p3)
    """
    x1 = pred[:,0]
    y1 = pred[:,1]
    x2 = pred[:,2]
    y2 = pred[:,3]
    x3 = -x1-x2
    y3 = -y1-y2
    
    p1 = np.column_stack((x1,y1))
    p2 = np.column_stack((x2,y2))
    p3 = np.column_stack((x3,y3))

    return (p1,p2,p3)
    #raise NotImplementedError

def plot_trajectories(p1, p2, p3, ax=None, **kwargs):
    """
    Plots trajectories for points p1, p2, p3
    
    Args:
        p1, p2, p3 (array): Three arrays each of shape (n, 2) where n is the number
                            of time steps. Each array is the (x, y) position for the
                            particles
        ax (axis object): Default None, in which case a new axis object is created.
        kwargs (dict): Optional keyword arguments for plotting
        
    Returns:
        ax: Axes object
    """
    if ax is None:
        fig, ax = plt.subplots()
        
    x1 = p1[:,0]
    y1 = p1[:,1]
    x2 = p2[:,0]
    y2 = p2[:,1]
    x3 = p3[:,0]
    y3 = p3[:,1]

    ax.plot(x1, y1, label="p1", **kwargs)
    ax.plot(x2, y2, label="p2", **kwargs)
    ax.plot(x3, y3, label="p3", **kwargs)

    ax.set_xlabel('X position')
    ax.set_ylabel('Y position')
    ax.legend()
    
    return ax
    #raise NotImplementedError