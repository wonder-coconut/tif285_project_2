# Load modules
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import time

from helper_funcs import get_data

# Load and unpack a compressed npy array
load_data = np.load('data_project2.npz')
data = load_data['arr_0']
print('data loaded')
tf.config.threading.set_intra_op_parallelism_threads(0)  # Uses all available threads
tf.config.threading.set_inter_op_parallelism_threads(0)

def trajectory_checker(): #checks whether the particles get stuck at origin for all trajectories in data
    
    data_res = []
    
    valid_trajectory_flag = True #flag to designate whether a trajectory is valid or faulty (stuck)
    
    c1 = 0 #counters for identifying faulty trajectories
    c2 = 0
    c3 = 0
    
    for i in data: #looping through trajectories
        valid_trajectory_flag = True
        c2 = 0
        for j in i: #looping through points in trajectory i
            
            x1 = j[1] #parsing coordinates for a specific time data entry
            y1 = j[2]
            x2 = j[3]
            y2 = j[4]
            x3 = -x1-x2
            y3 = -y1-y2
            
            x_equality = (x1 == x2) and (x2 == x3) and (x3 == 0) #validity checking (all particles stuck at origin)
            y_equality = (y1 == y2) and (y2 == y3) and (y3 == 0)
            

            if(x_equality and y_equality): #if all particles are stuck
                #print(c1,c2)
                #print(x1,x2,x3)
                #print(y1,y2,y3)
                c3 += 1
                valid_trajectory_flag = False #invalidate trajectory
                break
            
            c2 += 1
        if(valid_trajectory_flag): #if i is a valid (unstuck) trajectory, add to resultant dataset
            data_res.append(i)
        c1 += 1
    #print(c3)
    return np.array(data_res)

data = trajectory_checker()
print('data validated')

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
    y = 0
    
    data_instance = data[idx] #trajectory at idx
    #format : [t, x_1, y_1, x_2, y_2, v_{x,1}, v_{y,1}, v_{x,2}, v_{y,2}]
    t = data_instance[:,0]
    x1 = data_instance[:,1]
    y1 = data_instance[:,2]
    x2 = data_instance[:,3]
    y2 = data_instance[:,4]
    x2_init = np.array([x2[0]] * len(x2))#initial 2nd particle position
    y2_init = np.array([y2[0]] * len(y2))
    
    x = np.column_stack((t,x2_init,y2_init)) #input params
    y = np.column_stack((x1,y1,x2,y2)) #output params
    
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
    x3 = -x1-x2 #cm frame of reference
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

#data initialization
#splitting into input and output parameters

data_x = []
data_y = []

for i in range(data.shape[0]): #extracting x and y for each trajectory
    x,y = get_data(i)
    data_x.append(x) #appending x and y to a set
    data_y.append(y)

data_x = np.array(data_x)
data_y = np.array(data_y)
print('input and output split')

#splitting into training and validation datasets
split_index = int(0.9 * data.shape[0]) #9:1 ratio of train:val

data_x_train = data_x[:split_index]
data_y_train = data_y[:split_index]
data_x_val = data_x[split_index:]
data_y_val = data_y[split_index:]

#note: all training data should be in a single array and model fitting must be done in a single fit function
#iterating over all trajectories and performing individual fits will end up in an ann that is fitted only to the last trajectory
#learnt this the hard way with over 7 hours wasted on training absolute garbage
#remove this comment for submission

data_x_train = np.row_stack(data_x_train)
data_y_train = np.row_stack(data_y_train)
data_x_val = np.row_stack(data_x_val)
data_y_val = np.row_stack(data_y_val)

trajectory_length = 1000
batch_scale = 15
batch_size = batch_scale*trajectory_length #batch size must be an integer multiple of trajectory length to avoid mixing trajectory data points

#converting data into a tensorflow dataset
dataset_train = tf.data.Dataset.from_tensor_slices((data_x_train, data_y_train))
dataset_val = tf.data.Dataset.from_tensor_slices((data_x_val,data_y_val))

dataset_train = dataset_train.batch(batch_size)
dataset_val = dataset_val.batch(batch_size)

#model synthesis
three_body_model = keras.models.Sequential([
    keras.layers.InputLayer(shape=(3,)),
    keras.layers.Dense(128,activation='relu'),
    keras.layers.Dense(128,activation='relu'),
    keras.layers.Dense(128,activation='relu'),
    keras.layers.Dense(128,activation='relu'),
    keras.layers.Dense(128,activation='relu'),
    keras.layers.Dense(128,activation='relu'),
    keras.layers.Dense(128,activation='relu'),
    keras.layers.Dense(128,activation='relu'),
    keras.layers.Dense(128,activation='relu'),
    keras.layers.Dense(128,activation='relu'),
    keras.layers.Dense(4,activation='linear')]
)

#note: the final layer must have a linear activation function because we are defining this output for real valued parameters
#and not for probabilistic outputs.

print('model generated')

#model compilation
three_body_model.compile(optimizer=keras.optimizers.Adam(0.001, 0.5, 0.5),loss=keras.losses.MeanAbsoluteError(),metrics=['accuracy'])

print('model compiled\n\n\n\nstart training')

#training and validation
three_body_model.fit(dataset_train, epochs=250, validation_data=dataset_val)

tf.keras.models.save_model(three_body_model, 'myNN.keras')