# %load NN_Matrix.py
#!/usr/bin/env python3
"""
Created on Sun Apr 22 16:41:19 2018

@author: aditya
"""
import numpy as np
import matplotlib.pyplot as plt
from matmul_opencl import mat_mul


# Generating the training dataset


iter_t = 1000    # working with 1000 sample points
u = np.zeros([iter_t], np.float32)      # Input to the dynamical system
y = np.zeros([iter_t], np.float32)      # Output of the dynamical system
yd = np.zeros([iter_t], np.float32)     # Stores 1-left shifted output
y_act = np.zeros([iter_t], np.float32)  # Output of the neural network
x = np.zeros([iter_t, 3], np.float32)   # Store the two inputs to apply to NN
x[:, 2] = np.ones([iter_t], np.float32)

for i in range(0, iter_t-1):
    u[i] = 2 * np.random.rand()-1                    # random input between -1 and 1
    y [i+1] = ( y[i] / (1+y[i]**2) ) + u[i]**3       # y is the input for calculation of next state    
    yd[i] = y[i+1]

for i in range(0, iter_t-1):
    x[i, 0] = u[i]                                   # For Convinince inputs are stored in x
    x[i, 1] = y[i]

# Plot the input training samples in 2-d space    
plt.figure(1)
plt.grid()
plt.plot(u, yd, '.')


# Initializing the weights and other parametrs of the NN
n0 = 2          # No of inputs/input neurons 
n1 = 15         # no. of Neurons in the first layer
n2 = 1          # No. of Outputs

w = np.zeros([n1, n0+1], np.float32)   # weights between hidden and input layer, hidden layers = 15
W = np.zeros([n2, n1+1], np.float32)   # weights between hidden and output layer

for i in range(0, n1):
    w[i, 0] = (2*np.random.rand()-1) # randomly initialize the weights
    w[i, 1] = (2*np.random.rand()-1) # randomly initialize the weights
    w[i, 2] = (2*np.random.rand()-1) # randomly initialize the weights

for i in range(0, n1+1):
    W[0, i] = (2*np.random.rand()-1)   # randomly initialize the weights

lr = 0.1    # learning rate
epoch = 200 # no. of epochs
p = 1       # p is the index of each epoch
  

# Training the NN


h = np.zeros(n1, np.float32)   # the net activation of hidden neurons
v = np.zeros(n1+1, np.float32) #  final output of hidden neurons
v[n1] = 1
mse = 0     # mean square error for each iterartion is stored in this variable  
mse1 = np.zeros(epoch, np.float32) # mean square error after each epoch is stored in this array
y2 = np.zeros(iter_t, np.float32)
del1 = np.zeros(n1, np.float32)   # Rhe backpropogated sensetivity of hidden layers
y2[0] = y[0]

while ( p < epoch ):
    for j in range(0, iter_t):   # j is the index of each pattern

        # Forward evaluation of Neural Network  
        h = w @ x [j, :].T    # Net Output of the first layer
        # h = mat_mul(w, x[j, :].reshape(3,1))
        v[0:n1] = (np.exp(h) - np.exp(-h)) / (np.exp(h) + np.exp(-h))  # Output after activation

        # y_act[j] = mat_mul(W, v.reshape(n1+1,1))
        y_act[j] = W @ v    # Final Output of the neural network 
        

        mse += 0.5 * ( yd[j] - y_act[j] )**2 / iter_t    # calculating mse at each epoch

        # Calculation of dels and weight update by backpropogation
        del2 = ( yd[j] - y_act[j] )    # Sensetivity of output layer

        W += lr * del2 * v   # Update of output weights

        del1 = (1-v[0:n1]**2) * del2 * W[0, 0:n1]  # Sensetivity of hidden layer


        w += lr * del1.reshape(n1, 1) @ x[j, :].reshape(1, 3) # Update of hidden layer weights


    mse1[p] = mse
    mse = 0 
    p=p+1

# Plot training data and the final learned training data
plt.figure(2)
plt.grid()
plt.plot(yd)
plt.plot(y_act,'r')


# Plot of mse vs epoch
plt.figure(3)
plt.grid()
plt.plot(mse1)

# Testing of NN using different data
iter_t = 1000                          # No. of test data set
y[0] = 0                               # Initial condition            

for i in range(0, iter_t-1):
    u[i] = np.sin( 0.02 * i )                     # Test Input Data
    y[i+1] = ( y[i] / (1+y[i]**2) ) + u[i]**3     # y is the input for calculation of next state 
    yd[i] = y[i]                                  # Output of System for ith instant inputs 
yd[iter_t-1] = y[iter_t-1]  

for i in range(0, iter_t):                             # Storing inputs in x for convinience
    x[i, 0:2] = np.array([u[i], y[i]], np.float32)
    

# Plot using Neural Network
e = np.zeros([iter_t])
y_act[0] = y[0]
for j in range(0, iter_t-1):
    #h = w @ x [j, :].T       # Calculate the net input to hidden layers
    #v[0:n1] = (np.exp(h) - np.exp(-h)) / (np.exp(h) + np.exp(-h))
    #print(w, x[j,:].reshape(3,1))
    h1 = mat_mul(w, x[j, :].reshape(3,1))
    #print("start", h,"gap", h1, "end")
    h1 = h1.reshape(n1)
    v[0:n1] = (np.exp(h1) - np.exp(-h1)) / (np.exp(h1) + np.exp(-h1))
    #y_act[j+1] = W @ v
    y_act[j+1] = mat_mul(W, v.reshape(n1+1, 1))

    e[j] = ( y_act[j]-yd[j] )**2 # Calculating the error

# Plot of actual test data vs Predicted test data
plt.figure(4) 
plt.plot(yd)
plt.grid()
plt.plot(y_act, '-r')

# RMS Error
rms_err = np.sqrt(e.sum()/iter_t)
print(rms_err)

# Show all the figures that were plotted
plt.show()