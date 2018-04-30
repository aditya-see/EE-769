#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 22 16:41:19 2018

@author: aditya
"""
import numpy as np
import matplotlib.pyplot as plt


# Generating the training dataset


iter_t = 1000    # working with 1000 sample points
u = np.zeros([iter_t])
y = np.zeros([iter_t])
yd = np.zeros([iter_t])
y_act = np.zeros([iter_t])
x = np.zeros([iter_t, 2])

for i in range(0, iter_t-1):
    u[i] = 2 * np.random.rand()-1                    # random input between -1 and 1
    y [i+1] = ( y[i] / (1+y[i]**2) ) + u[i]**3       # y is the input for calculation of next state    
    yd[i] = y[i+1]

for i in range(0, iter_t-1):
    x[i, 0] = u[i]                                  # For Convinince inputs are stored in x
    x[i, 1] = y[i]

plt.figure(1)
plt.plot(u, yd, '.')
plt.show()


# Initializing the weights and other parametrs of the NN
n0 = 2          # No of inputs/input neurons 
n1 = 15         # no. of Neurons in the first layer
n2 = 1          # No. of Outputs

w = np.zeros([n1, n0+1])   # weights between hidden and input layer, hidden layers = 15
W = np.zeros([n1+1])       # weights between hidden and output layer

for i in range(0, n1):
    w[i, 0] = (2*np.random.rand()-1) # randomly initialize the weights
    w[i, 1] = (2*np.random.rand()-1) # randomly initialize the weights
    w[i, 2] = (2*np.random.rand()-1) # randomly initialize the weights

for i in range(0, n1+1):
    W[i] = (2*np.random.rand()-1)   # randomly initialize the weights

lr = 0.1    # learning rate
epoch = 200 # no. of epochs
p = 1       # p is the index of each epoch
  

# Training the NN


mse = 0     # mean square error for each iterartion is stored in this variable  
mse1 = np.zeros(epoch)
h = np.zeros(n1)
v = np.zeros(n1+1)
y2 = np.zeros(iter_t)
del1 = np.zeros(n1)

y2[0] = y[0]
while ( p < epoch ):

    for j in range(0, iter_t):   # j is the index of each pattern

        # Forward evaluation of Neural Network   
        for i in  range(0, n1):
            h[i] = w[i,0] * x[j,0] + w[i,1] * x[j,1] + w[i,2]
            v[i] = (np.exp(h[i]) - np.exp(-h[i])) / (np.exp(h[i]) + np.exp(-h[i]))
        v[n1]=1

        H = 0
        for i in range(0, n1+1):
            H += W[i]*v[i]

        y_act[j] = H

        mse = mse + 0.5 * ( yd[j] - y_act[j] ) * ( yd[j] - y_act[j] ) / iter_t    # update of weights

        # Calculation of dels and weight update by backpropogation

        del2 = ( yd[j] - y_act[j] )

        for i in range(0, n1+1):
            W[i] = W[i] + lr * del2 * v[i]

        for i in  range(0, n1):
            del1[i] = (1-v[i]**2) * del2 * W[i]

        for i in range(0, n1):
            w[i,0] +=  lr * del1[i] * x[j,0]
            w[i,1] +=  lr * del1[i] * x[j,1]
            w[i,2] +=  lr * del1[i]


    mse1[p] = mse
    mse = 0 
    p=p+1

plt.figure(2)
plt.plot(yd)
plt.plot(y_act,'r')


plt.figure(3)
plt.plot(mse1)

# plot actual test data

 
iter_t = 1000                          # No. of test data set
y[0] = 0                               # Initial condition            

for i in range(0, iter_t-1):
    u[i] = np.sin( 0.02 * i )                     # Test Input Data
    y[i+1] = ( y[i] / (1+y[i]**2) ) + u[i]**3    # y is the input for calculation of next state 
    yd[i] = y[i]                               # Output of System for ith instant inputs 

for i in range(0, iter_t):                             # Storing inputs in x for convinience
    x[i, 0] =  u[i]
    x[i, 1] =  y[i]



# Plot using Neural Network
e = np.zeros([iter_t])
y_act[0] = y[0]
for j in range(0, iter_t-1):
    for i in range(0, n1):
        h[i] = w[i,0] * x[j,0] + w[i,1] * y[j] + w[i,2]
        v[i] = ( np.exp(h[i]) - np.exp(-h[i]) ) / ( np.exp(h[i]) + np.exp(-h[i]) )
    v[n1] = 1
    
    H = 0
    for i in range(0, n1+1):
        H = H + W[i] * v[i]
     
    y_act[j+1] = H 

    e[j] = ( y_act[j]-yd[j] ) * ( y_act[j]-yd[j] )

plt.figure(4) 
plt.plot(yd)
plt.plot(y_act, 'r')

# RMS Error
 
e1 = 0
for i in range(0, iter_t):
    e1 = e1 + e[i]

rms_err = np.sqrt(e1/iter_t)
print(rms_err)

plt.show()
