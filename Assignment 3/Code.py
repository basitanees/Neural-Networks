# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 21:56:22 2019

@author: basit
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy.io
import pylab as py
import h5py
import random

question = '1'

def AbdulBasit_Anees_21600659_hw1(question):
    if question == '1' :
        print(question)
        return Q2()

def Q2():
    print("Running Q2a")
    data_norm = Q2a()
    print('Running Q2 b and c')
    Q2bc(data_norm)
    print("Running Q2d")
    Q2d(data_norm)
    
def Q2d(data_norm):
    x_tr1 = data_norm.reshape((data_norm.shape[0], 16 * 16))
    
    #Shuffle data
    arr   = np.arange(x_tr1.shape[0])
    np.random.shuffle(arr)
    #x_tr  = x_tr[arr]
    x_tr1 = x_tr1[arr]
    
    #Initialize network parameters
    lambd      = 0.001 #0:0.001
    beta       = 0.008
    rho        = 0.05
    epochs     = 1000
    batch_size = 128
    rate       = 0.008
    nBatches   = int(np.ceil(x_tr1.shape[0]/batch_size))
    J_train    = np.zeros((epochs, 1))
    
    mse_tr  = {}
    weights = {}
    N = [25, 64,100]
    nrows = [5, 8, 10]
    for L_hid in N:
        w0         = np.sqrt(6/(L_hid+256))
        w1         = np.random.uniform(-w0, w0, (256, L_hid))
        w2         = np.random.uniform(-w0, w0, (L_hid, 256))
        b1         = np.random.uniform(-w0, w0, (1, L_hid))
        b2         = np.random.uniform(-w0, w0, (1, 256))
        for nEpoch in range(epochs):
            w1, w2, b1, b2    = train (w1, w2, b1 , b2, x_tr1, x_tr1, batch_size, rate, nBatches, rho, lambd, beta)
            yp_tr             = forward(w1, w2, b1 , b2, x_tr1)
            mse_cost = (0.5 / x_tr1.shape[0]) * np.sum((yp_tr-x_tr1)**2)
            tykhonov = (lambd / 2) * (np.sum(w1**2) + np.sum(w2**2))
            rhos = (1 / x_tr1.shape[0]) * np.sum(sigmoid(x_tr1 @ w1), axis = 0)
            kl_diver = beta * KL_div(rho, rhos)
            J_train[nEpoch]   = mse_cost + tykhonov + kl_diver
            if nEpoch%50 == 0:
                print(nEpoch)
            if nEpoch%100 == 0:
                print(J_train[nEpoch])
        
        mse_tr[str(L_hid)]  = np.copy(J_train)
        weights[str(L_hid)] = {'w1': np.copy(w1), 'w2': np.copy(w2), 'b1': np.copy(b1), 'b2': np.copy(b2)}

    for i in range(len(N)):
        weights1 = ((weights[str(N[i])])['w1']).reshape((16,16,N[i]))
        make_grid1(weights1, nrows[i], nrows[i])
    
    make_grid1(data_norm[:64].transpose((1,2,0)), 8, 8)
    
    wn1 = weights[str(N[0])]
    wn2 = weights[str(N[1])]
    wn3 = weights[str(N[2])]
    wn = [wn1, wn2, wn3]
    #wn = [wn1]
    
    for i in range(len(N)):
        out = forward((wn[i])['w1'], (wn[i])['w2'], (wn[i])['b1'] , (wn[i])['b2'], data_norm[:64].reshape((64,256)))
        make_grid1(out.reshape(64,16,16).transpose((1,2,0)), 8, 8)
    
    for i in range(len(N)):
        loss = mse_tr[str(N[i])]
        plt.figure()
        plt.plot(np.arange(len(loss)), loss)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Loss vs epochs')

def Q2bc(data_norm):
    x_tr1 = data_norm.reshape((data_norm.shape[0], 16 * 16))
    
    #Shuffle data
    arr   = np.arange(x_tr1.shape[0])
    np.random.shuffle(arr)
    x_tr1 = x_tr1[arr]
    
    #Initialize network parameters
    L_hid      = 64 #10:100
    lambd      = 0.0005 #0:0.001
    beta       = 0.0002
    rho        = 0.008
    epochs     = 1000
    batch_size = 128
    rate       = 0.01
    w0         = np.sqrt(6/(L_hid+256))
    nBatches   = int(np.ceil(x_tr1.shape[0]/batch_size))
    w1         = np.random.uniform(-w0, w0, (256, L_hid))
    w2         = np.random.uniform(-w0, w0, (L_hid, 256))
    b1         = np.random.uniform(-w0, w0, (1, L_hid))
    b2         = np.random.uniform(-w0, w0, (1, 256))
    J_train    = np.zeros((epochs, 1))

#    mse_cost = (0.5 / x_tr1.shape[0]) * np.sum((forward(w1, w2, b1 , b2, x_tr1)-x_tr1)**2)
#    tykhonov = (lambd / 2) * (np.sum(w1**2) + np.sum(w2**2))
#    rhos = (1 / x_tr1.shape[0]) * np.sum(sigmoid(x_tr1 @ w1), axis = 0)
#    kl_diver = beta * KL_div(rho, rhos)
#    cost = mse_cost + tykhonov + kl_diver

    for nEpoch in range(epochs):
        w1, w2, b1, b2    = train (w1, w2, b1 , b2, x_tr1, x_tr1, batch_size, rate, nBatches, rho, lambd, beta)
        yp_tr             = forward(w1, w2, b1 , b2, x_tr1)
        mse_cost = (0.5 / x_tr1.shape[0]) * np.sum((yp_tr-x_tr1)**2)
        tykhonov = (lambd / 2) * (np.sum(w1**2) + np.sum(w2**2))
        rhos = (1 / x_tr1.shape[0]) * np.sum(sigmoid(x_tr1 @ w1), axis = 0)
        kl_diver = beta * KL_div(rho, rhos)
        J_train[nEpoch]   = mse_cost + tykhonov + kl_diver
        if nEpoch%50 == 0:
            print(nEpoch)
        if nEpoch%25 == 0:
            print(J_train[nEpoch])
    
    weights1 = w1.reshape((16,16,64))
    
    def make_grid1(weights, rows, col):
    #    col = 8
    #    rows = 8
        _, grid = plt.subplots(nrows=rows,ncols=col)
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace= None, hspace=0.08)
        for p in range(rows*col):
            grid[int(p/col),p%col].imshow(weights[:,:,p])     #, cmap = 'gray'
            grid[int(p/col),p%col].axes.get_xaxis().set_visible(False)
            grid[int(p/col),p%col].axes.get_yaxis().set_visible(False)
    
    make_grid1(weights1, 8, 8)

    plt.figure()
    make_grid1(data_norm[:64].transpose((1,2,0)), 8, 8)
    out = forward(w1, w2, b1 , b2, data_norm[:64].reshape((64,256)))
    plt.figure()
    make_grid1(out.reshape(64,16,16).transpose((1,2,0)), 8, 8)
    
    plt.figure()
    plt.plot(np.arange(len(J_train)), J_train)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss vs epochs')

def Q2a():
    #Load data
    with h5py.File("C:/4th Year/EEE443/HW3/assign3_data1.h5", 'r') as f:
        keys = list(f.keys())
        data = f[keys[0]].value
#        invXForm = f[keys[1]].value
#        xForm = f[keys[2]].value
        f.close()
        
    # Y = 0.2126* R + 0.7152*G + 0.0722*B
    scale = np.array((0.2126, 0.7152, 0.0722)).reshape((3,1,1))
    
    data_norm = np.zeros((len(data),16,16))
    for i in range(len(data)):
        iData = np.sum(data[i]*scale, axis = 0)
        iData -= iData.mean()
        data_norm[i] = iData
    
    clip = 3 * data_norm.std()
    data_norm[(data_norm > clip)] = clip
    data_norm[(data_norm < -clip)] = -clip
    data_norm = ((data_norm + clip) * (0.8 / (2 * clip))) + 0.1
    
    def make_grid(images):
        col = 17
        rows = 12
        for i in range(1):
            _, grid = plt.subplots(nrows=rows,ncols=col)
            plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace= None, hspace=0.08)
            plt.tight_layout()
            for p in range(rows*col):
                if(len(images.shape) == 4):
                    image = images[p+(i*col*rows)]
                    image = image - image.min()
                    image = (image / image.max())
                    grid[int(p/col),p%col].imshow(image.transpose((1,2,0)).reshape((16,16,3)))
                    save = "fig"+ str(i)
                else:
                    grid[int(p/col),p%col].imshow(images[p+(i*col*rows)])
                    save = "fig"+ str(i+1)
                grid[int(p/col),p%col].axes.get_xaxis().set_visible(False)
                grid[int(p/col),p%col].axes.get_yaxis().set_visible(False)
            plt.savefig(save + ".png", dpi = 600)
    
    index = np.arange(10240)
    random.shuffle(index)
    make_grid(data[index])
    make_grid(data_norm[index])
    return data_norm

#Plotting functions
def make_grid1(weights, rows, col):
#    col = 8
#    rows = 8
    plt.figure()
    _, grid = plt.subplots(nrows=rows,ncols=col)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace= None, hspace=0.08)
    for p in range(rows*col):
        grid[int(p/col),p%col].imshow(weights[:,:,p])     #, cmap = 'gray'
        grid[int(p/col),p%col].axes.get_xaxis().set_visible(False)
        grid[int(p/col),p%col].axes.get_yaxis().set_visible(False)
        
#Functions
def tanh(x):
    return (np.exp(2*x) - 1) / (np.exp(2*x) + 1)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoidGradient(x):
    return sigmoid(x)*(1-sigmoid(x))

def Relu(x):
    return x*(x>0)

def ReluGradient(x):
    return x>0

def forward(w_1, w_2, b_1, b_2, x_tr):
    o_1  = sigmoid((x_tr @ w_1) + b_1)
    y_p  = sigmoid((o_1 @ w_2) + b_2)
    return y_p
    
# calculate the kl divergence
def KL_div(rho, q):
    kl = rho*np.log(rho/q) + (1-rho)*np.log((1-rho)/(1-q))
    kl[np.isnan(kl)] = 0
    return ((1 / kl.shape[0]) * np.sum(kl))

def KL_der(rho, q):
    return ((1-rho)/(1-q)) - (rho/q)

#Training including forward and backward pass for one epoch
def train(w_1, w_2, b_1, b_2, x, y, batch_size, rate, nBatches, rho, lambd, beta):
    for i in range(nBatches):
        #Load batch
        batch_x = x[i*batch_size:(i+1)*batch_size]
        batch_y = y[i*batch_size:(i+1)*batch_size]
        #Forward pass
        o_1  = sigmoid((batch_x @ w_1) + b_1)
#        o_11 = np.hstack((np.ones((o_1.shape[0],1)), o_1))
        y_p  = sigmoid((o_1 @ w_2) + b_2)
        #Backward pass
        delta_2 = - (1 / batch_size) * (batch_y - y_p) * sigmoidGradient(y_p)
        grad_w2 = o_1.T @ delta_2 + (lambd * w_2)
        grad_b2 = np.sum(delta_2, axis = 0) #/ delta_2.shape[0]
        
        rhos    = (1 / o_1.shape[0]) * np.sum(o_1, axis = 0)
        rho_der = KL_der(rho, rhos).reshape((1,rhos.shape[0]))
        rho_der_w1 = (1/batch_x.shape[0]) * (batch_x.T @ sigmoidGradient(o_1))
        rho_der_w  = rho_der_w1 * rho_der
        rho_der_b1 = (1/batch_x.shape[0]) * (np.sum(sigmoidGradient(o_1), axis = 0))
        rho_der_b  = rho_der_b1 * rho_der
        
        delta_1 = (delta_2 @ w_2.T) * sigmoidGradient(o_1)
        grad_w1 = batch_x.T @ delta_1 + (lambd * w_1) + (beta * rho_der_w)
        grad_b1 = (np.sum(delta_1, axis = 0) ) + (beta * rho_der_b)
        #Gradient descent
        w_1 = w_1 - rate * grad_w1
        w_2 = w_2 - rate * grad_w2
        b_1 = b_1 - rate * grad_b1
        b_2 = b_2 - rate * grad_b2
    return w_1, w_2 , b_1, b_2

output = AbdulBasit_Anees_21600659_hw1(question)