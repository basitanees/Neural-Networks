# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 20:41:41 2019

@author: basit
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import pylab as py

question = '3'

def AbdulBasit_Anees_21600659_hw1(question):
    if question == '2' :
        print(question)
        acc, acc_b, acc_c = q2d()
        return {'Accuracy part b': acc, 'Accuracy Non Robust': acc_b, 'Accuracy Robust': acc_c}
    elif question == '3' :
        print(question)
        corr, w, b, acc_tr, acc_test, mse_test = q3()
        return {'Correlation': corr, 'w': w, 'b': b, 'Train Accuracy': acc_tr, 'Test Accuracy': acc_test, 'Test MSE': mse_test}

def q2d():
    def truth_table(N):
        X = np.zeros((2 ** N, N))
        for j in range(N):
            nIterate = 2 ** (N - (j + 1))
            step = 2 ** j
            prev = step
            for i in range(nIterate):
                X[prev:prev + step, N - (j + 1)] = np.ones((step))
                prev = prev + 2 * step
        return X
    
    def NNb(w1, w2):
        X = truth_table(4)
        Y = np.logical_xor(X[:,0] + (X[:,1] == 0), ((X[:,2] == 0) + (X[:,3] == 0)))
        X = np.hstack((X,-1*np.ones((16,1))))
        o = (np.hstack(( (X @ w1.T >= 0), -1*np.ones((16,1)))) @ w2) >= 0
        trues = np.sum(o == Y)
        acc = 100 * trues / 16
        return acc
    
    def NNd(w1, w2, sd):
        X = truth_table(4)
        Y = np.logical_xor(X[:,0] + (X[:,1] == 0), ((X[:,2] == 0) + (X[:,3] == 0)))
        y = np.tile(Y,25)
        noise = np.random.normal(0,sd,(25,16,4))
        x = (X + noise).reshape((400,4))
        x = np.hstack((x,-1*np.ones((400,1))))
        o = (np.hstack(( (x @ w1.T >= 0), -1*np.ones((400,1)))) @ w2) >= 0
        trues = np.sum(o == y)
        acc = trues / 4
        return acc
    #Non robust boundary weights
    w1 = np.array(((-1,1,-1,0,1),(-1,1,0,-1,1),(0,-1,1,1,2),(1,0,1,1,3)))
    w2 = np.array((1,1,1,1,0.5))
    acc = NNb(w1, w2)
    sd = 0.2
    acc_b = NNd(w1, w2, sd)
    #Robust boundary weights
    w1 = np.array(((-1,1,-1,0,0.5),(-1,1,0,-1,0.5),(0,-1,1,1,1.5),(1,0,1,1,2.5)))
    w2 = np.array((1,1,1,1,0.5))
    acc_c = NNd(w1, w2, sd)
    return acc, acc_b, acc_c

def q3():
    #Load data
    data = scipy.io.loadmat("assign1_data1.mat")
    x_tr = data["trainims"].transpose((2,0,1))
    x_test = data["testims"].transpose((2,0,1))
    y_tr = data["trainlbls"]
    y_test = data["testlbls"]
    
    def make_grid(images):
        _, grid = plt.subplots(nrows=4,ncols=7)
        for p in range(28):
            if p < 26:
                grid[int(p/7),p%7].imshow(images[p%26])
            grid[int(p/7),p%7].axes.get_xaxis().set_visible(False)
            grid[int(p/7),p%7].axes.get_yaxis().set_visible(False)
    #Show letters from each class
    images = []
    for i in range(1,27):
        index = np.argwhere(y_tr==i)[0,0]
        plt.figure()
        images.append(x_tr[index,:,:])
        plt.imshow(x_tr[index,:,:])
    make_grid(images)

    def corr(a, b):
        product = np.mean((a - a.mean()) * (b - b.mean()))
        stds = a.std() * b.std()
        product = product / stds
        return product
    #Plot correlation
    correlation = np.zeros((26,26))
    for i in range(26):
        for j in range(26):
            correlation[i,j] = corr(images[i], images[j])
    plt.figure()
    plt.title('Correlation')
    plt.imshow(correlation)
    
    def to_categorical(y):
        out = np.zeros((len(y),26))
        for i in range(26):
            out[:,i] = ((y-1) == i).reshape(len(y))
        return out
    
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    
    def train(w, b, x_tr, y_trn, iterations, rate):
        W = np.zeros((iterations, 28 * 28, 26))
        n = 0
        mse = np.zeros((iterations, 1))
        while n < iterations:
            ind = np.random.randint(0, x_tr.shape[0])
            x   = x_tr[ind,:].reshape((28*28,1))
            y   = y_trn[ind,:].reshape(26,1)
            y_p = sigmoid((w.T @ x) + b)
            grad_w = x @ (- (y - y_p) * (y_p * (1 - y_p))).T
            grad_b = - (y - y_p) * (y_p * (1 - y_p))
            w   = w - rate * grad_w
            b = b - rate * grad_b
            W[n,:,:] = w
            mse[n, :] = (y - y_p).T@(y - y_p)/2
            n = n + 1
        return w, b, mse
    
    def accuracy(w, b, x_tr, y_tr, samples):
        y_pr = sigmoid((x_tr.reshape((samples,28*28)) @ w)+b.T)
        y_pr = (np.argmax(y_pr, 1).reshape((samples,1))) + 1
        trues = np.sum(y_pr == y_tr)
        acc = 100 * (trues / samples)
        return acc
    #Initialize variables
    x_tr = x_tr.reshape((5200, 28*28))
    x_tr = (x_tr/255)
    y_train = to_categorical(y_tr)
    iterations = 10000
    L = [0.5, 0.05, 0.005]
    labels = ["High","Good","Low"]
    acc_tr = {}
    errors = []
    weights = {}
    biases = {}
    w_0 = np.random.normal(0, 0.01, (28 * 28, 26))
    b = np.random.normal(0, 0.01, (26, 1))
    #Train
    for i in range(len(L)):
        w, b, mse = train(w_0, b, x_tr, y_train, iterations, L[i])
        acc = accuracy(w, b, x_tr, y_tr, 5200)
        acc_tr[labels[i]] = (acc)
        weights[labels[i]] = (np.copy(w))
        biases[labels[i]] = (np.copy(b))
        errors.append(mse)
    #Display weights
    w_disp = []
    for i in range(26):
        w_disp.append(weights['Good'][:, i].reshape((28,28)))
        plt.figure()
        plt.imshow(w_disp[i])
    make_grid(w_disp)
    #Plot errors
    plt.figure()
    for i in range(len(L)):
        plt.plot(np.arange(iterations),errors[i], label = labels[i] )
    py.legend(loc = 'upper right')
    plt.xlabel('Itrations')
    plt.ylabel('MSE')
    plt.title('MSE vs Iterations')
    plt.show()
    # Find test accuracy and loss
    x_test = x_test.reshape((1300, 28*28))
    x_test = (x_test/255)
    y_testt = to_categorical(y_test)
    w = weights
    b = biases
    acc_test = {}
    mse_test = {}
    for i in range(len(w)):
        acc_test[labels[i]] = (accuracy(w[labels[i]], b[labels[i]], x_test, y_test, 1300))
        y_pr = sigmoid((x_test.reshape((1300,28*28)) @ w[labels[i]])+ b[labels[i]].T)
        mse_test[labels[i]] = (np.sum(((y_testt - y_pr)**2))/(2*1300))
    plt.close('all')
    return correlation, w, b, acc_tr, acc_test, mse_test

output = AbdulBasit_Anees_21600659_hw1(question)
