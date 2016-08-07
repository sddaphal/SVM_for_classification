# -*- coding: utf-8 -*-
"""
Created on Sun Aug  7 10:35:33 2016

@author: dell

"""

import random
#define some vector operations
def dot(x, y):
    'dot-multiple of two vectors'
    return sum([a*b for a, b in zip(x,y)])
    
def multi(c, y):
    'Constant multiplies vector'
    return [c*a for a in y]
    
def subtract(x, y):
    'subtract of two vectors x-y'
    return [a-b for a, b in zip(x,y)]    
#   
def predict(w, b, rho, X, Y):
    'get training set predicted vector pred_Y and total loss e'
    #X is matrix, y is a vector with 0 or 1
    #TODO yz
    pass

def subgradient(w, b, rho, x, y):
    'get subgradient of loss func = max(0,1-yi*(w^T*xi+b))+\rho*(w^T*w)'
    #Note SGD only choose one sample (x,y) to get subgradient
    #x is a vector, y is value 0 or 1
    if 1-y*(dot(w, x)+b)>0:
        g = multi(-y,x)+2*multi(rho,w)
        g.append(-y)
    else:
        g = 2*multi(rho,w)
        g.append(0)
    return g
    
def SSGD(w0, b0, rho, lr, N, e, X, Y):
    'get w, b, if_converge, iteration'
    #lr: learning rate    
    #N: maximum number of iterations
    #e: threshold to check convergence
    #TODO output learning curve
    w = w0
    b = b0
    for i in range(N):
        #randomly choose one sample
        k = random.randint(0, len(Y)-1)
        x = X[k]
        y = Y[k]
        #get subgradient
        g = subgradient(w, b, rho, x, y)
        #update w, b
        w_old = w
        b_old = b
        w = subtract(w, multi(lr, g[:-1]))
        b = b-lr*g[-1]
        #get l2-norm of (w, b)'s difference
        delta = subtract(w, w_old) + [b-b_old]
        norm_delta = sum([x*x for x in delta])
        if norm_delta<e:   #less than threshold
            print('Converge after {} iterations with threshold {}'.format(i+1, e))
            return w, b, True, i
    print('Fail to converge in {} iterations with threshold {}'.format(i+1, e))
    return w, b, False, i
    
#test
#data generation: 100 samples with 2 features
X = [[random.uniform(0,3), random.uniform(0,3)] for i in range(100)]
true_w = [1,2]
true_b = -4.5
Y = []
for x in X:
    if dot(true_w, x)+true_b>0:
        Y.append(1)
    else:
        Y.append(-1)
#Train
w0 = [0, 0]
b0 = 0
rho = 0.1
lr = 0.01    
N = 10000000
e = 0.0000000001
w, b, if_converge, iteration = SSGD(w0, b0, rho, lr, N, e, X, Y)

#visulization
