##################################
###### Writen by Boyuan Pan ######
##################################

import sys
import os
import scipy.io as sio
import numpy as np
import random
import gc
import pyximport
#pyximport.install(reload_support=True)

pwd = os.getcwd()

RAND_SEED = 1

#pwd = pwd + '/functions'
#sys.path.append(pwd)

import functions as f

save_path = 'results/'

dataset = 'bbcsport'
MAX_DICT_SIZE = 50000

max_iter = 50
save_frequency = max_iter
batch = 32
rangE = 200
lr_w = 1e+1
lr_A = 1e+0
lambdA = 10

cv_folds = 5#5
results_cv = np.zeros(cv_folds)

for split in range(1,cv_folds+1):
    #err_v = np.zeros([5,19])
    #err_t = np.zeros([5,19])
    sv = 0

    save_couter = 0
    Err_v = []
    Err_t = []
    w_all = []
    A_all = []
    [xtr,xte,ytr,yte, BOW_xtr,BOW_xte, indices_tr, indices_te] = f.load_data(dataset, split-1)
    [idx_tr, idx_val] = f.makesplits(ytr, 1-1.0/cv_folds, 1, 1)
    
    
    xtro = xtr
    ytro = ytr
    BOW_xtro = BOW_xtr
    indices_tro = indices_tr
    
    xv = xtr[idx_val]
    yv = ytr[idx_val]
    BOW_xv = BOW_xtr[idx_val]
    indices_v = indices_tr[idx_val]
    
    xtr = xtr[idx_tr]
    ytr = ytr[idx_tr]
    BOW_xtr = BOW_xtr[idx_tr]
    indices_tr = indices_tr[idx_tr]

    ntr = len(ytr)
    nv = len(yv)
    nte = len(yte)
    dim = np.size(xtr[0],0); # dimension of word vector


    ########## Compute document center
    xtr_center = np.zeros([dim, ntr],dtype = np.int)
    for i in range(0,ntr):
    	rc= np.dot(xtr[i], BOW_xtr[i].T )/ sum(sum(BOW_xtr[i]))
    	rc.shape = rc.size
    	xtr_center[:,i] = rc
    xv_center = np.zeros([dim, nv],dtype = np.int)
    for i in range(0,nv):
    	vc = np.dot(xv[i], BOW_xv[i].T)/ sum(sum(BOW_xv[i]))
    	vc.shape = vc.size
    	xv_center[:,i] = vc
    xte_center = np.zeros([dim, nte],dtype = np.int)
    for i in range(0,nte):
    	ec = np.dot(xte[i], BOW_xte[i].T) / sum(sum(BOW_xte[i]))
    	ec.shape = ec.size
    	xte_center[:,i] = ec


    ########### Load initialize A (train with WCD)
    dataA = 'metric_init/' + dataset + '_seed' + str(split) + '.mat'
    bbc_ini = sio.loadmat(dataA)
    A = bbc_ini['Ascaled']


    ########### Define optimization parameters
    w = np.ones([MAX_DICT_SIZE,1])

    ########### Test learned metric for WCD   TO BE CONTINUED!!
   # Dc = f.distance(xtr_center, xte_center)


    ########### Main loop
    for iter in range(1,max_iter+1):
        print 'Dataset: ' + dataset + ' split: ' + str(split) + ' Iteration: ' + str(iter)
        [dw, dA] = f.grad_swmd(xtr,ytr,BOW_xtr,indices_tr,xtr_center,w,A,lambdA,batch,rangE)


      #  raw_input(np.size(dw))
     #   raw_input(np.size(w))

        # Update w and A
        w = w - lr_w * dw
        lower_bound = 0.01
        upper_bound = 10
        w[w<lower_bound] = lower_bound
        w[w>upper_bound] = upper_bound
        A = A - lr_A * dA

        

        if iter == save_frequency: #iter == 1 or iter == 3 or iter == 10 or iter == 50 or iter == 200:
            ########### Compute loss
            filename = save_path + dataset + '_' + str(lambdA) + '_' + str(int(lr_w)) + '_' + str(int(lr_A)) + '_' + str(max_iter) + '_' + str(batch) + '_' + str(rangE) + '_' + str(split) + '.mat'
            err_v = f.knn_swmd(xtr, ytr, xv, yv, BOW_xtr, BOW_xv, indices_tr, indices_v, w, lambdA, A)    
            err_t = f.knn_swmd(xtro, ytro, xte, yte, BOW_xtro, BOW_xte, indices_tro, indices_te, w, lambdA, A)
            sv += 1
            sio.savemat(filename, {'err_v':err_v, 'err_t':err_t, 'w':w, 'A':A})

            
        del dw, dA
        gc.collect()
        
    err_t_cv = err_t[err_v == np.min(err_v)]
    results_cv[split-1] = err_t_cv[0]
    sio.savemat(save_path + dataset + '_results', {'results_cv':results_cv})
    
    


    
















