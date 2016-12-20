##################################
###### Writen by Boyuan Pan ######
##################################


import scipy.io as sio
import numpy as np
import random
import scipy as sp


def pp():
    print 'haha'
    return 'haha'



def load_data(dataset,seed):
    if dataset == 'ohsumed' or dataset == 'r83' or dataset == '20ng2' or dataset == '20ng2_500':
        data = 'dataset/' + dataset + '_tr_te.mat'
        data = sio.loadmat(data)
    else:
        data = 'dataset/' + dataset + '_tr_te_split.mat'
        data = sio.loadmat(data)
        xtr = data['X'][0][data['TR'][seed,:]-1]
        xte = data['X'][0][data['TE'][seed,:]-1]
        BOW_xtr = data['BOW_X'][0][data['TR'][seed,:]-1]
        BOW_xte = data['BOW_X'][0][data['TE'][seed,:]-1]
        indices_tr = data['indices'][0][data['TR'][seed,:]-1]
        indices_te = data['indices'][0][data['TE'][seed,:]-1]
        ytr = data['Y'][0][data['TR'][seed,:]-1]
        yte = data['Y'][0][data['TE'][seed,:]-1]
        return xtr,xte,ytr,yte, BOW_xtr,BOW_xte, indices_tr, indices_te

def minclass(y,ind):
    un = np.unique(y)
    m = float('inf')
    for i in range(0,len(un)):
        m = min(sum(y[ind] == un[i]),m)
    return m



def makesplits(y,split,splits,classsplit=0,k=1):
    # SPLITS "y" into "splits" sets with a "split" ratio.
    # if classsplit==1 then it takes a "split" fraction from each class
    
    if split == 1:
        train = np.array(range(len(y)))
        random.shuffle(train)
        test = []
        return train,test

    if split == 0:
        test = np.array(range(len(y)))
        random.shuffle(test)
        train = []
        return train,test

    n = len(y)
    if minclass(y,np.array(range(0,len(y)))) < k or split * len(y) / len(np.unique(y)) < k :
        print 'K:'+ k + ' split:' + split + ' n:' + len(y)
        print 'Cannot sub-sample splits! Reduce number of neighbors.'
        exit(1)


    if classsplit:
        un = np.unique(y)
        for i in range(0,splits):
            trsplit=[]
            tesplit=[]
            while minclass(y,trsplit)<k:
                for j in range(0,len(un)):
                    ii = np.where(y == un[j])
                    ii = np.array(ii[0])
                    co = int(round(split * np.size(ii)))
                    random.shuffle(ii)
                    trsplit = np.append(trsplit,ii[0:co])
                    tesplit = np.append(tesplit,ii[co:np.size(ii)])

                    trsplit = np.array(map(int,trsplit))
                    tesplit = np.array(map(int,tesplit))

            train = trsplit
            test = tesplit

    else:
        for i in range(0,splits):
            trsplit=[]
            tesplit=[]
            while minclass(y,trsplit)<k:
                ii = np.array(range(n))
                random.shuffle(ii)
                co = int(round(split*n))
                trsplit = ii[0:co]
                tesplit = ii[co:n]
            train = trsplit
            test = tesplit
 
    return train,test


def distance(X,x):
    D = np.size(X[:,0])
    N = np.size(X[0,:])
    d = np.size(x[:,0])
    n = np.size(x[0,:])
    if D!=d:
        print 'Both sets of vectors must have same dimensionality!'
        exit(1)   
    #dist = sp.spatial.distance.cdist
    dist = np.zeros([N,n])
    for i in range(0,N):
        for j in range(0,n):
            d = np.linalg.norm(X[:,i] - x[:,j])
            dist[i,j] = int(d**2)
    return dist

def grad_swmd(xtr, ytr, BOW_xtr, indices_tr, xtr_center, w, A, lambdA, batch, rangE):
    epsilon = 1e-8
    huge = 1e8

    dim = np.size(xtr[0],0); # dimension of word vector 
    ntr = len(ytr) # number of documents

    dw = np.zeros([np.size(w), 1])
    dA = np.zeros([dim,dim])

    # Sample documents
    sample_idx = random.sample(range(ntr),batch)

    Dc = distance(np.dot(A, xtr_center), np.dot(A, xtr_center))
    tr_loss = 0
    n_nan = 0

    for ii in range(0,batch):
        i = sample_idx[ii]
        xi = xtr[i]
        yi = ytr[i]
        idx_i = indices_tr[i]
        idx_i.shape = idx_i.size
        bow_i = BOW_xtr[i].T
        a = bow_i * w[idx_i]
        a = a / sum(a)

        nn_set = np.argsort(Dc[:,i]) #sort by the order of the distance to the 'i' document

        # Compute WMD from xi to the rest documents
        nn_set = nn_set[1:rangE+1]
        dd_dA_all = dict()
        alpha_all = dict()
        beta_all = dict()
        Di = np.zeros([rangE,1])

        xtr_nn = xtr[nn_set]
        ytr_nn = ytr[nn_set]
        BOW_xtr_nn = BOW_xtr[nn_set]
        indices_tr_nn = indices_tr[nn_set]

        for j in range(0,rangE):
            xj = xtr_nn[j]
            yj = ytr_nn[j]
            M = distance(np.dot(A,xi), np.dot(A,xj))
            idx_j = indices_tr_nn[j]
            bow_j = BOW_xtr_nn[j].T
            b = bow_j * w[idx_j]
            b = b / sum(b)
            [alpha, beta, T, dprimal] = sinkhorn(M, a, b, lambdA, 200, 1e-3) 
            Di[j] = dprimal

            alpha_all[j] = alpha
            beta_all[j] = beta

            # Gradient for metric
            sumA = np.dot(xi*a.T, xi.T) + np.dot(xj*b.T, xj.T) - np.dot(np.dot(xi, T), xj.T) - np.dot(np.dot(xj, T.T), xi.T)
            dd_dA_all[j] = sumA

        # Compute NCA probabilities
        Di[Di < 0] = 0
        dmin = min(Di)
        Pi = np.exp(-Di+dmin) + epsilon
        Pi[ytr_nn == i] = 0
        Pi = Pi/sum(Pi)
        pa = sum(Pi[ytr_nn==yi]) + epsilon # to avoid division by 0

        # Compute gradient wrt w and A
        dw_ii = np.zeros(np.size(w))
        dA_ii = np.zeros([dim,dim])
        for j in range(0,rangE):
            eq = ytr_nn[j] == yi
            eq = eq * 1
            cij = Pi[j]/pa * eq - Pi[j]
            idx_j = indices_tr_nn[j]
            idx_j.shape = idx_j.size
            bow_j = BOW_xtr_nn[j].T
            b = bow_j * w[idx_j]
            b = b / sum(b)
            a_sum = sum(w[idx_i] * bow_i)
            b_sum = sum(w[idx_j] * bow_j)
            dwmd_dwi = bow_i * alpha_all[j] /a_sum - bow_i * (np.dot(alpha_all[j].T, a)/ a_sum)
            dwmd_dwj = bow_j * beta_all[j] /b_sum - bow_j * (np.dot(beta_all[j].T, b)/ b_sum)
            dw_ii[idx_i] = dw_ii[idx_i] + cij*dwmd_dwi 
            dw_ii[idx_j] = dw_ii[idx_j] + cij*dwmd_dwi 
            dA_ii = dA_ii + cij*dd_dA_all[j]

        if sum(np.isnan(dw_ii)) == 0 and sum(sum(np.isnan(dA_ii))) == 0:
            dw = dw + dw_ii
            dA = dA + dA_ii
            tr_loss = tr_loss - np.log(pa)
        else:
            n_nan = n_nan + 1

    batch = batch - n_nan
    if n_nan > 0:
        print 'number of bad samples: ' + str(n_nan)

    tr_loss = tr_loss / batch
    dA = np.dot(A, dA)
    dw = dw / batch
    dA = dA / batch
    return dw, dA

def sinkhorn(M, a, b, lambdA, max_iter, tol):
    epsilon = 1e-10

    l = len(a)
    K = np.exp(-lambdA * M)
    Kt = K/a
    u = np.ones([l,1], dtype = np.int) /l
    iteR = 0
    change = np.inf
    b.shape = (np.size(b),1)
    
    while change > tol and iteR <= max_iter:
        iteR = iteR + 1
        u0 = u
        u = 1.0/(np.dot(Kt,(b/(np.dot(K.T,u)))))
        change = np.linalg.norm(u - u0) / np.linalg.norm(u)

    if min(u) <= 0:
        u = u - min(u) + epsilon

    v = b/(np.dot(K.T,u))

    if min(v) <= 0:
        v = v - min(v) + epsilon

    alpha = np.log(u)
    alpha = 1.0/lambdA * (alpha - np.mean(alpha))
    beta = np.log(v)
    beta = 1.0/lambdA * (beta - np.mean(beta))
    v.shape = (np.size(v),)
    T = v * (K * u)
    obj_primal = sum(sum(T*M))
  #  obj_dual = a * alpha + b * beta
    return alpha, beta, T, obj_primal

    




  















