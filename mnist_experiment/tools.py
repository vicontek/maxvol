import maxvolpy
import numpy as np
from mnist import MNIST
from matplotlib import pyplot as plt
import random
import maxvolpy.maxvol as mv
from numpy.linalg import svd
from scipy.sparse.linalg import svds
from scipy import sparse
from sklearn.decomposition import PCA
import pandas as pd
from sklearn.linear_model import Perceptron
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


def get_mnist(path):
    mndata = MNIST(path)

    train_images, train_labels = mndata.load_training()
    test_images, test_labels = mndata.load_testing()

    X_train = mndata.process_images_to_numpy(train_images).astype(np.float32)
    Y_train = np.array(train_labels)

    X_test = mndata.process_images_to_numpy(test_images).astype(np.float32)
    Y_test = np.array(test_labels)

    return X_train, Y_train, X_test, Y_test


def group_by_class(df, k, n_samples):
    print('class:', df.iloc[0, -1])
    X = df[df.columns[:-1]].as_matrix()
    sparse_X = sparse.csr_matrix(df[df.columns[:-1]].as_matrix())
    U, s, V = svds(sparse_X, k=k)

    maxvol_ind = mv.rect_maxvol(U, minK=n_samples, maxK=n_samples)[0]
    # X = U[maxvol_ind] @ np.diag(s) @ V
    # print(X.shape)
    # d = pd.DataFrame(X)
    # print(maxvol_ind.shape)
    d = df.iloc[maxvol_ind]
    print(d.shape)
    # print(d.shape)
    # print(df.iloc[maxvol_ind]['label'].as_matrix().shape)
    # d['label'] = df.iloc[maxvol_ind]['label'].as_matrix()# = df.iloc[maxvol_ind, 'labels']
    return d

def svd_mavol_sampling(data, labels, k, n_samples):
    print('Sampling dataset...')
    print('data shape:', data.shape, '\nlabels shape:', labels.shape)
    df = pd.DataFrame(data)
    df['label'] = labels
    sampled = df.groupby('label').apply(lambda x: group_by_class(x, k=k, n_samples=n_samples)).as_matrix()
    return sampled[:, :-1], sampled[:, -1]
    
# df = class_sets(X_train, Y_train)



def feature_selection(X_train, Y_train, X_test, Y_test, k):
    X_all = np.vstack((X_train, X_test)) 
    Y_all = np.concatenate((Y_train, Y_test))
    sparse_X_all = sparse.csr_matrix(X_train)
    print(X_train.shape)
    U, s, V = svds(sparse_X_all, k=k)
    maxvol_ind_all = mv.rect_maxvol(V.T, minK=k, maxK=k)[0]
    V = V.T[maxvol_ind_all].T
    print('Vshape', V.shape)
    # V = V[:, :100]
    print(V.shape)
    # X = U @ np.diag(s) @ V
    # X.shape
    return X_all[:X_train.shape[0], maxvol_ind_all], Y_train, X_all[X_train.shape[0]:, maxvol_ind_all], Y_test
    

def random_sampling(data, labels, n_samples):
    idx = np.random.randint(data.shape[0], size=n_samples)
    return data[idx, :], labels[idx]
    

def random_features(X_train, Y_train, X_test, Y_test, n_features):
    idx = np.random.randint(X_train.shape[1], size=n_features)
    return X_train[:, idx], Y_train, X_test[:, idx], Y_test

