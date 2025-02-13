import numpy as np
import sympy as sy
import scipy as sp


### Collection of functions to convert between different representations of polynomials in matlab and scipy. Very experimental.

def expand_multiindex(M, P):
    nt = np.shape(P)[1] # number of points to evaluate
    C = M['coeffs']
    I = M['ind']
    n_I = np.shape(I)[0]
    P_T = np.tile(P, (nt, 1)).T # equivalent to repmat in matlab
    P_kron = np.kron(P_T, np.ones((n_I, 1))) # equivalent to kron in matlab
    I_repeat = np.tile(I, (nt, 1)) # equivalent to repmat in matlab
    P_I = np.prod(P_kron ** (I_repeat), axis=1)
    # evaluate polynomials
    return np.dot(C, P_I.reshape(-1, nt))


def reduced_to_full(p, W0):
    N = np.shape(W0[0]['coeffs'])[0]
    z = np.zeros((N,1))*p[0]
    for i in range(len(W0)):
        if len(W0[i]['coeffs'])>0:
            if W0[i]['coeffs'].shape[1]>0:
                z += expand_multiindex(W0[i], p)
    return z

def reduced_to_full2(p, W0):
    N = np.shape(W0[0]['coeffs'])[0]
    z = np.zeros((N,1))*p[0]
    for i in range(100):
        if len(W0[i]['coeffs'])>0:
            z += expand_multiindex(W0[i], p)
    return z



def reduced_to_full_new(p, W0):
    N = np.shape(W0[0]['coeffs'])[0]
    z = np.zeros((N,1))*p[0]
    for i in range(len(W0)):
        if W0[i]['coeffs'].shape[1]>0:
            z += expand_multiindex(W0[i], p)
    return z


def extract_w0(matt):
    mat_array = []
    matt = matt['W_0'][0]
    order = len(matt)
    for i in range(order):
        dictt = {}
        if matt[i][0][0]['coeffs'].shape[0]>0:
            dictt['coeffs'] = matt[i][0][0]['coeffs'].toarray()
            dictt['ind'] = np.array(matt[i][0][0]['ind'].toarray(), dtype= int)
        else:
            dictt['coeffs'] = []
            dictt['ind'] = []
        mat_array.append(dictt)
    return mat_array

def extract_w02(matt):
    mat_array = []
    matt = matt['W_0'][0]
    order = len(matt)
    for i in range(order):
        dictt = {}
        if matt[i][0][0]['coeffs'].shape[0]>0:
            dictt['coeffs'] = matt[i][0][0]['coeffs']
            dictt['ind'] = np.array(matt[i][0][0]['ind'].toarray(), dtype= int)
        else:
            dictt['coeffs'] = []
            dictt['ind'] = []
        mat_array.append(dictt)
    return mat_array



def extract_w0_new(matt):
    mat_array = []
    matt = matt['W_0'][0]
    order = len(matt)
    for i in range(order):
        dictt = {}
        if matt['coeffs'][i].shape[0]>0:
            dictt['coeffs'] = matt['coeffs'][i]
            dictt['ind'] = np.array(matt['ind'][i].toarray(), dtype= int)
        else:
            dictt['coeffs'] = []
            dictt['ind'] = []
        mat_array.append(dictt)
    return mat_array


def extract_general(matt, label):
    mat_array = []
    matt = matt[label][0]
    order = len(matt)
    for i in range(order):
        dictt = {}
        if matt['coeffs'][i].shape[0]>0:
            dictt['coeffs'] = matt['coeffs'][i]
            dictt['ind'] = np.array(matt['ind'][i].toarray(), dtype= int)
        else:
            dictt['coeffs'] = []
            dictt['ind'] = []
        mat_array.append(dictt)
    return mat_array

def extract_r0(matt):
    mat_array = []
    matt = matt['R_0'][0]
    order = len(matt)
    for i in range(order):
        dictt = {}
        if matt[i][0][0]['coeffs'].shape[0]>0:
            dictt['coeffs'] = matt[i][0][0]['coeffs'].toarray()
            dictt['ind'] = np.array(matt[i][0][0]['ind'].toarray(), dtype= int)
        else:
            dictt['coeffs'] = []
            dictt['ind'] = []
        mat_array.append(dictt)
    return mat_array



def extract_r0_new(matt):
    mat_array = []
    matt = matt['R_0'][0]
    order = len(matt)
    for i in range(order):
        dictt = {}
        if matt['coeffs'][i].shape[0]>0:
            if isinstance(matt['coeffs'][i], sp.sparse.spmatrix):
                dictt['coeffs'] = matt['coeffs'][i].toarray()
            else:
                dictt['coeffs'] = matt['coeffs'][i]
            dictt['ind'] = np.array(matt['ind'][i].toarray(), dtype= int)
        else:
            dictt['coeffs'] = []
            dictt['ind'] = []
        mat_array.append(dictt)
    return mat_array

def extract_gen_new(matt, label):
    mat_array = []
    matt = matt[label][0]
    order = len(matt)
    for i in range(order):
        dictt = {}
        if matt['coeffs'][i].shape[0]>0:
            if isinstance(matt['coeffs'][i], sp.sparse.spmatrix):
                dictt['coeffs'] = matt['coeffs'][i].toarray()
            else:
                dictt['coeffs'] = matt['coeffs'][i]
            dictt['ind'] = np.array(matt['ind'][i].toarray(), dtype= int)
        else:
            dictt['coeffs'] = []
            dictt['ind'] = []
        mat_array.append(dictt)
    return mat_array



def extract_coefficients(expression, variables, size):
    taylor_coeffs = np.zeros((size+1,size+1)) * 1j
    p1 = variables[0]
    p2 = variables[1]
    for ii in range(0, size+1):
        for jj in range(0, size+1):
            taylor_coeffs[ii,jj] = expression.as_expr().coeff(p1,ii).coeff(p2, jj)
    return taylor_coeffs


def extract_coefficients_1d(expression, variable, size):
    coeffs = sy.Poly(expression, variable).all_coeffs()[::-1]
    coeffs = np.array(np.real([sy.re(c) for c in coeffs]), dtype=float)
    return coeffs