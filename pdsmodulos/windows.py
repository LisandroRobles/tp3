#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  3 14:39:22 2018

@author: lisandro
"""

#Librerias

import numpy as np
import matplotlib.pyplot as plt

#Funciones

def convolve(x,y):
    
    z = np.convolve(x[:,0],y[:,0])
    N = np.size(z)
    z = z.reshape(N,1)

    return z

def is_odd(num):
    return (num % 2) != 0

def is_even(num):
    return (num % 2) == 0

def b_spline(k,N):
    
    if k is 1:
        w = rectangular(N)
    else:
        n = np.zeros((int(k),),dtype = int)
        v = np.ceil(N/k)
        l = np.remainder(N-1,k)
        
        for i in range(k-1,-1,-1):
            j = k-(i+1)
            
            if j < l:
                n[i] = v + 1
            else:
                n[i] = v
        
        x0 = rectangular(n[0])
        r = x0[:,0]
        for ni in n[1:]:
            xi = rectangular(ni)
            r = np.convolve(xi[:,0],r)
        
        w = r.reshape(N,1)
            
    return w,n

def rectangular(N):
    
    wr = np.ones((N,1),dtype = float)
    
    return wr

def bartlett(N):
    
    n = np.linspace(0,N-1,N)
    
    wt = (1 - (np.abs(2*n - N + 1)/(N+1))).reshape(N,1)
    
    return wt

def hann(N):
    
    n = np.linspace(0,N-1,N)
    
    whn = (0.5 * (1 - np.cos( (2*np.pi*n)/(N-1) ))).reshape(N,1)
    
    return whn

def hamming(N):
    
    n = np.linspace(0,N-1,N)
    
    whm = (0.54 - 0.46*np.cos( (2*np.pi*n)/(N-1) )).reshape(N,1)
    
    return whm

def blackman(N):
    
    n = np.linspace(0,N-1,N)
    
    wb = (0.42 - 0.5*np.cos( (2*np.pi*n)/(N-1) ) + 0.08*np.cos( (4*np.pi*n)/(N-1) )).reshape(N,1)
    
    return wb

def flattop(N):
    
    n = np.linspace(0,N-1,N)
    
    wft = (1 - 1.93*np.cos( (2*np.pi*n)/(N-1) ) + 1.29*np.cos( (4*np.pi*n)/(N-1) ) - 0.388*np.cos( (6*np.pi*n)/(N-1) ) + 0.028*np.cos( (8*np.pi*n)/(N-1) )).reshape(N,1)
       
    return wft

#Testbench
    
def testbench():
    
    k = 4
    N = 2
    
    n = b_spline(k,N)
    
#    #Cantidad de muestras
#    N = 1024
#    
#    #Genera una ventana tipo bartlett(triangular)
#    wt = bartlett(N)
#    
#    #Presentacion grafica de los resultados temporales
#    plt.figure()
#    plt.title('Ventana de Bartlett, N = ' + str(N))
#    plt.plot(wt)
#    plt.xlabel('n')
#    plt.ylabel('wt[n]')
#    plt.grid()
#    
#    #Genera ventana tipo hanning
#    whn = hann(N)
#    
#    #Presentacion grafica de los resultados temporales
#    plt.figure()
#    plt.title('Ventana de Hann, N = ' + str(N))
#    plt.plot(whn)
#    plt.xlabel('n')
#    plt.ylabel('whn[n]')
#    plt.grid()
#    
#    #Genera ventana tipo hamming
#    whm = hamming(N)
#    
#    #Presentacion grafica de los resultados temporales
#    plt.figure()
#    plt.title('Ventana de Hamming, N = ' + str(N))
#    plt.plot(whm)
#    plt.xlabel('n')
#    plt.ylabel('whm[n]')
#    plt.grid()
#
#    #Genera ventana tipo blackman
#    wb = blackman(N)
#    
#    #Presentacion grafica de los resultados temporales
#    plt.figure()
#    plt.title('Ventana de Blackman, N = ' + str(N))
#    plt.plot(wb)
#    plt.xlabel('n')
#    plt.ylabel('wb[n]')
#    plt.grid()
#
#    #Genera ventana tipo flattop
#    wft = flattop(N)
#    
#    #Presentacion grafica de los resultados temporales
#    plt.figure()
#    plt.title('Ventana FlatTop, N = ' + str(N))
#    plt.plot(wft)
#    plt.xlabel('n')
#    plt.ylabel('wft[n]')
#    plt.grid()

testbench()