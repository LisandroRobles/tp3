#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  5 16:51:01 2018

@author: lisandro
"""
import numpy as np
import matplotlib.pyplot as plt

class statistic:
    
    def __init__(self):
        
        print("Toolbox Estadistico")
        
    def histogram(self,x,plot = False):
        
        for x_actual in np.transpose(x):
            #Presento el histograma, con esto se debe verificar la distribucion de probabilidad
            plt.figure()
            plt.hist(x_actual,bins = 'auto',normed = 'True')
            plt.axis('tight')
            plt.xlabel('Variable aleatoria')
            plt.ylabel('Probabilidad')
            plt.grid()    
        
    def correlation(self,x,y):
        
        print("Correlacion")
        Nx = np.size(x,axis = 0)
        Ny = np.size(y,axis = 0)
        
        x = np.reshape(x,Nx)
        y = np.reshape(y,Ny)
        
        Total = Nx + Ny - 1
    
        r = np.correlate(x,y,mode = 'full')/Total
        k = np.linspace(0,1,Total)
        
        plt.figure()
        plt.plot(k,r)
        
        return (k,r)
        
    def autocorrelation(self,x):
        
        print("Autocorrelacion")