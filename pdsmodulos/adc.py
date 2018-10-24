#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 09:53:23 2018

@author: lisandro
"""

import numpy as np
import matplotlib.pyplot as plt

class adc():
    
    def __init__(self,fs = 1024,Vref_low = -1,Vref_high = 1,N = 8):
        
        self.Vref_low = Vref_low
        
        self.Vref_high = Vref_high
        
        self.Vfsr = (Vref_high) - (Vref_low)
        
        self.N = N
        
        self.M = np.power(2,self.N);
        
        self.C_max = np.power(2,self.N-1) - 1
        
        self.C_min = -np.power(2,self.N-1)
        
        self.C = np.linspace(0,self.M-1,self.M)
        
        self.q = (self.Vfsr)/(self.M)
                
        self.fs = fs
        
    def muestrear(self,t,x):
        
        Ts_x = (t[1] - t[0])
        fs_x = 1/(Ts_x)
        
        k = int(np.round(fs_x/self.fs))
        
        end = np.size(t,0)
        
        t = t[0:end:k]
        
        x = x[0:end:k]
        
        return (t,x)
    
    def cuantizar(self,t,x,plot = False):
        
        xq = np.round(x/self.q)
        
        for i in range(np.size(x,0)):
            
            if xq[i] >= self.C_max:
                xq[i] = self.C_max
            if xq[i] <= self.C_min:
                xq[i] = self.C_min
        
        xq = xq*self.q
        
        if plot is True:
            plt.figure()
            plt.plot(t,x,t,xq)
        
        return (t,xq)
            
        
        
        
        
        
        
