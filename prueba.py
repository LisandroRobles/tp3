#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 10:00:20 2018

@author: lisandro
"""

#Paquetes

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal

import pdsmodulos.signal_generator as gen
import pdsmodulos.spectrum_analyzer as sa

#Testbench

def testbench():
    
    fs = 1024
    N = 1024
    Ts = 1/fs
    df = fs/N
    
    Ao = np.sqrt(1)
    fo1 = np.pi/2
    fo2 = np.pi/4
    po = 0
    
    generador = gen.signal_generator(fs,N)
    
    (t,x1) = generador.sinewave(Ao,fo1,po,freq = 'normalized_frequency')
    (t,x2) = generador.sinewave(Ao,fo2,po,freq = 'normalized_frequency')    
    x = x1 + x2
    
    parseval_1 = np.sum(np.power(x,2))*Ts
    
    print('Energia calculada calculada con x(t)')
    print(parseval_1)
    
    analizador = sa.spectrum_analyzer(fs,N)
    
    (f,Sxx) = analizador.psd(x,xaxis = 'phi')
    
    parseval_2 = np.sum(Sxx)*df 
    
    print('Energia calculada de X(f)')
    print(parseval_2)
    
    plt.figure()
    plt.title('Senoidal')
    plt.plot(t,x)
    plt.grid()
    plt.axis('tight')
    plt.xlabel('t[s]')
    plt.ylabel('x(t)')
    
    
    
#    print('Energia de la senoidal')
#    print(np.sum(Pxx_spec))
#
#    plt.figure()
#    plt.title('Periodograma')
#    plt.plot(f,Pxx_spec)
#    plt.grid()
#    plt.axis('tight')
#    plt.xlabel('f[Hz]')
#    plt.ylabel('Pxx(f)')    

#Script

testbench()    
    