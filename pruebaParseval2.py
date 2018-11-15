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
    N = 2048
        
    generador = gen.signal_generator(fs,N)
    
    ao = np.sqrt(3)
    fo = np.pi/2
    po = 0
    
    (t,x) = generador.sinewave(ao,fo,po,freq = 'normalized_frequency')
    
#    dist = ("normal",)
#    (t,x) = generador.noise(dist,a1 = 0,a2 = 2)
    
    parseval_1 = np.sum(np.power(x,2))
    print('Energia en un periodo calculada con x(t)')
    print(parseval_1)
    print('Potencia media en un periodo calculada con x(t)')
    print(np.mean(np.power(x,2)))
    
    analizador = sa.spectrum_analyzer(fs,N)
    (f,Sxx) = analizador.psd(x,xaxis = 'phi')
    
    parseval_2 = np.sum(Sxx)
    print('Energia calculada de X(f) en un periodo')
    print(parseval_2)
    print('Potencia media en un periodo calculada con X(f)')
    print(parseval_2/N)
    
    plt.figure()
    plt.title('Ruido: Energia = ' + str(parseval_1))
    plt.plot(t,x)
    plt.grid()
    plt.axis('tight')
    plt.xlabel('t[s]')
    plt.ylabel('x(t)')
    
    plt.figure()
    plt.title('Periodograma: Energia = ' + str(parseval_2))
    plt.plot(f,Sxx)
    plt.grid()
    plt.axis('tight')
    plt.xlabel('f[Hz]')
    plt.ylabel('Pxx(f)')    

#Script

testbench()    
    