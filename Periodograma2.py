#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 21:23:03 2018

@author: lisandro
"""

#Paquetes

import numpy as np
import spectrum as sp
import scipy.signal as sig
import matplotlib.pyplot as plt

import pdsmodulos.signal_generator as gen
import pdsmodulos.spectrum_analyzer as sa

#Funciones

def periodograma(x,fs):
    
    #Largo de x
    n = np.size(x,0)

    #Enciendo el analizador de espectro
    analizador = sa.spectrum_analyzer(fs,n,"fft")

    #Realizo de forma matricial el modulo del espectro de todas las realizaciones
    (f,Xmod) = analizador.module(x,xaxis = 'phi')
    #Para pasar de veces a psd tengo que dividir por dos, luego elevar al cuadrado y volver a multiplicar por dos
    Sx = np.transpose(np.array([[Xij/2 if (Xij != Xi[0] and Xij != Xi[np.size(Xmod,0)-1]) else Xij for Xij in Xi] for Xi in np.transpose(Xmod)],dtype = float))
    Sx = 2*np.power(Sx,2)
    
    return (f,Sx)

#Testbench

def testbench():
    
    fs = int(1024)
    N = int(1024)
    
    
    Ao = 1
    fo = np.pi/2
    po = 0
    
    generador = gen.signal_generator(fs,N)
    
    (t,x) = generador.sinewave(Ao,fo,po,freq = 'normalized_frequency')
    
    
    (f,Sxx) = sig.periodogram(np.transpose(x),fs = fs,nfft = N)
    Sxx = np.transpose(Sxx)

    Sxx2 = (sp.Periodogram(x,sampling = fs,NFFT = N)).psd
    
    (f,Sxx3) = periodograma(x,fs)
    
    print(max(Sxx))
    
    print(max(Sxx2))
    
    print(max(Sxx3))
    
    plt.plot(Sxx)
    
    plt.plot(Sxx2)
    
    plt.plot(Sxx3)

#Script
    
testbench()