#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 21:28:10 2018

@author: lisandro
"""

#Paquetes

import numpy as np
import matplotlib.pyplot as plt
import pdsmodulos.signal_generator as sg
import pdsmodulos.spectrum_analyzer as sa
from scipy.signal import periodogram

#Funciones

#Testbench

def testbench():
    
    #Parametros del muestreo
    fs = 1024
    N = 45
    
    #Parametros de la señal
    ao = np.sqrt(1)
    po = 0
    fo = np.pi/2
    
    #Inicializo el generador de funciones
    generador = sg.signal_generator(fs,N)
    
    #Genero la señal
    (t,x) = generador.sinewave(ao,fo,po,plot = False,freq = 'normalized_frequency')
        
    #Inicializo el analizador de espectro
    analizador = sa.spectrum_analyzer(fs,N)
    
    #Obtengo la psd de la señal
    (f1,Sx) = analizador.module(x,xaxis = 'phi')
    
    print(max(Sx))
    
    plt.figure()
    plt.plot(f1,Sx)
    
#Script

testbench()
