#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 16:07:40 2018

@author: lisandro
"""

#Librerias

import numpy as np
import matplotlib.pyplot as plt

import pdsmodulos.signal_generator as gen
import pdsmodulos.spectrum_analyzer as sa
import pdsmodulos.windows as win
from pandas import DataFrame

#Funciones
    
#Testbench

def testbench():
        
    #Paramettros del muestreo
    fs = 1024
    N = 1024
    
    #Realizaciones
    S = 200
    
    #Parametros de la señal x1
    a1 = np.sqrt(2)
    A1 = a1*np.ones((S,1),dtype = float)
    p1 = 0
    P1 = p1*np.ones((S,1),dtype = float)
    fo = np.pi/2
    
    #Parametros del ruido(distribucion normal)
    
    
    #Lista en que alamacenre las distribuciones para cada realizacion
    dist = []
    #Distribucion elegida para cada realizacion (todas normales)
    for i in range(0,S):
        dist.append("normal")    
    #Media - Todas las realizaciones de media 0
    u = 0
    U = u*np.ones((S,1),dtype = float)
    #Varianza - Se setea en funcion de snr,que indica cuantos db por debajo
    #quiero que este de x1
    snr = 10
    var = (N)*(np.power(a1,2)/2)*(np.power(10,-(snr/10)))
    SD = np.sqrt(var)*np.ones((S,1),dtype = float)
    
    #Limites de la distribucion uniforme de fr
    linf = -0.5*((2*np.pi)/N)
    lsup = 0.5*((2*np.pi)/N)
    
    #Fr sera una variable aleatoria de distribucion uniforme entre -1/2 y 1/2
    #Genero 200 realizaciones de fr
    fr = np.random.uniform(linf,lsup,S).reshape(S,1)
    
    #Genero 200 realizaciones de f1
    F1 = fo + fr

    #Enciendo el generador de funciones
    generador = gen.signal_generator(fs,N)
    
    #Genero 200 realizaciones de x
    (t,x1) = generador.sinewave(A1,F1,P1,freq = 'normalized_frequency')
    
    #Genero 200 realizaciones de n
    (t,n) = generador.noise(dist,U,SD)
    
    #Genero 200 realizaciones de x = x1 + n
    x = x1 + n
        
    #Estimador Periodograma
    (fp,Sxxm_p,Sxxv_p) = periodograma(x,fs,db = True) 

    f1_estimador_periodograma = (np.ediff1d(fp)[0])*np.argmax(Sxxm_p)
    sesgo_periodograma = f1_estimador_periodograma - fo
    
    #Estimador de Bartlett
    k = 32
    (fb,Sxxm_b,Sxxv_b) = bartlett(x,k,fs,db = True,window = 'hann')

    f1_estimador_bartlett = (np.ediff1d(fb)[0])*np.argmax(Sxxm_b)
    sesgo_bartlett = f1_estimador_bartlett - fo
    
    #Estimador de Welch
    k = 32
    (fw,Sxxm_w,Sxxv_w) = welch(x,k,fs,db  = True,window = 'hann',overlap = 50)
    
    f1_estimador_welch = (np.ediff1d(fw)[0])*np.argmax(Sxxm_w)
    sesgo_welch = f1_estimador_welch - fo

    print('f1p =' + str(f1_estimador_periodograma))
    print('sp =' + str(sesgo_periodograma))
    print('f1b =' + str(f1_estimador_bartlett))
    print('sb =' + str(sesgo_bartlett))
    print('f1w =' + str(f1_estimador_welch))
    print('sw =' + str(sesgo_welch))

    #Grafico los resultados
    plt.figure()
    plt.plot(fp,Sxxm_p)
    plt.grid()

    #Grafico los resultados
    plt.figure()
    plt.plot(fb,Sxxm_b)
    plt.grid()
    
    #Grafico los resultados
    plt.figure()
    plt.plot(fw,Sxxm_w)
    plt.grid()
    
#Script

testbench()