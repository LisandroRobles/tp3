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
import spectrum as sp

eps = np.finfo(float).eps

#Funciones

#Testbench

def testbench():
            
    #Paramettros del muestreo
    fs = 1000
    N = 1000
    
    #Realizaciones
    S = 200
    
    #Parametros de la señal x1
    a1 = np.sqrt(2)
    A1 = a1*np.ones((S,1),dtype = float)
    p1 = 0
    P1 = p1*np.ones((S,1),dtype = float)
    fo = np.pi/2

    #Limites de la distribucion uniforme de fr
    linf = -0.5*((2*np.pi)/N)
    lsup = 0.5*((2*np.pi)/N)
    
    #Fr sera una variable aleatoria de distribucion uniforme entre -1/2 y 1/2
    #Genero 200 realizaciones de fr
    fr = np.random.uniform(linf,lsup,S).reshape(S,1)
    
    #Genero 200 realizaciones de f1
    F1 = fo + fr
    
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
    snr = 20
    var = (N/2)*(np.power(a1,2)/2)*(np.power(10,-((snr+10)/10)))
    #var = np.power(10,(-snr/10))
    SD = np.sqrt(var)*np.ones((S,1),dtype = float)
    


    #Enciendo el generador de funciones
    generador = gen.signal_generator(fs,N)
    
    #Genero S realizaciones de x
    (t,x1) = generador.sinewave(A1,F1,P1,freq = 'normalized_frequency')
    
    #Genero S realizaciones de n
    (t,n) = generador.noise(dist,U,SD)
    
    #Genero S realizaciones de x = x1 + n
    x = x1 + n
        
    #Obtengo la psd para cada una de las realizaciones
    (f,Sxm,Sxv) = sa.periodograma(x,fs)
    ki = 4
    (f,Sxm,Sxv) = sa.welch(x,fs,k = ki,window = 'bartlett',overlap = 50,ensemble = True)
    
#    fo_estimador = np.zeros((S,),dtype = float)
#    
#    for i in range(0,S):
#        p = sp.pcorrelogram(x[:,i], lag=100, NFFT=N, scale_by_freq=True,sampling = (2*np.pi))
#        psd = p.psd + eps
#        Sxx = 10*np.log10(np.abs(psd/np.max(psd)))
#        df = p.df
#        fo_estimador[i] = df*(np.argmax(Sxx,axis = 0))
    
    df = f[1] - f[0]
    fo_welch = df*(np.argmax(Sxm,axis = 0))
    fo_valor_esperado = np.mean(fo_welch)
    sesgo_fo = fo_valor_esperado - fo
    varianza_fo = np.var(fo_welch)
        
    print(fo_valor_esperado)
    print(sesgo_fo)
    print(varianza_fo)
    
#Script

testbench()