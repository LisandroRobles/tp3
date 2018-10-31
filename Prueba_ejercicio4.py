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

def periodograma(x,fs):
    
    #Largo de x
    n = np.size(x,0)

    #Enciendo el analizador de espectro
    analizador = sa.spectrum_analyzer(fs,n,"fft")

    #Realizo de forma matricial el modulo del espectro de todas las realizaciones
    (f,Sx) = analizador.psd(x,xaxis = 'phi')

    #Hago el promedio en las realizaciones
    Sxm = np.mean(Sx,1)
    
    #Hago la varianza en las realizaciones
    Sxv = np.var(Sx,1)
    Sxv = Sxv*np.size(Sxv,0)
    
    return (f,Sxm,Sxv)

def bartlett(x,k,fs,window = 'rectangular'):
        
    k = int(k)
    
    n = np.size(x,0)
    l = int(np.floor(n/k))
    
    if l is not 0:
    
        realizaciones = np.size(x,1)
        
        if (l % 2) != 0:
            largo_espectro_un_lado = int((l+1)/2)
        else:
            largo_espectro_un_lado = int((l/2)+1)
        
        Sx = np.zeros([largo_espectro_un_lado,int(realizaciones),int(k)],dtype = float)
        
        if window is 'rectangular':
            w = win.rectangular(l)
        elif window is 'bartlett':
            w = win.bartlett(l)
        elif window is 'hann':
            w = win.hann(l)
        elif window is 'hamming':
            w = win.hamming(l)
        elif window is 'blackman':
            w = win.blackman(l)
        elif window is 'flat-top':
            w = win.flattop(l)
        else:
            w = win.rectangular(l)
            
        for i in range(0,k):
            
            #Obtengo el bloque i-esimo para cada realizacion
            xi = x[int(i*l):int((i+1)*l),:]
            
            #Al bloque i-esimo de cada realizacion le aplico el ventaneo correspondiente
            xwi = (xi*w)
            
            #Enciendo el analizador de espectro
            analizador = sa.spectrum_analyzer(fs,l,"fft")
            
            #Obtengo el espectro de modulo del bloque i-esimo para cada realizacion
            (f,Sxi) = analizador.psd(xwi,xaxis = 'phi')
            
            #Divido por la energia de la ventana
            Sxi = Sxi/(np.mean(np.power(w,2)))
            
            #Lo agrego al resultado general
            Sx[:,:,i] = Sxi
        
        #Promedio para cada realizacion cada uno de los bloques
        #Deberia quedar una matriz con lxr
        #Donde l es el largo del bloque y r es la cantidad de realizaciones
        Sx = np.mean(Sx,axis = 2)
        
        #Hago el promedio en las realizaciones
        Sxm = np.mean(Sx,1)
        
        #Hago la varianza en las realizaciones
        Sxv = np.var(Sx,1)
        Sxv = Sxv*np.size(Sxv,0)
    
    else:
        
        print('La cantidad de bloques es mayor al largo de las realizaciones')
        f = 0
        Sxm = 0
        Sxv = 0
    
    return f,Sxm,Sxv

def welch(x,k,fs,window = 'rectangular',overlap = 50):
        
    k = int(k)
    
    n = np.size(x,0)
    l = int(np.floor(n/k))
    
    if l is not 0:
    
        realizaciones = np.size(x,1)
        
        if (l % 2) != 0:
            largo_espectro_un_lado = int((l+1)/2)
        else:
            largo_espectro_un_lado = int((l/2)+1)
                
        if window is 'rectangular':
            w = win.rectangular(l)
        elif window is 'bartlett':
            w = win.bartlett(l)
        elif window is 'hann':
            w = win.hann(l)
        elif window is 'hamming':
            w = win.hamming(l)
        elif window is 'blackman':
            w = win.blackman(l)
        elif window is 'flat-top':
            w = win.flattop(l)
        else:
            w = win.rectangular(l)
        
        overlap_eficaz = np.ceil(l*(overlap/100))/l
        
        if int(l*overlap_eficaz) > (l-1):
            overlap_eficaz = ((l-1)/(l))
        
        print('Procentaje de  solapamiento entre frames: ' + str(overlap_eficaz))
        
        
        paso = int(l*(1-overlap_eficaz))
        
        print('Distancia entre frames: ' + str(paso))
        
        #Determinacion de la cantidad de frames
        for cant_frames in range(0,n):
            if int(n-(l + int(cant_frames*paso))) < paso:
                print('Cantidad de frames: ' +str(cant_frames+1))
                cant_frames = cant_frames+1
                break

        Sx = np.zeros([largo_espectro_un_lado,int(realizaciones),int(cant_frames)],dtype = float)
        
        for i in range(0,cant_frames):
            
            #Obtengo el bloque i-esimo para cada realizacion
            xi = x[int(i*paso):int((i*paso)+l),:]
            
            #Al bloque i-esimo de cada realizacion le aplico el ventaneo correspondiente
            xwi = (xi*w)
            
            #Enciendo el analizador de espectro
            analizador = sa.spectrum_analyzer(fs,l,"fft")
            
            #Obtengo el espectro de modulo del bloque i-esimo para cada realizacion
            (f,Sxi) = analizador.psd(xwi,xaxis = 'phi')
            
            #Divido por la energia de la ventana
            Sxi = Sxi/(np.mean(np.power(w,2)))
            
            #Lo agrego al resultado general
            Sx[:,:,i] = Sxi
        
        #Promedio para cada realizacion cada uno de los bloques
        #Deberia quedar una matriz con lxr
        #Donde l es el largo del bloque y r es la cantidad de realizaciones
        Sx = np.mean(Sx,axis = 2)
        
        #Hago el promedio en las realizaciones
        Sxm = np.mean(Sx,1)
        
        #Hago la varianza en las realizaciones
        Sxv = np.var(Sx,1)
        Sxv = Sxv*np.size(Sxv,0)
    
    else:
        
        print('La cantidad de bloques es mayor al largo de las realizaciones')
        f = 0
        Sxm = 0
        Sxv = 0
    
    return f,Sxm,Sxv

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
    snr = 3
    var = (N)*(np.power(a1,2)/2)*(np.power(10,-(snr/10)))
    SD = np.sqrt(var)*np.ones((S,1),dtype = float)
    
    #Limites de la distribucion uniforme de fr
    linf = -0*((2*np.pi)/N)
    lsup = 0*((2*np.pi)/N)
    
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
    
    #Enciendo el analizador de espectro
    analizador = sa.spectrum_analyzer(fs,N)
    
    #Obtengo la psd para cada una de las realizaciones
    (f,Sxx) = analizador.psd(x,db = True,xaxis = 'phi')
    
    Sxxm = np.mean(Sxx,1)
    
    #Grafico los resultados
    plt.figure()
    plt.plot(f,Sxxm)
    plt.grid()
    
#Script

testbench()