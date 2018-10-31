#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 21:33:15 2018

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
        
    k = int(np.power(2,k))
    
    n = np.size(x,0)
    l = int(n/k)
    
    realizaciones = np.size(x,1)
    
    Sx = np.zeros([int(l/2) + 1,int(realizaciones),int(k)],dtype = float)
    
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
    
    return f,Sxm,Sxv

#Testbench

def testbench():
        
    #Parametros del muestreo
    N = np.array([32,64,128,256,512,1024,2048,4096], dtype = int)
    
    #Frecuencias de muestreo
    fs = 1024
    
    #Cantidad de realizaciones
    S = 100
    
    #En cuantos bloques divido (2^k)
    k = 2
    
    #Aca se almacenaran los resultados
    tus_resultados = []
    sesgos = np.zeros([np.size(N),],dtype = float)
    varianzas = np.zeros([np.size(N),],dtype = float)
    
    #Contador
    j = 0
    
    #Para cada largo de señal
    for Ni in N:
                
        #Enciendo el generador de funciones
        generador = gen.signal_generator(fs,Ni)
                
        #Lista en que alamacenre las distribuciones para cada realizacion
        dist = []
        
        #Distribucion elegida para cada realizacion (todas normales)
        for i in range(0,S):
            dist.append("normal")
        
        #Media - Todas las realizaciones de media 0
        u = np.zeros((S,1),dtype = float)
        
        #Varianza - Todas las realizaciones de desvio estandar de raiz de 2
        s = np.sqrt(2)*np.ones((S,1),dtype = float)
        
        #Llamo al metodo que genera ruido blanco
        #Genera una matriz de NxS, donde N = Filas y S = Columnas
        (t,x) = generador.noise(dist,u,s)
            
        #Realizo de forma matricial el modulo del espectro de todas las realizaciones
        (f,Sxm,Sxv) = bartlett(x,k,fs,window = 'hann')
        
        #Calculo el area de ese espectro "promedio"
        #El area de la psd da la potencia
        valor_esperado = np.sum(Sxm)
        print('Valor esperado:' + str(valor_esperado))
        sesgo = valor_esperado - np.power(s[0,0],2)
        
        #Calculo el area de eso
        #TODO: Tengo un error de escala con esto. DETECTAR la fuente del problema
        varianza = np.sum(Sxv)
        print('Varianza del estimador:' + str(varianza))
        
        
        #Grafico la media de cada punto de frecuencia y la varianza de
        #cada punto de frecuencia
        plt.figure()
        plt.plot(f,Sxm,f,Sxv)
        plt.grid()
                
        #Almaceno los resultados para esta largo de señal
        tus_resultados.append([str(sesgo),str(varianza)])
        
        #Sesgos
        sesgos[j] = sesgo
        
        #Varianzas
        varianzas[j] = varianza
        
        #Aumento el contador
        j = j + 1
    
    
    #Presentación gráfica de resultados
    plt.figure()
    fig, axarr = plt.subplots(2, 1,figsize = (10,5)) 
    fig.suptitle('Evolución de los parámetros del periodograma en función del largo de la señal',fontsize=12,y = 1.08)
    fig.tight_layout()
    
    axarr[0].plot(N,sesgos)
    axarr[0].set_title('Sesgo del periodograma en función del largo de la señal')
    axarr[0].set_ylabel('$s_{p}[N]$')
    axarr[0].set_xlabel('$N$')
    axarr[0].set_ylim((1.1*min(sesgos),max(sesgos)*1.1))
    axarr[0].axis('tight')
    axarr[0].grid()
    
    axarr[1].plot(N,varianzas)
    axarr[1].set_title('Varianza del periodograma en función del largo de la señal')
    axarr[1].set_ylabel('$v_{p}[N]$')
    axarr[1].set_xlabel('$N$')
    axarr[1].set_ylim((1.1*min(varianzas),max(varianzas)*1.1))
    axarr[1].axis('tight')
    axarr[1].grid()
    
    #Almaceno el resultado en el dataframe
    df = DataFrame(tus_resultados, columns=['$s_P$', '$v_P$'],index=N)
    
    print(df)
    
#Script

testbench()