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
from pandas import DataFrame

import spectrum as sp

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
        
    #Parametros del muestreo
    N = np.array([256, 512, 1024, 2048], dtype = int)
    
    #Frecuencias de muestreo
    fs = 1024
    
    #Cantidad de realizaciones
    S = 100
    
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
            
        #Periodograma
        (f,Xpsd) = periodograma(x,fs)
        
        Xpsd2 = sp.Periodogram(x,sampling = fs,NFFT = int(np.size(x,0)))

        #Una vez que tengo todas las realizaciones de la PSD le calculo el espectro promedio. 
        #Esto quiere decir, calcular la media de ca
        #da fila. Si la matriz es de NxS. Me tiene que quedar una matriz de Nx1
        #Es decir, el promedio a cada frecuencia
        m = np.mean(Xpsd,1)
        
        #Calculo el area de ese espectro "promedio"
        valor_esperado = np.sum(m)
        
        sesgo = valor_esperado - np.power(s[0,0],2)
        
        #Calculo la varianza a cada frecuencia
        v = np.var(Xpsd,1)
        
        #Calculo el area de eso
        #TODO: Tengo un error de escala con esto. DETECTAR la fuente del problema
        varianza = np.sum(v)
        
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