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

#Funciones

def bartlett(x,k,fs):
    
    k = int(np.power(2,k))
    
    n = np.size(x,0)
    l = int(n/k)
    
    realizaciones = np.size(x,1)
    
    X = np.zeros([int(l/2) + 1,int(realizaciones),int(k)],dtype = float)
    
    for i in range(0,k):
        
        #Obtengo el bloque i-esimo para cada realizacion
        xi = x[int(i*l):int((i+1)*l),:]
        
        #Enciendo el analizador de espectro
        analizador = sa.spectrum_analyzer(fs,l,"fft")
        
        #Obtengo el espectro de modulo del bloque i-esimo para cada realizacion
        (f,Ximod) = analizador.module(xi,xaxis = 'phi')
        
        #Paso de veces a potencia
        Si = np.transpose(np.array([[Xij/2 if (Xij != Xi[0] and Xij != Xi[np.size(Ximod,0)-1]) else Xij for Xij in Xi] for Xi in np.transpose(Ximod)],dtype = float))
        Si = 2*np.power(Si,2)
        
        X[:,:,i] = Si
    
    #Promedio para cada realizacion cada uno de los bloques
    #Deberia quedar una matriz con lxr
    #Donde l es el largo del bloque y r es la cantidad de realizaciones
    X = np.mean(X,axis = 2)
        
    return f,X

#Testbench

def testbench():
        
    #Parametros del muestreo
    N = np.array([8,16,32,64,128,256,512,1024], dtype = int)
    
    #Frecuencias de muestreo
    fs = 1024
    
    #Cantidad de realizaciones
    S = 100
    
    #En cuantos bloques divido (2^k)
    k = 1
    
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
        (f,Sx) = bartlett(x,k,fs)
        
        #Calculo el espectro promediado en ensemble
        
        m = np.mean(Sx,axis = 1)
        
        #Calculo el area de ese espectro "promedio"
        valor_esperado = np.sum(m)
        
        sesgo = valor_esperado - np.power(s[0,0],2)
        
        #Calculo la varianza en ensemble
        v = np.var(Sx,axis = 1)
        
        #Calculo el area de eso
        #TODO: Tengo un error de escala con esto. DETECTAR la fuente del problema
        varianza = (Ni/2)*np.sum(v)
        
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