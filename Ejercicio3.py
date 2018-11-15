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

#Testbench

def testbench():
        
    #Parametros del muestreo
    N = np.array([1024], dtype = int)

    
    #Frecuencias de muestreo
    fs = 1000
    
    #Cantidad de realizaciones
    S = 500
    
    #En cuantos bloques divido (2^k)
    ki = 4
    
    #Overlap entre bloques
    ovi = 50
    
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
        s = np.sqrt(4)*np.ones((S,1),dtype = float)
        
        #Llamo al metodo que genera ruido blanco
        #Genera una matriz de NxS, donde N = Filas y S = Columnas
        (t,x) = generador.noise(dist,u,s)
            
        #Realizo de forma matricial el modulo del espectro de todas las realizaciones
        (f,Sxm,Sxv) = sa.welch(x,fs,k = ki,window = 'bartlett',overlap = 50)
        
        #Calculo el area de ese espectro "promedio"
        #El area de la psd da la potencia
        valor_esperado = (np.mean(Sxm))
        print('Valor esperado:' + str(valor_esperado))
        sesgo = valor_esperado - np.power(s[0,0],2)
        
        #Calculo el area de eso
        varianza = (np.mean(Sxv))
        print('Varianza del estimador:' + str(varianza))
        
        #Almaceno los resultados para esta largo de señal
        tus_resultados.append([str(sesgo),str(varianza)])
        
        #Sesgos
        sesgos[j] = sesgo
        
        #Varianzas
        varianzas[j] = varianza
        
        #Aumento el contador
        j = j + 1
    
    
    #Presentación gráfica de resultados
#    plt.figure()
#    fig, axarr = plt.subplots(2, 1,figsize = (10,5)) 
#    fig.suptitle('Evolución de los parámetros del periodograma en función del largo de la señal',fontsize=12,y = 1.08)
#    fig.tight_layout()
#    
#    axarr[0].stem(N,np.abs(sesgos))
#    axarr[0].set_title('Sesgo del periodograma en función del largo de la señal')
#    axarr[0].set_ylabel('$s_{p}[N]$')
#    axarr[0].set_xlabel('$N$')
#    axarr[0].set_ylim((1.1*min(sesgos),max(sesgos)*1.1))
#    axarr[0].axis('tight')
#    axarr[0].grid()
#    
#    axarr[1].stem(N,varianzas)
#    axarr[1].set_title('Varianza del periodograma en función del largo de la señal')
#    axarr[1].set_ylabel('$v_{p}[N]$')
#    axarr[1].set_xlabel('$N$')
#    axarr[1].set_ylim((1.1*min(varianzas),max(varianzas)*1.1))
#    axarr[1].axis('tight')
#    axarr[1].grid()
    
    #Almaceno el resultado en el dataframe
    df = DataFrame(tus_resultados, columns=['$s_P$', '$v_P$'],index=N)
    
    print(df)
    
#Script

testbench()