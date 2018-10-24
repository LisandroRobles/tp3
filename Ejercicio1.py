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
import pdsmodulos.statistic as sta
import pdsmodulos.adc as converter
import pdsmodulos.windows as win

import scipy.signal.windows as win

#Testbench

def testbench():
    
    #Parametros del muestreo
    N = 512
    fs = N
    
    #Cantidad de realizaciones
    S = 200
    
    #Enciendo el generador de funciones
    generador = gen.signal_generator(fs,N)
    
    #Enciendo el analizador de espectro
    analizador = sa.spectrum_analyzer(fs,N,"fft")
    
    #Lista en que alamacenre las distribuciones para cada realizacion
    dist = []
    
    #Distribucion elegida para cada realizacion (todas normales)
    for i in range(0,S):
        dist.append("normal")
    
    #Media - Todas las realizaciones de media 0
    u = np.zeros((S,1),dtype = float)
    
    #Varianza - Todas las realizaciones de raiz de 2
    s = np.sqrt(2)*np.ones((S,1),dtype = float)
    
    #Llamo al metodo que genera ruido blanco
    #Genera una matriz de NxS, donde N = Filas y S = Columnas
    (t,x) = generador.noise(dist,u,s)
        
    #Realizo de forma matricial el modulo del espectro de todas las realizaciones
    (f,Xmod) = analizador.module(x,xaxis = 'phi')
    #Para pasar de veces a psd tengo que dividir por dos, luego elevar al cuadrado y volver a multiplicar por dos
    Xpsd = np.transpose(np.array([[Xij/2 if (Xij != Xi[0] and Xij != Xi[np.size(Xmod,0)-1]) else Xij for Xij in Xi] for Xi in np.transpose(Xmod)],dtype = float))
    Xpsd = 2*np.power(Xpsd,2)
    
    #Una vez que tengo todas las realizaciones de la PSD le calculo el espectro promedio. 
    #Esto quiere decir, calcular la media de ca
    #da fila. Si la matriz es de NxS. Me tiene que quedar una matriz de Nx1
    #Es decir, el promedio a cada frecuencia
    m = np.mean(Xpsd,1)
    
    #Calculo el area de ese espectro "promedio"
    valor_esperado = np.sum(m)
    
    #Calculo la varianza a cada frecuencia
    v = np.var(Xpsd,1)
    
    #Calculo el area de eso
    #TODO: Tengo un error de escala con esto. DETECTAR la fuente del problema
    varianza = (N/2)*np.sum(v)
    
    return (valor_esperado,varianza)
    
#Script

(valor_esperado,varianza) = testbench()