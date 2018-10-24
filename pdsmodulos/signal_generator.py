#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  4 09:28:10 2018

@author: lisandro
"""

import numpy as np
import matplotlib.pyplot as plt

class signal_generator:
    
    def __init__(self,fs = 1024,N = 1024):
                
        #Frecuencia de muestreo
        self.fs = fs
        #Periodo de muestreo (resolucion temporal)
        self.Ts = 1/fs
        #Cantidad de muestras
        self.N = N
        #Resolucion espectral
        self.df = fs/N
        #Vector de muestras
        self.n = np.linspace(0,self.N-1,self.N)
        #Vector temporal
        self.t = np.linspace(0,(self.N-1)*self.Ts,self.N)

        
    def zero_padding(self,x,Mj,plot = False):
        
        x = np.pad(x,((0,Mj),(0,0)),'constant')
        t_extra = np.linspace(0,(Mj - 1)*self.Ts,Mj);
        t_extra = t_extra + self.t[np.size(self.t,0)-1] + self.Ts
        t = np.append(self.t,t_extra)
    
        #Presentacion temporal de los resultados estocasticos
        if plot is True:
            plt.figure()
            plt.title('Zero padding')
            plt.plot(t,x)
            plt.axis('tight')
            plt.xlabel('t[s]')
            plt.ylabel('x(t)[V]')
            plt.grid()
        
        return (t,x)
    
    def sinewave(self,Ao,fo,po,plot = False,freq = 'bin'):
                
        #Genero una matriz vacia en la que voy a ir cargando todos los resultados.
        x = np.array([],dtype='float').reshape(self.N,0)
                
        #Transformo los parametros en tuplas. Al pasarle a esta funcion varios valo
        #res para cada uno de los parametros de la forma Ao = (Ao1,Ao2) seran inter
        #pretados como tuplas (parametro iterable) por la funcion zip. Pero al pasa
        #rle un solo parametros de la forma Ao = Ao1, zip detectara que no es un pa
        #rametro iterable y dara error. La forma correcta seria Ao = (Ao1,), pero
        #para garantizar el formato de los datos se realiza la tranformacion dentro
        #de la funcion.
    
        try:
            iter(Ao)
        except TypeError:
            Ao = tuple((Ao,))
    
        try:
            iter(fo)
        except TypeError:
            fo = tuple((fo,))
            
        try:
            iter(po)
        except TypeError:
            po = tuple((po,))   
        
        label = []
        
        #En cada ciclo de la iteracion se va obteniendo de las tuplas Ao,fo y po el
        #parametro necesario para generar la senoidal correspondiente.
        for amp_actual,frec_actual,fase_actual in zip(Ao,fo,po):

            if freq is 'bin':
                #Pulsacion angular actual(wo = 2*pi*fo).        
                puls_actual = 2*np.pi*frec_actual*self.df
                t = self.t
            elif freq is 'frequency':
                puls_actual = 2*np.pi*frec_actual
                t = self.t
            elif freq is 'normalized_frequency':
                puls_actual = frec_actual
                t = self.n
            elif freq is 'normalized':
                puls_actual = 2*np.pi*frec_actual
                t = self.n
            else:
                #Pulsacion angular actual(wo = 2*pi*fo).        
                puls_actual = 2*np.pi*frec_actual*self.df
                t = self.t
                
            #paso la fase actual a radianes
            fase_actual = np.deg2rad(fase_actual)
            
            
            #Calcula la senoidal actual.
            senoidal_actual = amp_actual*np.sin((puls_actual*t) + fase_actual);
                        
            #Agrego la senoidal actual a la matriz que contendra la senoidal
            #actual.
            x = np.hstack([x, senoidal_actual.reshape(self.N,1)])

            label.append("Ao: " + str(amp_actual) + "V fo: " + str(frec_actual) + "Hz po: " + str(np.rad2deg(fase_actual)) + "°")

        #Presentacion temporal de las senales senoidales
        if plot is True:
            plt.figure()
            plt.title('Senal senoidal')
            plt.plot(self.t,x),plt.legend(label,loc = 'upper right')
            plt.axis('tight')
            plt.xlabel('t[s]')
            plt.ylabel('x(t)[V]')
            plt.grid()
    
        return (self.t,x)
    
    def squarewave(self,Ao,fo,do,plot = False):
                
        #Genero una matriz vacia en la que voy a ir cargando todos los resultados.
        x = np.array([],dtype='float').reshape(self.N,0)
        
        #Transformo los parametros en tuplas. Al pasarle a esta funcion varios valo
        #res para cada uno de los parametros de la forma Ao = (Ao1,Ao2) seran inter
        #pretados como tuplas (parametro iterable) por la funcion zip. Pero al pasa
        #rle un solo parametros de la forma Ao = Ao1, zip detectara que no es un pa
        #rametro iterable y dara error. La forma correcta seria Ao = (Ao1,), pero
        #para garantizar el formato de los datos se realiza la tranformacion dentro
        #de la funcion.
        
        try:
            iter(Ao)
        except TypeError:
            Ao = tuple((Ao,))
        
        try:
            iter(fo)
        except TypeError:
            fo = tuple((fo,))
            
        try:
            iter(do)
        except TypeError:
            do = tuple((do,))        
        
        label = []
        
        #En cada ciclo de la iteracion se va obteniendo de las tuplas Ao,fo y do el
        #parametro necesario para generar la cuadrada correspondiente.
        for amp_actual,frec_actual,duty_actual in zip(Ao,fo,do):
            
            #Obtengo el periodo actual
            T_actual = 1/(frec_actual*self.df)
            
            #Paso el duty_actual de porcentaje a veces [0,1]
            duty_actual = duty_actual/100
            
            #A partir del duty actual y del periodo actual obtengo los tiempo 
            #en ON y OFF
            Ton = duty_actual*T_actual
            
            #Se toma el vector de tiempo y se genera un vector repetitivo que
            #vaya de 0 a (T_actual - Ts). Para esto se aplica el operador 
            #remainder,el cual genera el resto de t[i]/T_actual para i de 0 a N
            t_resto = np.remainder(self.t,T_actual)
            
            #Calcula la cuadrada actual
            cuadrada_actual = ((t_resto < Ton)*amp_actual) - ((t_resto >= Ton)*amp_actual);
            
            #Agrego la senoidal actual a la matriz que contendra la senoidal
            #actual
            x = np.hstack([x, cuadrada_actual.reshape(self.N,1)])
            
            label.append("Ao: " + str(amp_actual) + "V fo: " + str(frec_actual) + "Hz do: " + str(duty_actual))

            
        #Presentacion temporal de las senales cuadradas
        if plot is True:
            plt.figure()
            plt.title('Senal cuadrada')
            plt.plot(self.t,x),plt.legend(label,loc = 'upper right')
            plt.axis('tight')
            plt.xlabel('t[s]')
            plt.ylabel('x(t)[V]')
            plt.grid()
            
        #Se devuelven el vector de tiempos y la matriz de funciones cuadradas    
        return (self.t,x)
    
    def trianglewave(self,Ao,fo,do,plot = False):
         
        #Genero una matriz vacia en la que voy a ir cargando todos los resultados.
        x = np.array([],dtype='float').reshape(self.N,0)
        
        #Transformo los parametros en tuplas. Al pasarle a esta funcion varios valo
        #res para cada uno de los parametros de la forma Ao = (Ao1,Ao2) seran inter
        #pretados como tuplas (parametro iterable) por la funcion zip. Pero al pasa
        #rle un solo parametros de la forma Ao = Ao1, zip detectara que no es un pa
        #rametro iterable y dara error. La forma correcta seria Ao = (Ao1,), pero
        #para garantizar el formato de los datos se realiza la tranformacion dentro
        #de la funcion.
        
        try:
            iter(Ao)
        except TypeError:
            Ao = tuple((Ao,))
        
        try:
            iter(fo)
        except TypeError:
            fo = tuple((fo,))
        
        try:
            iter(do)
        except TypeError:
            do = tuple((do,))
        
        label = []
        
        #En cada ciclo de la iteracion se va obteniendo de las tuplas Ao,fo y do el
        #parametro necesario para generar la cuadrada correspondiente.
        for amp_actual,frec_actual,duty_actual in zip(Ao,fo,do):
        
            #Obtengo el periodo actual
            T_actual = 1/(frec_actual*self.df)
            
            #Paso el duty_actual de porcentaje a veces [0,1]
            duty_actual = duty_actual/100
            
            #A partir del duty actual y del periodo actual obtengo los tiempo 
            #en ON y OFF
            Tpos = duty_actual*T_actual
            Tneg = (1-duty_actual)*T_actual
            
            #Se obtiene los coeficientes a11 y a21
            #Esta funcion generara una triangular cuyo valor oscila entre +Ao y -Ao
               
            try:
                a11 = (2*amp_actual)/Tpos
            except ZeroDivisionError:
                a11 = 0
            
            try:
                a21 = -(2*amp_actual)/Tneg
            except ZeroDivisionError:
                a21 = 0
            
            a12 = -amp_actual
            a22 = amp_actual
            
            #Se toma el vector de tiempo y se genera un vector repetitivo que
            #vaya de 0 a (T_actual - Ts). Para esto se aplica el operador 
            #remainder,el cual genera el resto de t[i]/T_actual para i de 0 a N
            t_resto = np.remainder(self.t,T_actual)
            
            #Se genera las partes positivas por separado
            
            t_pos = t_resto*(t_resto <= Tpos)
            triangular_pos = a12 + a11*t_pos
            
            #Se genera la parte negativa por separado        
            
            t_neg = t_resto*(t_resto > Tpos)
            
            if Tpos != 0:
                t_neg = np.remainder(t_neg,Tpos)
            
            triangular_neg = a22 + a21*t_neg
            
            #Genera la triangular completa sumando la parte positiva y la negativa
            
            triangular_actual = (triangular_pos*(t_resto <= Tpos)) + (triangular_neg*(t_resto > Tpos))
                        
            #Agrego la senoidal actual a la matriz que contendra la senoidal
            #actual
            x = np.hstack([x, triangular_actual.reshape(self.N,1)])
        
            label.append("Ao: " + str(amp_actual) + "V fo: " + str(frec_actual) + "Hz do: " + str(duty_actual))

        #Presentacion temporal de las senales triangulares
        if plot is True:
            plt.figure()
            plt.title('Senal triangular')
            plt.plot(self.t,x),plt.legend(label,loc = 'upper right')
            plt.axis('tight')
            plt.xlabel('t[s]')
            plt.ylabel('x(t)[V]')
            plt.grid()         
        
        #Se devuelven el vector de tiempos y la matriz de funciones triangulares    
        return (self.t,x)        
        
    def noise(self,dist = "uniform",a1 = -0.5 ,a2 = 0.5,plot = False):
                       
        #Genero una matriz vacia en la que voy a ir cargando todos los resultados.
        x = np.array([],dtype='float').reshape(self.N,0)
        
        #Transformo los parametros en tuplas. Al pasarle a esta funcion varios valo
        #res para cada uno de los parametros de la forma Ao = (Ao1,Ao2) seran inter
        #pretados como tuplas (parametro iterable) por la funcion zip. Pero al pasa
        #rle un solo parametros de la forma Ao = Ao1, zip detectara que no es un pa
        #rametro iterable y dara error. La forma correcta seria Ao = (Ao1,), pero
        #para garantizar el formato de los datos se realiza la tranformacion dentro
        #de la funcion.
    
        try:
            iter(a1)
        except TypeError:
            a1 = tuple((a1,))
    
        try:
            iter(a2)
        except TypeError:
            a2 = tuple((a2,))
    
        try:
            iter(dist)
        except TypeError:
            dist = tuple((dist,))
        
        label = []
        
        #En cada ciclo de la iteracion se va obteniendo de las tuplas Ao,fo y po el
        #parametro necesario para generar la senoidal correspondiente.
        for a1_actual,a2_actual,dist_actual in zip(a1,a2,dist):
                        
            #Se fija cual es la distribucion elegida
            #Si distribucion es 1 la distribucion elegida es normal
            #a1 = media, a2 = varianza
            if dist_actual == "normal":
                ruido_actual = np.random.normal(a1_actual,a2_actual,self.N)
            
            #Si distribucion es 2 la distribucion elegida es la uniforme
            #a1 = limite inferior, a2 = limite superior
            elif dist_actual == "uniform":
                ruido_actual = np.random.uniform(a1_actual,a2_actual,self.N)
            
            #Si distribucion es 3 la distribucion elegida es laplaciana
            #a1 = loc,a2 = scale
            elif dist_actual == "laplace":
                ruido_actual = np.random.laplace(a1_actual,a2_actual,self.N)
            else:
                print("Distribucion no implementada")
            
            #Se calculan los parametros que se mostraran como legend
            #En este caso son media y varianza
            u = np.mean(ruido_actual,axis = 0)
            u = np.around(u,decimals=2)
            s = np.var(ruido_actual,axis = 0)
            s = np.around(s,decimals=4)
            
            label.append("Dist:" + dist_actual + " Media: " + str(u) + "Varianza: " + str(s))
                        
            #Agrego la senoidal actual a la matriz que contendra la senoidal
            #actual
            x = np.hstack([x, ruido_actual.reshape(self.N,1)])        
            
        #Presentacion temporal de las senales estocasticas
        if plot is True:
            plt.figure()
            plt.title('Ruido')
            plt.plot(self.t,x),plt.legend(label,loc = 'upper right')
            plt.axis('tight')
            plt.xlabel('t[s]')
            plt.ylabel('x(t)[V]')
            plt.grid()  
            
        return(self.t,x)