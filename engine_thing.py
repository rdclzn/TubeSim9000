# -*- coding: utf-8 -*-
import numpy as np
import enum

"""
Spyder Editor

This is a temporary script file.
"""

"""Parametros do Motor
    Inline - 6 cyl - Dia cyl 70mm
    24V - 2.0L
    ⅓ L (0.33333 L)
    9.5:1
    
    Carburador:
        Coeficiente de Descarga: 0.94
        Coeficiente de Descargda para Idle: 0.85
        AF choke: 1:1
        AF full: 15:1
    R (l/a) = 2.56 ou 3.85
    
    Gasolina
    96 RON
    750 kg/m3
    45 150 kJ/kg
    Molecular Weight: 111
    
    Ar
    k: {500: 1.39, 850: 1.35, 1000: 1.34, 1500: 1.31, 2000: 1.30, 2500: 1.29}
    
    if 350K < T < 3000K:
        cv = 0.0001*T[K] + 0.736429
        cp = cv*k
    else: 
        cv = 0.718
        cp = 1.005
    R = 0.287 kJ/kg-K
    
"""

class Firing(enum.Enum):
    INLINE_SYMETRIC=1
    CROSS_FIRE=2
    
class State(enum.Enum):
    INTAKE=1
    COMPRESSION=2
    POWER=3
    EXHAUST=4

class air(object):
    R = 0.287 #kj/kg-K
    amb_T = 300
    amb_P = 101.325 #kPa
    rho = 1.177
    k_temps = [300, 500, 850, 1000, 1500, 2000, 2500]
    k_list = [1.40, 1.39, 1.35, 1.34, 1.31, 1.30, 1.29]
    k = 1.40

    @classmethod
    def get_rho(cls,P,T):
        '''P in kPA and T in Kelvin'''
        rho = P/(air.R*T)
        return rho

    def get_k(T):
        return np.interp(T, air.k_temps, air.k_list)

    def get_cv(T):
        return air.R/(air.get_k(T) - 1)

    def get_cp(T):
        k = air.get_k(T)
        return air.R*k/(k - 1)

    def get_soundSPD(T):
        return np.sqrt(air.get_k(T)*air.R*1000*T)
    
#REFAZER
#https://en.wikipedia.org/wiki/Piston_motion_equations
#https://www.fiatforum.com/threads/all-fire-engine-cam-specs-needed.36359/

class Cylinder:
    def __init__(self,Displacement,Bore,CR,R,Pos,Piston_Mass,theta_e0, theta_e1, theta_i0, theta_i1, state: State):
        self.displacement = Displacement/1E3
        self.area = ((Bore/1000)**2)*np.pi/4.0
        self.stroke_length = self.displacement/(self.area)
        self.crankshaft_radius = self.stroke_length/2
        self.min_volume = self.displacement/(CR - 1)
        self.Ai = 1.3*((Bore/1000)**2)*((self.stroke_length*2/(5000/60))/air.get_soundSPD(90))
        self.Ae = self.Ai*0.9
        self.degpos = Pos
        self.pos = np.deg2rad(self.degpos)
        self.ypos = self.crankshaft_radius*(R+1) - (self.crankshaft_radius*np.cos(np.deg2rad(Pos)) + (self.crankshaft_radius)*np.sqrt(R**2 - np.sin(np.deg2rad(Pos))**2))
        self.P = air.amb_P #kPa amb temp
        self.T = air.amb_T
        self.volume = self.min_volume + (np.pi*self.area/4)*self.ypos
        self.bulk_spd = 0
        self.speed = 0
        self.acc = 0
        self.gasMASS = self.volume*air.get_rho(self.P, self.T)
        if not isinstance(state,State):
            self.state = State(state)
        else:
            self.state = state
        

class Engine:
    def __init__(self, cyl_number, displacement, bore, ratio, valves, R, piston_mass, CD_idle, CD_carb,theta_e0, theta_e1, theta_i0, theta_i1, timing: Firing):
        self.cyls = cyl_number
        self.cyl = [0]*cyl_number
        self.displacement = displacement
        self.valves = valves
        if not isinstance(timing,Firing):
            self.firing = Firing(timing)
        else:
            self.firing = timing
        if self.firing == Firing.INLINE_SYMETRIC:            
            for i in range(0,self.cyls):
                Pos = np.mod(i,2)*(180)
                k = min(((i*2) + 1), (self.cyls - i)*2)
                if k<5:
                    k = np.mod(k,5) 
                else:
                    k = np.mod(k,5) + 1
                self.cyl[i] = Cylinder((displacement/self.cyls),bore,ratio,R,Pos,piston_mass,theta_e0, theta_e1, theta_i0, theta_i1, State(k))
        if self.firing == Firing.CROSS_FIRE:
            for i in range(0,self.cyls):
                Pos = i*int(360/self.cyls)
                if k < 5:
                    k = np.mod(i,5)
                else:
                    k = np.mod(i,5) + 1
                self.cyl[i] = Cylinder((displacement/self.cyls),bore,ratio,R,Pos,piston_mass,theta_e0, theta_e1, theta_i0, theta_i1, State(k))
                

bob = Engine(4, 1.0, 70, 9.5, 12, 2.560, 3.5, 0.95, 0.80, 300, 0, 20, 90, Firing.INLINE_SYMETRIC)

