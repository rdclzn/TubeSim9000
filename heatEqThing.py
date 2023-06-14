# -*- coding: utf-8 -*-
"""
Created on Tue May  2 19:51:04 2023

@author: Alunos
https://aquaulb.github.io/book_solving_pde_mooc/solving_pde_mooc/notebooks/04_PartialDifferentialEquations/04_04_Diffusion_Implicit.html
"""
import sys

sys.path.insert(0, 'C:/Users/Alunos/Downloads/solving_pde_mooc-master/notebooks/modules')

import time

import numpy as np

# Function to compute an error in L2 norm
from norms import l2_diff

# Function to compute d^2 / dx^2 with second-order 
# accurate centered scheme
from matrices import d2_mat_dirichlet

# Function for the RHS of the heat equation and
# exact solution of our example problem
from pde_module import rhs_heat_centered, exact_solution


# Physical parameters
alpha = 25.0E-6                    # Heat transfer coefficient
lx = 1.                      # Size of computational domain
ti = 0.0                       # Initial time
tf = 0.1                       # Final time

# Grid parameters
nx = 25                       # number of grid points
dx = lx / (nx-1)               # grid spacing
x = np.linspace(0., lx, nx)    # coordinates of grid points

fourier = 1E-3                 # Fourier number
#dt = fourier*(dx**2)/alpha       # time step
dt = 1/44E3
nt = int((tf-ti) / dt)         # number of time steps

# d^2 / dx^2 matrix with Dirichlet boundary conditions
D2 = d2_mat_dirichlet(nx, dx)     

# I+A matrix
M = np.eye(nx-2) + alpha*dt*D2

# I-A matrix
M = np.eye(nx-2) - alpha*dt*D2
Minv = np.linalg.inv(M)

# Solution parameters
T0 = np.sin(2*np.pi*x)              # initial condition
sol = exact_solution(x, tf, alpha)  # Exact solution

T = np.empty((nt+1, nx)) # Allocate storage for the solution    
T[0] = T0.copy()         # Set the initial condition

times = np.empty(nt)
source = 2*np.sin(np.pi*x)          # heat source term
start = int(time.perf_counter_ns())
for i in range(nt):
    T[i+1, 1:-1] = np.dot(Minv, T[i, 1:-1] + source[1:-1]*dt)
    end = int(time.perf_counter_ns())
    times[i] = end - start
    if((times[i] > 0.01*1E6) and (times[i-1]< 0.01*1E6)):
        print(f'Veja a iteração número {i}')
    start = int(time.perf_counter_ns())
    

# Set boundary values
T[-1,0] = 0
T[-1,-1] = 0

total = np.sum(times)/1E6
std_dev = np.std(times/1E6)
per_loop = np.average(np.abs(times))/1E6

print(f'Total time was {total} ms')
print(f'Time per time-step was {per_loop} ±{std_dev} ms')

diff_exact = l2_diff(T[-1], sol)
print(f'The L2-error made in the computed solution is {diff_exact}')
print(f'Time integration required {nt} steps')