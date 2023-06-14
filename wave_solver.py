# -*- coding: utf-8 -*-
"""
Created on Thu May 11 13:32:31 2023

@author: Delícia, Pai
"""

import cv2
import os
import numpy as np
import time
import glob

"""Function solver solves the wave equation
   u_tt = c**2*u_xx + f(x,t) 
   
   On (0,L) with u=U_0 or du/dn=0 on x=0, and u=u_L or du/dn=0
   on x = L. If U_0 or U_L equals None, the du/dn=0 condition
   is used, otherwise U_0(t) and/or U_L(t) are used for Dirichlet cond.
   
   Initial conditions: u=I(x), u_t=V(x).
   
    T is the stop time for the simulation.
    dt is the desired time step.
    C is the Courant number (=c*dt/dx).
       
    I, f, U_0, U_L are functions: I(x), f(x,t), U_0(t), U_L(t).
   
    U_0 and U_L can also be 0, or None, where None implies:
    du/dn=0 boundary condition. f and V can also be 0 or None (equivalent to 0).
"""

def video_generator(image_folder='images',video_name='video.avi'):

    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape
    
    video = cv2.VideoWriter(video_name, 0, 1, (width,height))
    
    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))
    
    cv2.destroyAllWindows()
    video.release()


def sound_speed_recalculator(c, x):
    
    # Treat c(x) as array
    if isinstance(c, (float,int)):
        c = np.zeros(x.shape) + c
    elif callable(c):
        # Call c(x) and fill array c
        c_ = np.zeros(x.size)
        for i in range():
            c_[i] = c(x[i])
        c = c_
        
    q = np.power(c,2)
    return c, q
    

def initial_conditioner(I, V, f, c, U_0, U_L, dt, C, L, safety_factor=0.7071, user_action=None, attenuation=False):
    """
    @desc: Sets initial conditions and first step for u_tt=c^2*u_xx + f on (0,L)x(0,T]
    if U_0 or U_L are None, it indicates a Neumann condition du/dt = 0
    user_action is a function of f(u[n],x[n],t), n being Nx+1
    The safety factor must be less than 0.7072 if there's damping
    
    @return: u, u_1, u_2, x, dt*2, c, q, C2, B_p, B1, B_m, U_L, U_0, I, V, f
    """
    
    #checks if c is a function or vector that varies with x[n], and gets c_max
    if isinstance(c, (float,int)):
        c_max = c        
    elif callable(c):
        c_max = max([c(x_) for x_ in np.linspace(0, L, 1001)])
    
    dx = dt*c_max/(safety_factor*C)
    Nx = int(np.round(L/dx))
    x =  np.linspace(0, L, num=Nx+1)
    
    # Treat c(x) as array
    if isinstance(c, (float,int)):
        c = np.zeros(x.shape) + c
    elif callable(c):
        # Call c(x) and fill array c
        c_ = np.zeros(x.shape)
        for i in range(Nx+1):
            c_[i] = c(x[i])
        c = c_

    
    #checks if damping coefficient is defined
    #it must be a float or integer
    
    if attenuation is not False:
        if not isinstance(attenuation,(float,int)):
            #if it's a function, this is the default valu
            b = 5.0E-2
        else:
            b = attenuation
    else:
        #if None, then there's no damping
        b = 0.0
    
    #coefficient helper variable
    q = np.power(c,2)
    
    #damping helper variables
    B_p = 2 + b*dt
    B1 = 1/B_p
    B_m = b*dt - 2
    
    #Last scheme helper variable
    C2 = np.power((dt/dx), 2)
    
    
    if f is None or f == 0 :
        f = lambda x, t: 0
    if V is None or V == 0:
        V = lambda x: 0
    if I is None or I == 0:
        I = lambda x: 0
    if U_0 is not None:
        if isinstance(U_0, (float,int)):
            if U_0 == 0:
                U_0 = lambda t: 0
            else:
                aux = U_0
                U_0 = lambda t: float(aux)
        # else: U_0(t) is a function
    if U_L is not None:
        if isinstance(U_L, (float,int)):
            if U_L == 0:
                U_L = lambda t: 0
            else:
                aux = U_L
                U_L = lambda t: float(aux)
        # else: U_L(t) is a function
        
    u = np.zeros(Nx+1)
    u_1 = np.zeros(Nx+1)
    u_2 = np.zeros(Nx+1)

    # Load initial condition into u_1
    u_1[:] = I(x)
    
    if user_action is not None:
        user_action(u_1, x, 0)
        
    # Special formula for first time step
    
    #This is the scheme for model without damping
    # u[1:-1] = u_1[1:-1] + dt*V(x[1:-1]) + 0.5*C2*(0.5*(q[1:-1] + q[2:])*(u_1[2:] - u_1[1:-1]) - \
    #                                               0.5*(q[1:-1] + q[0:-2])*(u_1[1:-1] - u_1[0:-2])) + \
    #                                                 0.5*np.power(dt,2)*f(x[1:-1], t[0])
    
    u[1:-1] = (u_1[1:-1] - B_m*0.5*dt*V(x[1:-1]) + 0.5*C2*(0.5*(q[1:-1] + q[2:])*(u_1[2:] - u_1[1:-1]) - \
                                                           0.5*(q[1:-1] + q[0:-2])*(u_1[1:-1] - u_1[0:-2])) + \
                                                    0.5*np.power(dt,2)*f(x[1:-1], dt))
    
        
    #if U_0 or U_L are None, it indicates Neumann boundary condition
    #otherwise, it's a Dirichlet condition    
    if U_0 is None:
        # Set boundary values du/dn = 0
        # x=0: i-1 -> i+1 since u[i-1]=u[i+1]
        # x=L: i+1 -> i-1 since u[i+1]=u[i-1])
        # u[0] = u_1[0] + dt*V(x[0]) + \
        #         0.5*C2*(0.5*(q[0]+q[1])*(u_1[1] - u_1[0]) - \
        #                 0.5*(q[0]+q[1])*(u_1[0] - u_1[1])) + \
        #         0.5*np.power(dt,2)*f(x[0], t[0])
        u[0] = (u_1[0] - B_m*0.5*dt*V(x[0]) + \
                0.5*C2*(0.5*(q[0]+q[1])*(u_1[1] - u_1[0]) - \
                        0.5*(q[0]+q[1])*(u_1[0] - u_1[1])) + \
                0.5*np.power(dt,2)*f(x[0], dt))
        
    else:
        u[0] = U_0(dt)
        

    if U_L is None:
        # u[-1] = u_1[-1] + dt*V(x[-1]) + \
        #         0.5*C2*(0.5*(q[-1]+q[-2])*(u_1[-2] - u_1[-1]) - \
        #                 0.5*(q[-1]+q[-2])*(u_1[-1] - u_1[-2])) + \
        #         0.5*np.power(dt,2)*f(x[-1], t[0])
        u[-1] = u_1[-1] - B_m*2*dt*V(x[-1]) + \
                0.5*C2*(0.5*(q[-1]+q[-2])*(u_1[-2] - u_1[-1]) - \
                        0.5*(q[-1]+q[-2])*(u_1[-1] - u_1[-2])) + \
                0.5*np.power(dt,2)*f(x[-1], dt)
    else:
        u[-1] = U_L(dt)
        
    if user_action is not None:
        user_action(u, x, dt)
    
    # Switch variables before next step
    u_2[:] = u_1; u_1[:] = u
    
    return u, u_1, u_2, x, dt*2, c, q, C2, B_p, B1, B_m, U_L, U_0, I, V, f



def single_timestep(I, V, f, U_0, U_L, dt, u, u_1, u_2, x, t, c, q, C2, B_p, B1, B_m, user_action=None):
        """
        returns the outputs of a single timestep, depends on previous results and parameters
        calculated with the first step conditioner
        
        @return: u, u_1, u_2, x, t, cpu_time
        """
        
        t0 = time.perf_counter_ns()
        
        #update all inner points at time t[n+1]
        #first is the model for undamped condition
        # u[1:-1] = - u_2[1:-1] + 2*u_1[1:-1] + C2*(0.5*(q[1:-1] + q[2:])*(u_1[2:] - u_1[1:-1]) - 0.5*(q[1:-1] + q[:-2])*(u_1[1:-1] - u_1[:-2])) + np.power(dt,2)*f(x[1:-1], t[n])
        
        u[1:-1] = B1*(B_m*u_2[1:-1] + 4*u_1[1:-1] + C2*((q[1:-1] + q[2:])*(u_1[2:] - u_1[1:-1]) - (q[1:-1] + q[:-2])*(u_1[1:-1] - u_1[:-2])) + 2*np.power(dt,2)*f(x[1:-1], t))
        
        # Insert boundary conditions
        if U_0 is None:
            # Set boundary values
            # x=0: i-1 -> i+1 since u[i-1]=u[i+1] when du/dn=0
            # x=L: i+1 -> i-1 since u[i+1]=u[i-1] when du/dn=0
            # u[0] = - u_2[0] + 2*u_1[0] + \
            #         C2*(0.5*(q[0]+q[1])*(u_1[1] - u_1[0]) - \
            #             0.5*(q[0]+q[1])*(u_1[0] - u_1[1])) + \
            #         np.power(dt,2)*f(x[0], t[n])
            u[0] = B1*(B_m*u_2[0] + 4*u_1[0] + \
                    C2*((q[0]+q[1])*(u_1[1] - u_1[0]) - \
                        (q[0]+q[1])*(u_1[0] - u_1[1])) + \
                    2*np.power(dt,2)*f(x[0], t))
        else:
            u[0] = U_0(t)
        

        if U_L is None:
            # u[-1] = - u_2[-1] + 2*u_1[-1] + \
            #         C2*(0.5*(q[-1]+q[-2])*(u_1[-2] - u_1[-1]) - \
            #             0.5*(q[-1]+q[-2])*(u_1[-1] - u_1[-2])) + \
            #         np.power(dt,2)*f(x[-1], t[n])
            u[-1] = B1*(B_m*u_2[-1] + 4*u_1[-1] + \
                    C2*((q[-1]+q[-2])*(u_1[-2] - u_1[-1]) - \
                        (q[-1]+q[-2])*(u_1[-1] - u_1[-2])) + \
                    2*np.power(dt,2)*f(x[-1], t))
        else:
            u[-1] = U_L(t)
        
        if user_action is not None:
            if user_action(u, x, t):
                return
            
        #switch variables before next step
        u_2[:] = u_1; u_1[:] = u
        
        cpu_time = t0 - time.perf_counter_ns()
        t = t + dt
        
        return u, u_1, u_2, x, t, cpu_time
        
        
        






def solver(I, V, f, c, U_0, U_L, L, dt, C, T, safety_factor=0.7071, user_action=None, attenuation=False):
    """Solve u_tt=c^2*u_xx + f on (0,L)x(0,T]."""  
        
    Nt = int(np.round(T/dt))
    t = np.linspace(0,Nt*dt,num=Nt+1)
    
    if isinstance(c, (float,int)):
        c_max = c        
    elif callable(c):
        c_max = max([c(x_) for x_ in np.linspace(0, L, 1001)])
    
    dx = dt*c_max/(safety_factor*C)
    Nx = int(np.round(L/dx))
    x = np.linspace(0,L,num=Nx+1)
    
    # Treat c(x) as array
    if isinstance(c, (float,int)):
        c = np.zeros(x.shape) + c
    elif callable(c):
        # Call c(x) and fill array c
        c_ = np.zeros(x.shape)
        for i in range(Nx+1):
            c_[i] = c(x[i])
        c = c_
    
    if attenuation is not False:
        if not isinstance(attenuation,(float,int)):
            b = 5.0E-2
        else:
            b = attenuation
    else:
        b = 0.0
        
    q = np.power(c,2)
    B_p = 2 + b*dt
    B1 = 1/B_p
    B_m = b*dt - 2
    C2 = np.power((dt/dx), 2)    # Help variables in the scheme
    
    if f is None or f == 0 :
        f = lambda x, t: 0
    if V is None or V == 0:
        V = lambda x: 0
    if I is None or I == 0:
        I = lambda x: 0
    if U_0 is not None:
        if isinstance(U_0, (float,int)): 
            if U_0 == 0:
                U_0 = lambda t: 0
            else:
                aux = U_0
                U_0 = lambda t: float(aux)
        # else: U_0(t) is a function
    if U_L is not None:
        if isinstance(U_L, (float,int)):
            if U_L == 0:
                U_L = lambda t: 0
            else:
                aux = U_L
                U_L = lambda t: float(aux)
        # else: U_L(t) is a function    

    # import hashlib, inspect
    # data = inspect.getsource(I) + '_' + inspect.getsource(V) + \
    #        '_' + inspect.getsource(f) + '_' + str(c) + '_' + \
    #        ('None' if U_0 is None else inspect.getsource(U_0)) + \
    #        ('None' if U_L is None else inspect.getsource(U_L)) + \
    #        '_' + str(L) + str(dt) + '_' + str(C) + '_' + str(T) + \
    #        '_' + str(safety_factor)
    # data = data.encode()
    # hashed_input = hashlib.sha1(data).hexdigest()
    # if os.path.isfile('.' + hashed_input + '_archive.npz'):
    #     # Simulation is already run
    #     return -1, hashed_input    
     
    u = np.zeros(Nx+1)
    u_1 = np.zeros(Nx+1)
    u_2 = np.zeros(Nx+1)
    t0 = time.perf_counter_ns()
    
    # Load initial condition into u_1
    u_1[:] = I(x)
    
    if user_action is not None:
        user_action(u_1, x, t, 0)    

    # Special formula for first time step
    n = 0
    # u[1:-1] = u_1[1:-1] + dt*V(x[1:-1]) + 0.5*C2*(0.5*(q[1:-1] + q[2:])*(u_1[2:] - u_1[1:-1]) - \
    #                                               0.5*(q[1:-1] + q[0:-2])*(u_1[1:-1] - u_1[0:-2])) + \
    #                                                 0.5*np.power(dt,2)*f(x[1:-1], t[0])
    u[1:-1] = (u_1[1:-1] - B_m*0.5*dt*V(x[1:-1]) + 0.5*C2*(0.5*(q[1:-1] + q[2:])*(u_1[2:] - u_1[1:-1]) - \
                                                  0.5*(q[1:-1] + q[0:-2])*(u_1[1:-1] - u_1[0:-2])) + \
                                                    0.5*np.power(dt,2)*f(x[1:-1], t[0]))
        
    if U_0 is None:
        # Set boundary values du/dn = 0
        # x=0: i-1 -> i+1 since u[i-1]=u[i+1]
        # x=L: i+1 -> i-1 since u[i+1]=u[i-1])
        # u[0] = u_1[0] + dt*V(x[0]) + \
        #         0.5*C2*(0.5*(q[0]+q[1])*(u_1[1] - u_1[0]) - \
        #                 0.5*(q[0]+q[1])*(u_1[0] - u_1[1])) + \
        #         0.5*np.power(dt,2)*f(x[0], t[0])
        u[0] = (u_1[0] - B_m*0.5*dt*V(x[0]) + \
                0.5*C2*(0.5*(q[0]+q[1])*(u_1[1] - u_1[0]) - \
                      0.5*(q[0]+q[1])*(u_1[0] - u_1[1])) + \
                0.5*np.power(dt,2)*f(x[0], t[0]))
        
    else:
        u[0] = U_0(dt)
        

    if U_L is None:
        # u[-1] = u_1[-1] + dt*V(x[-1]) + \
        #         0.5*C2*(0.5*(q[-1]+q[-2])*(u_1[-2] - u_1[-1]) - \
        #                 0.5*(q[-1]+q[-2])*(u_1[-1] - u_1[-2])) + \
        #         0.5*np.power(dt,2)*f(x[-1], t[0])
        u[-1] = u_1[-1] - B_m*2*dt*V(x[-1]) + \
                0.5*C2*(0.5*(q[-1]+q[-2])*(u_1[-2] - u_1[-1]) - \
                        0.5*(q[-1]+q[-2])*(u_1[-1] - u_1[-2])) + \
                0.5*np.power(dt,2)*f(x[-1], t[0])
    else:
        u[-1] = U_L(dt)
        
    if user_action is not None:
        user_action(u, x, t, 1)
    
    # Switch variables before next step
    u_2[:] = u_1; u_1[:] = u
    
    for n in range(1,Nt):
        
        #update all inner points at time t[n+1]
        # u[1:-1] = - u_2[1:-1] + 2*u_1[1:-1] + C2*(0.5*(q[1:-1] + q[2:])*(u_1[2:] - u_1[1:-1]) - 0.5*(q[1:-1] + q[:-2])*(u_1[1:-1] - u_1[:-2])) + np.power(dt,2)*f(x[1:-1], t[n])
        u[1:-1] = B1*(B_m*u_2[1:-1] + 4*u_1[1:-1] + C2*((q[1:-1] + q[2:])*(u_1[2:] - u_1[1:-1]) - (q[1:-1] + q[:-2])*(u_1[1:-1] - u_1[:-2])) + 2*np.power(dt,2)*f(x[1:-1], t[n]))
        
        # Insert boundary conditions
        if U_0 is None:
            # Set boundary values
            # x=0: i-1 -> i+1 since u[i-1]=u[i+1] when du/dn=0
            # x=L: i+1 -> i-1 since u[i+1]=u[i-1] when du/dn=0
            # u[0] = - u_2[0] + 2*u_1[0] + \
            #         C2*(0.5*(q[0]+q[1])*(u_1[1] - u_1[0]) - \
            #             0.5*(q[0]+q[1])*(u_1[0] - u_1[1])) + \
            #         np.power(dt,2)*f(x[0], t[n])
            u[0] = B1*(B_m*u_2[0] + 4*u_1[0] + \
                    C2*((q[0]+q[1])*(u_1[1] - u_1[0]) - \
                        (q[0]+q[1])*(u_1[0] - u_1[1])) + \
                    2*np.power(dt,2)*f(x[0], t[n]))
                
        else:
            u[0] = U_0(t[n+1])
        

        if U_L is None:
            # u[-1] = - u_2[-1] + 2*u_1[-1] + \
            #         C2*(0.5*(q[-1]+q[-2])*(u_1[-2] - u_1[-1]) - \
            #             0.5*(q[-1]+q[-2])*(u_1[-1] - u_1[-2])) + \
            #         np.power(dt,2)*f(x[-1], t[n])
            u[-1] = B1*(B_m*u_2[-1] + 4*u_1[-1] + \
                    C2*((q[-1]+q[-2])*(u_1[-2] - u_1[-1]) - \
                        (q[-1]+q[-2])*(u_1[-1] - u_1[-2])) + \
                    2*np.power(dt,2)*f(x[-1], t[n]))
        else:
            u[-1] = U_L(t[n+1])
        
        if user_action is not None:
            if user_action(u, x, t, n+1):
                break
            
        #switch variables before next step
        u_2[:] = u_1; u_1[:] = u
        
    cpu_time = t0 - time.perf_counter_ns()
    return u, x, t, cpu_time

def test_quadratic():
    """Check that u(x,t)=x(L-x)(1+t/2) is exactly reproduced."""
    
    def u_exact(x, t):
        return x*(L-x)*(1 + 0.5*t)
    
    def I(x):
        return u_exact(x, 0)
    
    def V(x):
        return 0.5*u_exact(x, 0)
    
    def f(x, t):
        return 2*(1 + 0.5*t)*c**2
    
    L = 2.5
    c = 1.5
    C = 0.75
    Nx = 6 # Very coarse mesh for this exact test
    dt = C*(L/Nx)/c
    T = 18
    
    def assert_no_error(u, x, t, n):
        u_e = u_exact(x, t[n])
        diff = np.abs(u - u_e).max()
        tol = 1E-13
        assert diff < tol
        
    solver(I, V, f, c, L, dt, C, T,
    user_action=assert_no_error)

def viz(

    I, V, f, c, U_0, U_L, L, dt, C, T, # PDE paramteres
    umin, umax, # Interval for u in plots
    animate=True, # Simulation with animation?
    tool='matplotlib', # 'matplotlib' or 'scitools'
    solver_function=solver, # Function with numerical algorithm
    folder_name=None,
    fps = 4, # frames per second,
    safety_factor = 1.0,
    attenuation=False,
    ):
    """Run solver and visualize u at each time level."""
    
    import moviepy.editor as mpy
    
    if folder_name is not None:
        os.makedirs('./'+folder_name,exist_ok=True)
        folder_name = './' + folder_name + '/'
    else:
        folder_name = ''
    
    def plot_u_st(u, x, t, n):
        """user_action function for solver."""
        plt.plot(x, u, 'r-',
        xlabel='x', ylabel='u',
        axis=[0, L, umin, umax],
        title='t=%f' % t[n], show=True)
        # Let the initial condition stay on the screen for 2
        # seconds, else insert a pause of 0.2 s between each plot
        time.sleep(2) if t[n] == 0 else time.sleep(0.0)
        plt.savefig(folder_name +'tmp_%04d.png' % n) # for movie making
    
    class PlotMatplotlib:
        def __call__(self, u, x, t, n):
            """user_action function for solver."""
            if n == 0:
                plt.ion()
                self.lines = plt.plot(x, u, 'r-')
                plt.xlabel('x'); plt.ylabel('u')
                plt.axis([0, L, umin, umax])
                plt.legend(['t=%f' % t[n]], loc='lower left')
            else:
                self.lines[0].set_ydata(u)
                plt.legend(['t=%f' % t[n]], loc='lower left')
                plt.draw()
                time.sleep(2) if t[n] == 0 else time.sleep(0.0)
                plt.savefig(folder_name +'tmp_%06d.png' % n) # for movie making
    
    if tool == 'matplotlib':
        import matplotlib.pyplot as plt
        plot_u = PlotMatplotlib()
    elif tool == 'scitools':
        import scitools.std as plt # scitools.easyviz interface
        plot_u = plot_u_st

    # Clean up old movie frames
    for filename in glob.glob(folder_name +'tmp_*.png'):
        os.remove(filename)
        
    # Call solver and do the simulaton
    user_action = plot_u if animate else None
    u, x, t, cpu = solver_function(
    I, V, f, c, U_0, U_L, L, dt, C, T, safety_factor, user_action,attenuation)
    
    # Make video files
    
    image_list = glob.glob(folder_name +'tmp_*.png')
    image_list.sort()
    
    
    codec2ext = dict(libx264='mp4', libvpx='webm', libtheora='ogg') # video formats
    #flv='flv'
    
    clip = mpy.ImageSequenceClip(image_list, int(fps))
    for codec in codec2ext:
        ext = codec2ext[codec]
        clip.write_videofile(folder_name + 'movie.'+ ext, int(fps), codec)
    
    # filespec = 'tmp_%04d.png'
    # movie_program = 'ffmpeg' # or 'avconv'
    # for codec in codec2ext:
    #     ext = codec2ext[codec]
    #     cmd = '%(movie_program)s -r %(fps)d -i %(filespec)s '\
    #     '-vcodec %(codec)s movie.%(ext)s' % vars()
    #     os.system(cmd)
    # if tool == 'scitools':
    #     # Make an HTML play for showing the animation in a browser
    #     plt.movie('tmp_*.png', encoder='html', fps=fps,
    #     output_file='movie.html') 
    
    # AUDIO CLIPS
    # clip = AudioFileClip("my_audiofile.mp3") # or .ogg, .wav... or a video !
    # clip = AudioArrayClip(numpy_array, fps=44100) # from a numerical array
    # clip = AudioClip(make_frame, duration=3) # uses a function make_frame(t)
    
    
    return cpu, u, x, t