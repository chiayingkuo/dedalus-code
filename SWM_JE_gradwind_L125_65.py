import numpy as np
from mpi4py import MPI
import time

from dedalus import public as de
from dedalus.extras import flow_tools

import logging
logger = logging.getLogger(__name__)


# Parameters (all dimensions are in m and sec)
Nx, Ny = (512, 256)
basinscale = 6000.e3
Lx, Ly = (basinscale, basinscale*Ny/Nx)Nx, Ny = (512, 256)


# Create bases and domain
x_basis = de.Fourier('x', Nx, interval=(-Lx*3/4, Lx/4), dealias=3/2)
y_basis = de.Chebyshev('y', Ny, interval=(-Ly/2, Ly/2), dealias=3/2)
domain = de.Domain([x_basis, y_basis], grid_dtype=np.float64)


# Define Variables
problem = de.IVP(domain, variables=['u','v','h','uy','vy'])
problem.meta[:]['y']['dirchlet'] = True

N0 = 0.65               # amplitude
L  = 125e3              # length scale
H  = 0.8                 # mean depth 80 cm
g  = 9.8                 # gravity
f0 = 5.9318e-05        # Coriolis parameter
LR = np.sqrt(g*H)/f0  # deformation radius
beta0 = 2.0912e-11    # Rossby parameter
mu    = 1.e-8           # horizontal viscosity
gamma = 0.              # Rayleigh bottom friction

problem.parameters['N0'] = N
problem.parameters['L'] = L
problem.parameters['LR'] = LR
problem.parameters["H"]= H
problem.parameters['g'] = g
problem.parameters['f0'] = f0
problem.parameters['beta0'] = beta0
problem.parameters["mu"]= mu
problem.parameters["gamma"]= gamma

f = domain.new_field()
f['g'] = f0 + beta0*y
f.meta['x']['constant'] = True
problem.parameters['f'] = f


# Define Laplacian and Jacobian operators
problem.substitutions['L1(a)']  = "  d(a,x=2) + d(a,y=2) "
problem.substitutions['J(a,b)'] = "  dx(a)*dy(b) - dy(a)*dx(b) "
problem.substitutions['HD(a)']  = "  mu*L1(L1(a)) "


# Define Governing Equations
problem.add_equation("dt(h) + H*dx(u) + H*vy = -dx(u*h) - dy(v*h)")
problem.add_equation("dt(u) + g*dx(h) - mu*(dx(dx(u))+dy(uy)) + gamma*u - f*v = -u*dx(u) - v*uy ")
problem.add_equation("dt(v) + g*dy(h) - mu*(dx(dx(v))+dy(vy)) + gamma*v + f*u = -u*dx(v) - v*vy ")
problem.add_equation(" uy - dy(u) = 0 ")
problem.add_equation(" vy - dy(v) = 0 ")


# Meridional Boundary Conditions
problem.add_bc("left(uy)  = 0")
problem.add_bc("right(uy) = 0")
problem.add_bc("left(v)   = 0")
problem.add_bc("right(v)  = 0")


# Timestepping
ts = de.timesteppers.RK443

# Initial Value Problem
solver =  problem.build_solver(ts)


# Initial condition (put a gaussian pump at (x0,y0))
h = solver.state['h']
xm, ym = np.meshgrid(x,y, indexing='ij')
x0 = 0.e3
y0 = 0.e3
h['g'] = N0*np.exp(-((xm-x0)**2+(ym-y0)**2)/(L**2))

u = solver.state['u']
u['g'] = -0.5*(-f0*(ym-y0) + (ym-y0)*np.sqrt(f0**2 - 8*N0*g/L**2*np.exp(-( (xm-x0)**2 + (ym-y0)**2 )/L**2)));
v = solver.state['v']
v['g'] = 0.5*(-f0*(xm-x0) + (xm-x0)*np.sqrt(f0**2 - 8*N0*g/L**2*np.exp(-( (xm-x0)**2 + (ym-y0)**2 )/L**2)));


# Analysis
snap = solver.evaluator.add_file_handler('SW_gradwind_2yr_N65_L125', sim_dt=3600*24., max_writes=1000)
snap.add_system(solver.state, layout='g')

# Integration parameters
solver.stop_sim_time = 731.* 86400.
solver.stop_wall_time = np.inf
solver.stop_iteration = np.inf


# CFL
dt = 0.1*Lx/Nx    # initial time step<
CFL = flow_tools.CFL(solver, initial_dt=dt, cadence=10, safety=2,
                         max_change=1.2, min_change=0.5, max_dt=3*dt)
flow = flow_tools.GlobalFlowProperty(solver, cadence=10)

# Main loop
try:
    logger.info('Starting loop')
    start_time = time.time()
    while solver.ok:
        dt = solver.step(dt, trim=True)
        if (solver.iteration-1) % 10 == 0:
            logger.info('Iteration: %i, Time: %e, dt: %e' %(solver.iteration, solver.sim_time, dt))
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    end_time = time.time()
    logger.info('Iterations: %i' %solver.iteration)
    logger.info('Sim end time: %f' %solver.sim_time)
    logger.info('Run time: %.2f sec' %(end_time-start_time))
    logger.info('Run time: %f cpu-hr' %((end_time-start_time)/60/60 *domain.dist.comm_cart.size))
