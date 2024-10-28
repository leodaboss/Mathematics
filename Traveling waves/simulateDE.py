import numpy as np
import matplotlib.pyplot as plt

plt.show()
# accuracy of approximation
size=100 # size of 2d grid
dx=2./size # space step

T=20.0 # total time
dt=.001 # time step
n=int(T/dt) #number of iterations

#details of model
a=1
mu=1

def laplacian(Z):
    Ztop = Z[0:-2, 1:-1]
    Zleft = Z[1:-1, 0:-2]
    Zbottom = Z[2:, 1:-1]
    Zright = Z[1:-1, 2:]
    Zcenter = Z[1:-1, 1:-1]
    return (Ztop + Zleft + Zbottom + Zright -
            4 * Zcenter) / dx**2

def show_patterns(U, ax=None):
    ax.imshow(U, cmap=plt.cm.copper,
              interpolation='bilinear',
              extent=[-1, 1, -1, 1])
    ax.set_axis_off()
U=np.zeros((size,size))
fig, axes = plt.subplots(3, 3, figsize=(8, 8))
step_plot = n // 9
# We simulate the PDE with the finite difference
# method.
for i in range(n):
    # We compute the Laplacian of u.
    deltaU = laplacian(U)
    # We take the values of u inside the grid.
    Uc = U[1:-1, 1:-1]
    # We update the variables.
    U[1:-1, 1:-1]= \
        Uc + dt * (a * deltaU + mu * Uc * (1- Uc))

    # We plot the state of the system at
    # 9 different times.
    if i % step_plot == 0 and i < 9 * step_plot:
        ax = axes.flat[i // step_plot]
        show_patterns(U, ax=ax)
        ax.set_title(f'$t={i * dt:.2f}$')
        print('well done')
fig, ax = plt.subplots(1, 1, figsize=(8, 8))
show_patterns(U, ax=ax)