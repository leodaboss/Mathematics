import numpy as np
import math
import matplotlib.pyplot as plt
import scipy

# model details
# initial conditions
a = 1
mu = 1
dimension = 1
T = 1  # total time
L = 25.0  # How far it extends in space

variance_init = 1.5  # Variance of initial distribution
sd_limit_init = 4  # How far we compute the initial configuration away in standard deviations
variance_kernel = 0.5  # Variance of the convolution kernel
sd_limit_kernel = 2  # How far we compute the kernel away in standard deviations
radius_kernel = sd_limit_kernel * variance_kernel  # Radius we compute the kernel up to
radius_init = sd_limit_init * variance_init  # Radius we compute the initial distribution up to
f = lambda x, variance: math.exp(-(sum([z ** 2 for z in x])) / (2 * variance)) / math.sqrt(
    (2 * math.pi * variance) ** dimension)
f_init = lambda x: f(x, variance_init)
f_kernel = lambda x: f(x, variance_kernel)

# accuracy of approximation+ bounds
dx = .1  # space step
dt = .001  # time step

# scaling
f_init_scaled = lambda x: f_init([z * dx for z in x])
f_kernel_scaled = lambda x: f_kernel([z * dx for z in x])
radius_init_scaled = int(radius_init / dx)
radius_kernel_scaled = int(radius_kernel / dx)
diameter_kernel_scaled = 2 * radius_kernel_scaled + 1
radius_grid = int(L / dx)  # size of 2d grid
diameter_grid = 2 * radius_grid + 1  # since it goes both positive and negative
iterations = int(T / dt)  # number of iterations

# speed of wave
wave_speed = 2 * math.sqrt(mu)
# correct for inefficiencies
wave_speed *= 0.95

# graphs
number_plots = 9
step_plot = iterations // number_plots


class one_dimensional():
    f_init_scaled = lambda x: dx * f_init([z * dx for z in x])
    f_kernel_scaled = lambda x: dx * f_kernel([z * dx for z in x])

    def convolution_matrix(g, size):
        return [g([i - size]) for i in range(2 * size + 1)]

    def laplacian(Z):
        Ztop = Z[0:-2]
        Zbottom = Z[2:]
        Zcenter = Z[1:-1]
        return Ztop + Zbottom - 2 * Zcenter

    def convolution(V, conv_matrix, dirac=False):
        if not dirac:
            return scipy.ndimage.convolve1d(V, weights=conv_matrix, mode='reflect')
        return np.array(V)

    # This returns a plot of the graph of U
    def show_patterns(U, ax=None, axis_off=True):
        ax.plot(U)
        if axis_off:
            ax.set_axis_off()

    # This function intialises the initial condition
    def initialise_specific(radius_initial, g, zero_point=radius_grid):
        radius_initial = min(radius_grid, radius_initial)  # truncate in case given wrong values
        diameter_initial = 2 * radius_initial + 1
        V = np.zeros(diameter_grid)
        centre = [g([i - radius_initial]) for i in range(diameter_initial)]
        V[zero_point[0] - radius_initial:zero_point[0] + radius_initial + 1] += centre
        return V

    def initialise_heaviside(min_point, max_point):
        max_point = min(diameter_grid, max_point)  # truncate in case given wrong values
        min_point = max(0, min_point)
        V = np.zeros(diameter_grid)
        V[min_point:max_point] += np.ones(max_point - min_point)
        return V

    def update_explicit_euler_equation(t, V, conv_matrix, dirac):
        return one_dimensional.update_explicit_euler_k_equation(t, V, conv_matrix, dirac)

    # We use Runge Kutta 4 to solve the autonomous system
    def update_explicit_euler_k_equation(t, V, conv_matrix, dirac):
        k_1 = one_dimensional.f(t, V, conv_matrix, dirac)
        k_2 = one_dimensional.f(t + dt / 2, V + dt / 2 * k_1, conv_matrix, dirac)
        k_3 = one_dimensional.f(t + dt / 2, V + dt / 2 * k_2, conv_matrix, dirac)
        k_4 = one_dimensional.f(t + dt, V + dt * k_3, conv_matrix, dirac)
        W = V + dt / 6 * (k_1 + 2 * k_2 + 2 * k_3 + k_4)
        W[:, 0] = W[:, 1]
        W[:, -1] = W[:, -2]
        return W

    # we consider autonomous ODE y'(t)=f(t,y), gives back vector f(t,y)
    def f(t, V, conv_matrix, dirac):
        sumV = [sum([V[j, i] for j in range(len(V[:, 0]))]) for i in range(len(V[0, :]))]
        conv = one_dimensional.convolution(sumV, conv_matrix, dirac)
        W = np.zeros(V.shape)
        for i in range(len(V[:, 0])):
            # We compute the Laplacian of V.
            deltaV = one_dimensional.laplacian(V[i, :]) / dx ** 2
            # We take the values of V inside the grid.
            Vc = V[i, 1:-1]
            # We update the variables.
            W[i, 1:-1] = a * deltaV + mu * Vc * (1 - conv[1:-1])
            # Neumann conditions on Boundary
            W[i, 0] = V[i, 1]
            W[i, -1] = V[i, -2]
        return W


class two_dimensional():
    f_init_scaled = lambda x: dx ** 2 * f_init([z * dx for z in x])
    f_kernel_scaled = lambda x: dx ** 2 * f_kernel([z * dx for z in x])

    def convolution_matrix(g, size):
        return [[g([i, j] - size * np.ones(2)) \
                 for i in range(2 * size + 1)] for j in range(2 * size + 1)]

    def laplacian(Z):
        Ztop = Z[0:-2, 1:-1]
        Zleft = Z[1:-1, 0:-2]
        Zbottom = Z[2:, 1:-1]
        Zright = Z[1:-1, 2:]
        Zcenter = Z[1:-1, 1:-1]
        return Ztop + Zleft + Zbottom + Zright - \
            4 * Zcenter

    def convolution(V, conv_matrix, dirac=False):
        if not dirac:
            return scipy.ndimage.convolve1d(V, weights=conv_matrix, mode='reflect')
        return np.array(V)

    # This returns a plot of the graph of U
    def show_patterns(U, ax=None, axis_off=True):
        ax.imshow(U, cmap=plt.cm.copper,
                  interpolation='bilinear',
                  extent=[-1, 1, -1, 1])
        if axis_off:
            ax.set_axis_off()
        # This function intialises the initial condition

    def initialise_specific(radius_initial, g, zero_point=[radius_grid, radius_grid]):
        radius_initial = min(radius_grid, radius_initial)  # truncate in case given wrong values
        radius_initial_array = radius_initial * np.ones(2)
        diameter_initial = 2 * radius_initial + 1
        V = np.zeros(tuple(diameter_grid * np.ones(2, dtype=int)))
        centre = [[g([i, j] - radius_initial_array) for i in range(diameter_initial)] for j in range(diameter_initial)]
        V[zero_point[0] - radius_initial:zero_point[0] + radius_initial + 1, \
        zero_point[1] - radius_initial:zero_point[1] + radius_initial + 1] += centre
        return V
        # We use Runge Kutta 4 to solve the autonomous system

    def update_explicit_euler_equation(t, V, conv_matrix, dirac):
        k_1 = one_dimensional.f(t, V, conv_matrix, dirac)
        k_2 = one_dimensional.f(t + dt / 2, V + dt / 2 * k_1, conv_matrix, dirac)
        k_3 = one_dimensional.f(t + dt / 2, V + dt / 2 * k_2, conv_matrix, dirac)
        k_4 = one_dimensional.f(t + dt, V + dt * k_3, conv_matrix, dirac)
        W = V + dt / 6 * (k_1 + 2 * k_2 + 2 * k_3 + k_4)
        # Neumann conditions on Boundary
        W[:, 0] = W[:, 1]
        W[:, -1] = W[:, -2]
        for Z in W:
            Z[0] = Z[1]
            Z[-1] = Z[-2]
        W[0] = W[1]
        W[-1] = W[-2]
        return W

    # we consider autonomous ODE y'(t)=f(t,y), gives back vector f(t,y)
    def f(t, V, conv_matrix, dirac):
        conv = two_dimensional.convolution( conv_matrix, dirac)
        W = np.zeros(V.shape)
        # We compute the Laplacian of V.
        deltaV = two_dimensional.laplacian(V) / dx ** 2
        # We take the values of V inside the grid.
        Vc = V[1:-1, 1:-1]
        # We update the variables.
        W[1:-1, 1:-1] = a * deltaV + mu * Vc * (1 - conv[1:-1, 1:-1])
        # Neumann conditions on Boundary
        W[:, 0] = W[:, 1]
        W[:, -1] = W[:, -2]
        for Z in W:
            Z[0] = Z[1]
            Z[-1] = Z[-2]
        W[0] = W[1]
        W[-1] = W[-2]
        return W


def main1():
    # U=initialise_specific(radius_initial=radius_init_scaled,g=f_init_scaled)
    U = one_dimensional.initialise_heaviside()
    fig, axes = plt.subplots(3, 3, figsize=(8, 8))
    # We simulate the PDE with the finite difference method.
    for i in range(iterations):

        # We plot the state of the system at
        # 9 different times.
        if i % step_plot == 0 and i < number_plots * step_plot:
            ax = axes.flat[i // step_plot]
            one_dimensional.show_patterns(U, ax=ax)
            ax.set_title(f'$t={i * dt:.2f}$')
            print('well done')
        U = one_dimensional.update_explicit_euler_1_equation(U)
    fig.show()
    return


def main2(k, dirac):
    # U=initialise_specific(radius_initial=radius_init_scaled,g=f_init_scaled)
    interval = int(diameter_grid / 2 / k)
    shift_wave = None if wave_speed == 0.0 else int(dx / (dt * wave_speed))
    U = np.zeros(shape=(k, diameter_grid))
    for i in range(k):
        U[i, :] = one_dimensional.initialise_heaviside(i * interval, (i + 1) * interval)
    conv_matrix = one_dimensional.convolution_matrix(g=one_dimensional.f_kernel_scaled, size=radius_kernel_scaled)
    fig, axes = plt.subplots(3, 3, figsize=(8, 8))
    # We simulate the PDE with the finite difference method.
    for i in range(iterations):

        # We plot the state of the system at
        # 9 different times.
        if i % step_plot == 0 and i < number_plots * step_plot:
            ax = axes.flat[i // step_plot]
            V = np.zeros(len(U[0, :]))
            length = len(U[:, 0])
            for j in range(length):
                end = (j == length - 1)
                V = V + U[j, :]
                one_dimensional.show_patterns(V, ax=ax, axis_off=end)
            ax.set_title(f'$t={i * dt:.2f}$')
            print(i * dt)
            print('well done')
        if shift_wave != None and i % shift_wave == 0:
            for j in range(len(U)):
                U[j, :] = np.append(U[j, 1:], 0)

        U = one_dimensional.update_explicit_euler_k_equation(i * dt, U, conv_matrix, dirac)
    fig, ax=plt.subplots(1,1,figsize=(8,8))
    one_dimensional.show_patterns(U, ax=ax)
    plt.show()
    return


# main1()
main2(k=6, dirac=False)
