import math
import sympy as sp
import mpmath
import numpy as np
import matplotlib.pyplot as plt

#Set the precision
mpmath.mp.dps=50


#Define a symbolic variable
x,y,z=sp.symbols('x y z')
r=3.9

#Define a symbolic expression
f=x**2-1
g=r*x*(1-x)
h=z-f.subs(x,z)*(y-z)/(f.subs(x,y)-f.subs(x,z))

def newton_iterate(start,func,iterations,threshold=0,damping_factor=1.0,show_working=False,return_sequence=False):
    prime=sp.diff(func,x)
    expr_1=x-damping_factor*func/prime
    return banach_fixed_point(start,expr_1,iterations,threshold,show_working,return_sequence)

def banach_fixed_point(start,func,iterations,threshold=0,show_working=False,return_sequence=False):
    sequence=np.empty(iterations+1)
    sequence[0]=start
    expr=sp.lambdify(x,func,'mpmath')
    error=sp.lambdify(x,func-x,'mpmath')
    for i in range(iterations):
        start=expr(start)
        if show_working:
            print(f"In {i}th iterate, x = {start}")
        if abs(error(start))<threshold:
            print("Converged to a root at x = ",start)
            return start
        if return_sequence:
            sequence[i+1]=start
    if return_sequence:
        return sequence
    else:
        return start
def secant_method(start_1,start_2,func,iterations,threshold=0,show_working=False,return_sequence=False):
    sequence=np.empty(iterations+1)
    sequence[0]=start_1
    expr=sp.lambdify(x,func,'mpmath')
    error=sp.lambdify(x,func-x,'mpmath')
    for i in range(iterations):
        if start_1==start_2:
            print("Error: Division by zero")
            return sequence
        start_3=start_2-expr(start_2)*(start_2-start_1)/(expr(start_2)-expr(start_1))
        if show_working:
            print(f"In {i}th iterate, x = {start_3}")
        if abs(error(start_3))<threshold:
            print("Converged to a root at x = ",start_3)
            return start_3
        start_1=start_2
        start_2=start_3
    if return_sequence:
        return sequence
    else:
        return start_3
def banach_fixed_point_vector(start, func, iterations, threshold=0, show_working=False):
    expr = sp.lambdify((y,z), func, 'numpy')
    error_1=sp.lambdify((y,z),func-z,'numpy')
    start=np.array(start,dtype=mpmath.mpf)
    for i in range(iterations):
        result=expr(start[0],start[1])
        start[0]=start[1]
        start[1]=result
        if show_working:
            print(f"In {i}th iterate, x = {start}")
        if np.linalg.norm(error_1(start[0],start[1])) < threshold:
            print(f"Converged to a fixed point at x = {start}")
            return start
    return start
def interval_bisection(func, a, b, threshold=0.001, show_working=False):
    error = sp.lambdify(x, func, 'mpmath')
    c=(a+b)/2
    i=0
    while not (error(c) == 0 or (b - a) / 2 < threshold):
        c = (a + b) / 2
        if np.sign(error(c)) == np.sign(error(a)):
            a = c
        else:
            b = c
        if show_working:
            print(f"In {i}th iterate, x = {c}")
        i+=1
    print(f"Converged to a root at x = {c}")
    return c
# Example usage:

def main():
    threshold = 10 ** -15
    iterations=100

    #start_1 = mpmath.mpf(1)
    #newton_iterate(start_1, f, iterations=iterations, threshold=threshold, damping_factor=1.0, show_working=True)

    lower_bound = mpmath.mpf(0)
    upper_bound = mpmath.mpf(2)
    epsilon=mpmath.mpf(0.1)
    # print(interval_bisection(f,lower_bound,upper_bound,threshold=threshold,show_working=True))
    #sequence = banach_fixed_point(0.75, g, iterations=iterations, threshold=threshold, show_working=True, return_sequence=True)
    sequence=secant_method(-1+epsilon,1+epsilon,f,iterations=iterations,threshold=threshold,show_working=True,return_sequence=True)
    horizontal = np.linspace(0, iterations+1, iterations+1)
    #plt.plot(horizontal,sequence)
    #plt.show()

    threshold = 1e-6
    start_2 = [0.5, 0.6]
    # banach_fixed_point_vector(start_2, h, iterations=100, threshold=threshold, show_working=True)

if __name__ == '__main__':
    main()

