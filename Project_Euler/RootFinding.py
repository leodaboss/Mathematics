import math

class RootFinding():
    @staticmethod
    def get_linear():
        return lambda x: x
    @staticmethod
    def get_middle():
        return lambda x,y,z:(z(x)+z(y))/2

    # here are the functions to determine the midpoint
    @staticmethod
    def secant_mid(a, b, func):
        return (a * func(b) - b * func(a)) / (func(b) - func(a))

    # here are the bisection algorithms
    @staticmethod
    def secant_bisection(bottom, top, tol, funct):
        return RootFinding.generalised_bisection(bottom, top, tol, funct, RootFinding.secant_mid, funct)[0]

    @staticmethod
    def bisection(bottom, top, tol, funct):
        return RootFinding.generalised_bisection(bottom, top, tol, funct, RootFinding.get_middle(),
                                                 RootFinding.get_linear())[0]

    # here is the generalised bisection algorithm upon which all others rely on
    @staticmethod
    def generalised_bisection(bottom, top, tol, func, mid, midfunc):
        if func(bottom) == 0: return [bottom, 0]
        if func(top) == 0: return [top, 0]
        if func(top) * func(bottom) > 0:
            print('Not really possible')
            return
        # initialise

        counter = 0
        # loop repeats until bottom and top are close enough together
        while top - bottom >= tol:
            # initialises new values
            midpoint = mid(top, bottom, midfunc)
            midpoint_value = func(midpoint)
            bottom_value = func(bottom)
            counter += 1

            # exceptional break if we get the right answer and are done
            if func(midpoint) == 0: break

            # showing computation if wanted
            # print('['+str(bottom)+' : '+str(midpoint)+' : '+str(top)+']')
            # print('bottom has value:'+str(bottom_value))
            # print('midpoint has value:'+str(midpoint_value))

            # general algorithm
            if midpoint_value * bottom_value < 0:
                top = midpoint
            else:
                bottom = midpoint

        return [midpoint, counter]

    # iterative algorithm for Newton's method
    @staticmethod
    def newton_iteration(initial, tol, func, Dfunc):
        # initialises x and the value of the function and counts the steps
        x = initial
        value = func(x)
        counter = 0

        # while a certain threshold isn't reached the algorithm continues
        while abs(value) >= tol:
            # initialises new values
            value = func(x)
            value_deriv = Dfunc(x)
            counter += 1

            # exceptional break if the derivative is 0 then we are done
            if value_deriv == 0:
                break

            # changes the new value of x to be the new guess
            x -= value / value_deriv
        return [x, counter]


# defining the functions and their derivatives
def f1(x):
    return math.sin(x) * math.sinh(x)


def f2(x):
    return x * math.cos(x)


def f3(x):
    return x - math.sin(x)


def fD1(x):
    return math.sin(x) * math.cosh(x) + math.cos(x) * math.sinh(x)


def fD2(x):
    return math.cos(x) - x * math.sin(x)


def fD3(x):
    return 1 - math.cos(x)


# putting all the functions in a nice list
funcs = [f1, f2, f3]
Dfuncs = [fD1, fD2, fD3]




def get_pi_1(tol):
    top = 4
    bottom = 0
    # unique solution of cos(x)=0 between 4 and 0 is pi/2
    return 2 * RootFinding.bisection(bottom, top, tol, math.cos)


def get_pi_2(tol):
    top = 4
    bottom = 2
    # unique solution of sin(x)=0 between 2 and 4 is pi
    return RootFinding.bisection(bottom, top, tol, math.sin)


def main():
    # defining the initial step and the tolerance
    initial = 0.5
    tol = 0.0001

    # for each of the functions it we apply Newton's root finding algorithm
    for i in range(3):
        root = RootFinding.newton_iteration(initial, tol, funcs[i], Dfuncs[i])
        print(root[0])
    tol = 0.001
    a, b = 0, 4
    c = get_pi_1(tol)
    # d=get_pi_2(tol)
    # e=secant_bisection(a,b,tol,math.cos)
    print(c)
    # print(d)
    # print(2*e)
main()
