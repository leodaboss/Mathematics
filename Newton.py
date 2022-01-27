import math

def f1(x):
    return math.sin(x)*math.sinh(x)
def f2(x):
    return x* math.cos(x)
def f3(x):
    return x-math.sin(x)

def f1Deriv(x):
    return math.sin(x)*math.cosh(x)+math.cos(x)*math.sinh(x)
def f2Deriv(x):
    return math.cos(x)-x*math.sin(x)
def f3Deriv(x):
    return 1-math.cos(x)

def f(x, number):
    if number==1:return f1(x)
    elif number==2:return f2(x)
    return f3(x)
def fDeriv(x, number):
    if number==1:return f1Deriv(x)
    elif number==2:return f2Deriv(x)
    return f3Deriv(x)

def NewtonIteration(initial, tol, number):
    x=initial
    value=f(x, number)
    counter=0
    while abs(value)>=tol:
        valueDeriv=fDeriv(x, number)
        counter+=1
        if valueDeriv==0:break
        x-=value/valueDeriv
        value=f(x, number)
    return [x,counter]

def main():
    initial=0.5
    root=NewtonIteration(initial, 0.0001, 1)
    print(root)
    root=NewtonIteration(initial, 0.0001, 2)
    print(root)
    root=NewtonIteration(initial, 0.0001, 3)
    print(root)

main()
