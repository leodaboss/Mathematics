import math

#defining the functions and their derivaties
def f1(x):
    return math.sin(x)*math.sinh(x)
def f2(x):
    return x* math.cos(x)
def f3(x):
    return x-math.sin(x)

def Df1(x):
    return math.sin(x)*math.cosh(x)+math.cos(x)*math.sinh(x)
def Df2(x):
    return math.cos(x)-x*math.sin(x)
def Df3(x):
    return 1-math.cos(x)

#putting all the functions in a nice list
funcs=[f1,f2,f3]
Dfuncs=[Df1,Df2,Df3]

#iterative algorithm for Newton's method
def newton_iteration(initial, tol, func,Dfunc):
    #initialises x and the value of the function and counts the steps
    x=initial
    value=func(x)
    counter=0
    
    #while a certain threshold isn't reached the algorithm continues
    while abs(value)>=tol:
        #initialises new values
        value=func(x)
        valueDeriv=Dfunc(x)
        counter+=1
        
        #exceptional break if the derivative is 0 then we are done
        if valueDeriv==0:break
        
        #changes the new value of x to be the new guess
        x-=value/valueDeriv
    return [x,counter]


def main():
    #defining the initial step and the tolerance
    initial=0.5
    tol=0.0001
    
    #for each of the functions it we apply Newton's root finding algorithm
    for i in range(3):
        root=newton_iteration(initial, tol, funcs[i],Dfuncs[i])
        print(root[0])
    

main()
