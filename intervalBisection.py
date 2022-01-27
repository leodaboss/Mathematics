import math

def f(x):
    return math.cos(x)

def Mid(a,b):
    return (a+b)/2

def secantMid(a,b):
    return (a*f(b)-b*f(a))/(f(b)-f(a))

def secantBisection(a,b, tol):
    return generalBisection(a,b,tol,False)

def bisection(a,b, tol):
    return generalBisection(a,b,tol,True)

def generalBisection(a,b,tol,isNormal):
    #initialise
    bottom=a
    top=b
    counter=0
    
    #loop repeats until bottom and top are close enough together
    while top-bottom>=tol:
        
        #works out whether to calculate secant or normal midpoint
        if isNormal:midpoint=Mid(top,bottom)
        else: midpoint=secantMid(top,bottom)
        
        #adds 1 to number of steps
        counter+=1

        #in case we somehow manage to get right answer
        if f(midpoint)==0:break

        #general algorithm
        if f(midpoint)*f(bottom)<0:top=midpoint
        else: bottom=midpoint
    return [midpoint, counter]

def main():
    a=1/2
    b=3
    c=bisection(a,b,0.0001)
    d=secantBisection(a,b,0.0001)
    print(c)
    print(d)

main()
#from this python experiment we can observe that the secant method is much more
#precise and it achieves convergence in fewer steps
#at least in this case
    
