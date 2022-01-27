import math

#here are the functions we use
def linear(x):
    return x

#here are the functions to determine the midpoint
def secantMid(a,b,func):
    return (a*func(b)-b*func(a))/(func(b)-func(a))

def Mid(a,b,func):
    return (func(a)+func(b))/2

#here are the bisection algorithms
def secant_bisection(bottom, top, tol,funct):
    return generalised_bisection(bottom,top,tol,funct,secantMid,funct)[0]
def bisection(bottom, top, tol,func):
    return generalised_bisection(bottom,top,tol,func,Mid,linear)[0]

#here is the generalised bisection algorithm upon which all others rely on
def generalised_bisection(bottom, top, tol,func,mid,midfunc):
    #initialise
    
    counter=0
    #loop repeats until bottom and top are close enough together
    while top-bottom>=tol:
        #works out midpoint
        midpoint=mid(top,bottom,midfunc)
        value=func(midpoint)
        
        #adds 1 to number of steps
        counter+=1

        #in case we somehow manage to get right answer
        if func(midpoint)==0:break

        #general algorithm
        if value*func(bottom)<0:top=midpoint
        else: bottom=midpoint
        print(str(bottom)+':'+str(top))
    return [midpoint, counter]

#here is test code
def main():
    a=1/2
    b=3
    tol=0.00001
    c=bisection(a,b,tol,math.cos)
    print(2*c)
    print()
    #d=secant_bisection(a,b,tol,math.cos)
    #print(2*d)
    
main()
#from this python experiment we can observe that the secant method is much more
#precise and it achieves convergence in fewer steps
#at least in this case
    
