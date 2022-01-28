import math
def hcf(x,y):
    if x<y:
        return hcf_helper(y,x)
    return hcf_helper(x,y)
def hcf_helper(x,y):
    if y==0:
        return x
    z=x%y
    
    #extra stuff if we want to show the computation
    d=math.floor(x/y)
    print(x,"=",d,"*",y,"+",z)
    
    return hcf(y,z)
x1=int(input('First number: '))
x2=int(input('Second number: '))
print(hcf(x1,x2))
