import math
def hcf(x,y):
    if x<y:
        return hcf(y,x)
    if y==0:
        return x
    z=x%y
    d=math.floor(x/y)
    print(x,"=",d,"*",y,"+",z)
    return hcf(y,z)
x1=int(input('First number: '))
x2=int(input('Second number: '))
print(hcf(x1,x2))
