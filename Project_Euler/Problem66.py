# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 09:37:12 2023

@author: leoda
"""

import math

def is_square(x):
    return x==math.floor(math.sqrt(x))**2

def diophantine(repetitions):
    maximum=0
    max_value=0
    for j in range(1,repetitions+1,1):
        value1=solve_diophantine1(j)
        value2=solve_diophantine2(j)
        if value1!=value2:
            print('for '+str(j)+' we have a problem with '+str(value1)+'!='+str(value2))
            
        if value1>max_value:
            maximum=j
            max_value=value1
    return maximum

def solve_diophantine1(D):
    if is_square(D):
        return 0
    x,y=0,0
    x_increment=math.floor(math.sqrt(D))
    square=0
    while x**2!=1+square:
        y+=1
        square+=D*(2*y-1)
        x+=x_increment
        while x**2<1+square:
            x+=1
    print (str(x)+'^2 - '+str(D)+' * '+str(y)+'^2 = 1')
    return x

def solve_diophantine2(D):
    if is_square(D):
        return 0
    x,y=D,1
    counter=0
    while x**2-D*y**2!=1 and counter<10:
        z=D*y+x
        y+=x
        x=z
        print(str(x)+'/'+str(y))
        counter+=1
    return x

print(solve_diophantine2(3))