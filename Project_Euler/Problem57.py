# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 18:57:27 2023

@author: leoda
"""
import math


def continued_fraction(repetitions):
    counter=1
    a=3
    b=2
    difference=0
    while counter<=repetitions:
        if math.floor(math.log10(a))-math.floor(math.log10(b))==1:
            difference+=1
            print(str(a)+'/'+str(b))
        counter+=1
        b+=a
        c=a
        a=2*b-c
    print(difference)
    
continued_fraction(1000)