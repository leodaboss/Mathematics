# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 19:11:47 2023

@author: leoda
"""
import math

def digit_sum(x):
    sum=0
    while x>0:
        sum+=x%10
        x=x//10
    return sum

def powerful_digitsum(repetitions):
    max_digit=0
    for i in range(repetitions):
        current=1
        for j in range(repetitions-1):
            current*=i
            current_digit_sum=digit_sum(current)
            #print(str(i)+'^'+str(j+1)+'='+str(current)+' with digit sum '+str(current_digit_sum))
            if current_digit_sum>max_digit:
                max_digit=current_digit_sum
                print(str(i)+'^'+str(j+1)+'='+str(current)+' with digit sum '+str(current_digit_sum))
    print(max_digit)
        
    
powerful_digitsum(100)
#for i in range(100):
#    print(str(i)+' has digit sum '+ str(digit_sum(i)))