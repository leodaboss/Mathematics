# -*- coding: utf-8 -*-
"""
Created on Sat Sep 23 09:22:51 2023

@author: leoda
"""
import math

tens=['twenty','thirty','forty','fifty','sixty','seventy','eighty','ninety']
teens=['ten','eleven','twelve','thirteen','fourteen','fifteen','sixteen'
           ,'seventeen','eighteen','nineteen']
digits=['zero','one','two','three','four','five','six','seven','eight','nine']
hundred=len('hundred')
thousand=len('thousand')
zero=len('zero')
AND=len('and')

def letter_count():
    sum_teens=sum(len(s) for s in teens)
    sum_digits=sum(len(s) for s in digits)
    print(sum_teens)
def letter_count(number):
    if number>1000 or number<=0:
        return None
    if number==1000:
            return thousand
    if number<100:
        if number<10:
            return len(digits[number])
        if number>=10 and number<20:
            return len(teens[number-10])
        remainder=number%10
        divisor=len(tens[math.floor(number/10)-2])
        if remainder==0:return divisor
        return divisor+len(digits[remainder])
        
    remainder=number%100
    divisor=len(digits[math.floor(number/100)])
    if remainder==0:
        return hundred+divisor
        
    return divisor+hundred+AND+letter_count(remainder)


sum_numbers=0
for i in range(1,1001):
    count=letter_count(i)
    sum_numbers+=count
    print(str(i)+' has letter count '+str(count))
    
print(sum_numbers)    